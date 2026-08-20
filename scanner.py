# -*- coding: utf-8 -*-
"""
NSE Momentum v5.3    Main Scanner
Single pass through 504 stocks. No double scoring.
Outputs 3-tier report + sends combined HTML email.
Run: python scanner.py (or double-click run_scanner.bat)
"""

import sys, time, logging, json, argparse
from datetime import datetime
from pathlib import Path

# [2026-08-18] Windows consoles default to cp1252, which can't encode the
# Unicode box-drawing/checkmark characters used in a few log/print
# statements elsewhere in this pipeline (market_breadth_agent.py's
# dashboard, data_fetcher.py's CMP validation message) -- crashed every
# local Windows run at that point with UnicodeEncodeError. Never surfaced
# on the GitHub Actions cloud runner (Linux, UTF-8 locale by default), only
# when running locally to diagnose/regenerate a report. errors="replace"
# rather than a hard requirement, so an unrelated console quirk can never
# be why a real data source failure goes uninvestigated.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data_fetcher import (fetch_batch_ohlcv, get_market_context, BhavcopyFetcher,
                           validate_cmp_vs_bhavcopy, get_last_trading_day)
from orchestrator import AgentOrchestrator
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from emailer      import send_email_report
from trade_logger import init_tables
from sanity_gate  import (run_sanity_gate, send_sanity_alert_email,
                           send_data_freshness_alert_email, MIN_OHLCV_COVERAGE_PCT)
from market_calendar.staleness_check import get_code_provenance

BASE_DIR = Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs";    LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "data";    DATA_DIR.mkdir(exist_ok=True)
TODAY    = datetime.today().strftime("%Y%m%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"scanner_{TODAY}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _lookup_symbol_master_names(tickers_ns: list) -> dict:
    """
    P1-07: best-effort company-name lookup for held tickers being folded
    into the scan universe (see run_scan). Reads the nse_symbol_master table
    published nightly by publish_symbol_master.py. Non-fatal on any failure
    — a held ticker still gets scanned even if this returns {} (it just
    displays as its own ticker instead of a company name, same as any other
    lookup failure elsewhere in this pipeline per P1-06).
    """
    bare = {t: t[:-3] if t.upper().endswith((".NS", ".BO")) else t for t in tickers_ns}
    try:
        from turso_sync import get_client
        client = get_client()
    except SystemExit as e:
        log.warning(f"      Symbol master lookup skipped (non-fatal): {e}")
        return {}
    try:
        placeholders = ",".join("?" * len(bare))
        rs = client.execute(
            f"SELECT symbol, company_name FROM nse_symbol_master WHERE symbol IN ({placeholders})",
            list(bare.values()),
        )
        by_bare = {row[0]: row[1] for row in rs.rows if row[1]}
        return {ticker_ns: by_bare[bare_sym] for ticker_ns, bare_sym in bare.items() if bare_sym in by_bare}
    except Exception as e:
        log.warning(f"      Symbol master lookup failed (non-fatal): {e}")
        return {}
    finally:
        client.close()


def find_stale_tickers(stock_data: dict, expected_trading_date) -> list:
    """
    [2026-08-18] Returns the list of tickers whose latest loaded OHLCV bar
    is dated BEFORE expected_trading_date (empty-dataframe tickers are
    skipped -- those are already counted as "not loaded" elsewhere, this
    function only flags "loaded but stale"). Pure/pulled out of run_scan()
    specifically so it's unit-testable without mocking the whole fetch
    pipeline.
    """
    stale = []
    for t, df in stock_data.items():
        if df.empty:
            continue
        try:
            last_bar_date = df.index[-1].date()
        except Exception:
            continue
        if last_bar_date < expected_trading_date:
            stale.append(t)
    return stale


def run_scan(dry_run: bool = False, max_tickers: int = None):
    log.info("=" * 60)
    log.info("  NSE MOMENTUM SCANNER v5.3" + ("  [DRY RUN]" if dry_run else ""))
    log.info(f"  {datetime.today().strftime('%d %b %Y  %H:%M')}")
    log.info("=" * 60)

    init_tables()

    # [2026-08-18] Data-freshness gate, part 1: code provenance. A fresh
    # GitHub Actions checkout should NEVER be dirty -- if it is, something
    # is wrong with the runner itself (not a normal "someone's iterating
    # locally" case), so this is treated as fatal on the cloud path only.
    # This is the direct fix for the incident where uncommitted local
    # fixes silently never reached the scheduled cloud scan: the scan now
    # records exactly which commit produced its output, so that class of
    # divergence is at least detectable after the fact even where it
    # can't be prevented outright.
    code_prov = get_code_provenance()
    if code_prov["is_cloud_runner"] and code_prov["git_dirty"]:
        reason = ("Cloud runner (GitHub Actions) reports a DIRTY working tree "
                   "-- a fresh checkout should never be dirty. This indicates "
                   "the runner itself is compromised or misconfigured, not a "
                   "normal data issue.")
        log.error(f"  DATA FRESHNESS GATE: {reason}")
        if not dry_run:
            try:
                send_data_freshness_alert_email([reason])
            except Exception as e:
                log.error(f"      Alert email failed: {e}")
        return None
    if code_prov["git_dirty"]:
        log.warning("  Running from an UNCOMMITTED (dirty) working tree -- "
                     "this run's output will be flagged LOCAL/DIRTY in "
                     "picks_latest.json and must not be confused with a "
                     "verified production run.")

    #      Build de-duplicated universe
    seen, universe = set(), []
    for item in NSE_UNIVERSE:
        if item[0] not in seen:
            seen.add(item[0])
            universe.append(item)

    if max_tickers:
        universe = universe[:max_tickers]
        log.info(f"      [DRY RUN] Universe capped to first {max_tickers} tickers for a fast sanity check")

    log.info(f"\n[0/6] Universe: {len(universe)} stocks")
    for u in ["LARGE", "MID", "SMALL"]:
        count = sum(1 for s in universe if s[3] == u)
        cfg   = UNIVERSE_CONFIG[u]
        log.info(f"      {cfg['label']:<24} {count:>3} stocks | "
                 f"gate>={cfg['score_gate']} | rrr>={cfg['min_rrr']}x | "
                 f"adt>=Rs.{cfg['min_adt_cr']}Cr")

    #      Market context
    log.info("\n[1/6] Fetching market context (Nifty / BankNifty / VIX)...")
    ctx = get_market_context()
    if ctx.get("vix_is_fallback"):
        log.warning(f"      VIX: {ctx['vix']:.1f} (FALLBACK DEFAULT -- live "
                     f"^INDIAVIX fetch failed, this is NOT a real reading)")
    else:
        log.info(f"      VIX: {ctx['vix']:.1f} (live)")

    #      Delivery data + full Bhavcopy for NSE-wide breadth
    #      [2026-08-18] Data-freshness gate, part 2: Bhavcopy must match the
    #      exact expected trading date, not "whichever of the last 5 days
    #      we could find" -- see BhavcopyFetcher.get_delivery_pct()'s
    #      docstring. A failure here aborts the scan before any further
    #      work rather than running the whole pipeline on data that would
    #      fail validation later anyway.
    log.info("\n[2/6] Fetching NSE Bhavcopy (delivery %)...")
    expected_trading_date = get_last_trading_day()
    bhav                        = BhavcopyFetcher()
    delivery, bhav_provenance   = bhav.get_delivery_pct(expected_trading_date)
    bhavcopy_full_df  = getattr(bhav, 'full_df', None)
    bhavcopy_cmp_map  = getattr(bhav, 'bhavcopy_cmp_map', {})

    if not bhav_provenance.ok:
        reason = f"Bhavcopy verification failed: {bhav_provenance.reason}"
        log.error(f"  DATA FRESHNESS GATE: {reason}")
        if not dry_run:
            try:
                send_data_freshness_alert_email([reason])
            except Exception as e:
                log.error(f"      Alert email failed: {e}")
        else:
            log.info("      [DRY RUN] Would have held the scan and sent a "
                       "data-freshness alert -- continuing since this is a dry run.")
            # Dry runs are explicitly allowed to proceed past this gate (for
            # sanity-checking scoring changes against whatever data IS
            # available) but a real run must not.
            pass
        if not dry_run:
            return None

    log.info(f"      {len(delivery)} symbols loaded | {len(bhavcopy_cmp_map)} CMP prices from Bhavcopy | "
             f"date verified: {bhav_provenance.actual_date}")

    # Holdings from Portfolio Dashboard (P1-04, via Turso bridge) — "what do
    # I currently own". Non-fatal on failure per P1-06: a bridge read must
    # never be able to break the scan.
    log.info("\n      Fetching current holdings from Turso (Portfolio Dashboard bridge)...")
    try:
        from turso_sync import get_holdings
        holdings = get_holdings()
        log.info(f"      {len(holdings)} tickers currently held")
    except Exception as e:
        log.warning(f"      Holdings fetch failed (non-fatal): {e}")
        holdings = {}

    # P1-07: fold held tickers not already in the static universe into the
    # scan, so a small/new listing you actually hold isn't silently skipped
    # just because it missed the 500-stock build list. Resolved symbols only
    # (see backfill_holdings_symbols.py / Portfolio Dashboard's entry-time
    # resolution) — a held symbol is always TICKER.NS/.BO by that point, so
    # this is a direct set-membership check, not fuzzy matching. Defaults to
    # the SMALL tier's config (strictest score_gate + min_rrr of the three
    # tiers) since we don't know its real market-cap bucket — safer to be
    # conservative on an unclassified stock than to accidentally grant it
    # LARGE's looser gate.
    held_not_in_universe = [t for t in holdings if t not in seen]
    if held_not_in_universe:
        names = _lookup_symbol_master_names(held_not_in_universe)
        for ticker in held_not_in_universe:
            name = names.get(ticker, ticker)
            universe.append((ticker, name, "Held (unclassified)", "SMALL"))
            seen.add(ticker)
        log.info(f"      Folded {len(held_not_in_universe)} held ticker(s) not in the "
                 f"static universe into today's scan: {', '.join(held_not_in_universe)}")

    #      OHLCV batch fetch
    # [2026-08-20] Retry-with-backoff for Dhan publish lag. Evidence from the
    # 2026-08-19 evening run: the cloud runner saw 0/503 (0%) fresh Dhan
    # coverage at BOTH the 19:30 IST trigger AND a retimed 20:15 IST trigger
    # (51 min after close), while a local fetch in between the two, at 19:52
    # IST, got 502/503 (99.8%) from the same Dhan endpoint. Moving the
    # trigger later did not fix it -- the cloud-vs-local gap isn't explained
    # by anything in this repo's date math (fromDate/toDate compute
    # identically in both environments), so it's treated as an
    # unpredictable-length publish/propagation delay on Dhan's side rather
    # than a fixed offset to schedule around. Retrying the fetch itself a
    # few times, with a real gap, is robust to that regardless of mechanism.
    # Cache is naturally bypassed on retry: _is_cache_fresh() in
    # data_fetcher.py compares the cached bar's date against today's trading
    # day, so a stale bar stored by a failed attempt fails that check on the
    # next attempt and forces a real re-fetch.
    log.info("\n[3/6] Fetching / loading OHLCV (2yr history)...")
    tickers = [item[0] for item in universe]

    MAX_FETCH_ATTEMPTS = 3
    RETRY_WAIT_SECONDS = 600  # 10 min

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        stock_data = fetch_batch_ohlcv(tickers, period="2y")
        loaded     = sum(1 for df in stock_data.values() if not df.empty)
        log.info(f"      {loaded}/{len(tickers)} tickers loaded")

        # [2026-08-18] Data-freshness gate, part 3: PER-TICKER date check, not
        # just aggregate presence/absence. "loaded" above only means "we got
        # SOMETHING for this ticker" -- a ticker whose latest bar is 2 trading
        # days old still counts as "loaded" there. This check asks the real
        # question per ticker: is the latest bar actually dated the expected
        # trading day? Named stale tickers are reported individually (not
        # folded into one aggregate %) so "1 of your actual picks was built on
        # stale data" stays visible even when overall coverage looks healthy.
        # Pulled out as a pure function (find_stale_tickers) so it's directly
        # unit-testable without mocking the whole fetch pipeline -- see
        # tests/test_data_freshness.py.
        stale_tickers   = find_stale_tickers(stock_data, expected_trading_date)
        fresh_loaded    = loaded - len(stale_tickers)
        coverage_pct    = round(100.0 * fresh_loaded / len(tickers), 1) if tickers else 0.0
        if stale_tickers:
            log.warning(f"      {len(stale_tickers)} ticker(s) loaded but NOT dated "
                         f"{expected_trading_date} (stale OHLCV): "
                         f"{', '.join(stale_tickers[:20])}"
                         f"{' ...' if len(stale_tickers) > 20 else ''}")
        log.info(f"      Per-ticker-verified fresh coverage: {fresh_loaded}/{len(tickers)} "
                 f"({coverage_pct}%) [attempt {attempt}/{MAX_FETCH_ATTEMPTS}]")

        if coverage_pct >= MIN_OHLCV_COVERAGE_PCT:
            break
        if attempt < MAX_FETCH_ATTEMPTS and not dry_run:
            log.warning(f"      Coverage below the {MIN_OHLCV_COVERAGE_PCT}% floor -- "
                        f"Dhan may not have finished publishing {expected_trading_date}'s "
                        f"EOD data yet. Retrying in {RETRY_WAIT_SECONDS // 60} min "
                        f"(attempt {attempt + 1}/{MAX_FETCH_ATTEMPTS})...")
            time.sleep(RETRY_WAIT_SECONDS)
        elif attempt < MAX_FETCH_ATTEMPTS and dry_run:
            log.info("      [DRY RUN] Would retry here -- skipping the wait, continuing "
                     "with whatever coverage this attempt got.")
            break

    if coverage_pct < MIN_OHLCV_COVERAGE_PCT:
        # [2026-08-18] Distinguish the two very different root causes that
        # both collapse to "0% fresh coverage" otherwise: a TOTAL fetch
        # failure (loaded=0 -- every data source down/unreachable, e.g. an
        # expired Dhan token + tvDatafeed CI install failure + Yahoo rate-
        # limiting all at once) vs tickers that DID load but are dated
        # before the expected trading day. The fix and the urgency are
        # different (network/auth outage vs a genuine publish delay), so
        # the alert email should say which one happened, not leave it to
        # be re-diagnosed by hand every time.
        if loaded == 0:
            cause = ("ALL data sources returned nothing for the entire universe "
                      "(Dhan/tvDatafeed/Yahoo all failed or are unreachable) -- "
                      "this looks like a total data-source outage, not a "
                      "publish-delay issue. Check Dhan token validity and "
                      "network/API reachability from the runner first.")
        else:
            cause = (f"{loaded} ticker(s) loaded but only {fresh_loaded} were dated "
                      f"{expected_trading_date} — the rest are dated earlier, which "
                      f"looks like a genuine data-publish delay rather than an outage.")
        reason = (f"OHLCV coverage verified fresh for only {coverage_pct}% of the "
                   f"universe ({fresh_loaded}/{len(tickers)}), below the "
                   f"{MIN_OHLCV_COVERAGE_PCT}% floor after {MAX_FETCH_ATTEMPTS} attempts "
                   f"{RETRY_WAIT_SECONDS // 60} min apart -- too incomplete a picture "
                   f"of the market to trust tonight's picks. {cause}")
        log.error(f"  DATA FRESHNESS GATE: {reason}")
        if not dry_run:
            try:
                send_data_freshness_alert_email([reason])
            except Exception as e:
                log.error(f"      Alert email failed: {e}")
            return None
        else:
            log.info("      [DRY RUN] Would have held the scan for low coverage -- continuing.")

    # Prepare data_dict for orchestrator
    data_dict = {
        "stock_data":       stock_data,
        "nifty50_data":     ctx["nifty50"],
        "banknifty_data":   ctx["banknifty"],
        "nifty500_data":    ctx["nifty500"],
        "vix":              ctx["vix"],
        "vix_is_fallback":  ctx.get("vix_is_fallback", True),
        "delivery_data":    delivery,
        "universe_meta":    {item[0]: item[2] for item in universe},
        "bhavcopy_full_df": bhavcopy_full_df,
        "bhavcopy_cmp_map": bhavcopy_cmp_map,
        "holdings":         holdings,
        # [2026-08-18] Provenance passthrough for orchestrator.py's
        # picks_latest.json meta block (section 2 of the data-freshness plan).
        "data_provenance": {
            "generated_at":              datetime.now().astimezone().isoformat(),
            "bhavcopy_trading_date":     bhav_provenance.actual_date.isoformat() if bhav_provenance.actual_date else None,
            "bhavcopy_requested_date":   expected_trading_date.isoformat(),
            "bhavcopy_date_verified":    bhav_provenance.ok,
            "ohlcv_universe_coverage":   {"requested": len(tickers), "fetched_fresh": fresh_loaded,
                                           "fetched_any": loaded, "pct": coverage_pct,
                                           "stale_tickers": stale_tickers},
            "vix_is_fallback":           ctx.get("vix_is_fallback", True),
            "code_provenance":           code_prov,
        },
    }

    #      CMP cross-validation (BUG-5 FIX)
    if bhavcopy_cmp_map:
        cmp_failures = validate_cmp_vs_bhavcopy(stock_data, bhavcopy_cmp_map)
        if cmp_failures:
            log.warning(f"      {len(cmp_failures)} stocks had stale CMP — corrected by Bhavcopy injection")

    #      Single-pass scoring
    log.info("\n[4/6] Running agent pipeline (single pass)...")
    orc = AgentOrchestrator(data_dict)
    log.info(f"      Regime: {orc.regime} ({orc.regime_name}) | "
             f"Breadth: {orc.breadth_score}/10")

    tiers = orc.run_universe(universe, stock_data, delivery, dry_run=dry_run)

    #      Pre-send sanity gate (2026-08-17) — drops any pick whose entry
    #      price has drifted too far from CMP or whose SL/target ordering
    #      is broken, BEFORE it can reach an email. See sanity_gate.py
    #      module docstring for why this exists.
    tiers, sanity_report = run_sanity_gate(tiers)
    t1, t2, t3, all_r = tiers["tier1"], tiers["tier2"], tiers["tier3"], tiers["all_results"]

    log.info(f"\n[5/6] Scoring complete:")
    log.info(f"      Tier 1 (Top Picks):   {len(t1)}")
    log.info(f"      Tier 2 (Aggressive):  {len(t2)}")
    log.info(f"      Tier 3 (Watchlist):   {len(t3)}")
    log.info(f"      Rejected/below gate:  {len(universe) - len(t1) - len(t2) - len(t3)}")

    #      Console print
    print_results(tiers)

    if dry_run:
        #      DRY RUN: skip email entirely, dump results + pattern distribution
        #      to disk instead so the new weights can be sanity-checked without
        #      sending a real email or needing repeated manual review.
        log.info("\n[6/6] DRY RUN — skipping email send.")
        print_pattern_distribution(all_r, log)

        dump_path = LOG_DIR / f"dryrun_results_{TODAY}_{datetime.now().strftime('%H%M%S')}.json"
        serializable = {
            "regime": tiers["regime"],
            "regime_name": tiers["regime_name"],
            "breadth": tiers["breadth"],
            "tier_counts": {"tier1": len(t1), "tier2": len(t2), "tier3": len(t3)},
            "sanity_gate": {
                "checked": sanity_report.checked,
                "excluded_count": sanity_report.excluded_count,
                "failure_rate": round(sanity_report.failure_rate, 3),
                "systemic_alert": sanity_report.systemic_alert,
                "excluded": sanity_report.excluded,
            },
            "all_results": [
                {
                    "ticker": r.ticker, "pattern": r.pattern, "total_score": r.total_score,
                    "rs_percentile": r.rs_percentile, "rvol": r.rvol, "entry": r.entry,
                    "stop_loss": r.stop_loss, "target1": r.target1, "target2": r.target2,
                    "rrr": r.rrr, "universe": r.universe,
                }
                for r in all_r
            ],
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        log.info(f"      Results written to {dump_path}")
    else:
        if sanity_report.systemic_alert:
            #      Sanity gate found too high a failure rate to trust the
            #      rest of the report — hold the normal send and alert
            #      instead of shipping a degraded-but-plausible-looking
            #      email. See sanity_gate.py module docstring.
            log.error("\n[6/6] SANITY GATE SYSTEMIC ALERT — holding normal report, "
                      "sending alert email instead.")
            try:
                send_sanity_alert_email(sanity_report, tiers["regime"], tiers["regime_name"])
                log.info("      Alert email sent.")
            except Exception as e:
                log.error(f"      Alert email failed: {e}")
        else:
            #      Email
            log.info("\n[6/6] Sending email report...")
            try:
                send_email_report(tiers)
                log.info("      Email sent successfully.")
            except Exception as e:
                log.error(f"      Email failed: {e}")

    log.info("\n" + "=" * 60)
    log.info("  Scan complete.")
    log.info("=" * 60 + "\n")
    return tiers


def print_pattern_distribution(all_r, log):
    """
    v6.2 weight-rebuild sanity check: how many picks came from each
    pattern, across ALL results (not just Tier 1). Lets you eyeball
    whether the new weights (Cup & Handle / Swing High Breakout / VCP)
    are actually winning pattern selection now, versus the old 5
    (High Tight Flag / Flat Base / Rounded Base / High Base / Volume
    Expansion) which should no longer appear at all since they're now
    in PRUNED_PATTERNS with weight 0.
    """
    from collections import Counter
    counts = Counter(r.pattern for r in all_r if r.pattern)
    log.info("\n      Pattern distribution across all scored results:")
    if not counts:
        log.info("        (no patterns assigned — check gate thresholds / data)")
        return
    old_five = {"High Tight Flag", "Flat Base", "Rounded Base", "High Base", "Volume Expansion"}
    for pattern, count in counts.most_common():
        flag = "  <-- SHOULD NOT APPEAR (old pruned pattern)" if pattern in old_five else ""
        log.info(f"        {pattern:<24} {count:>4}{flag}")


def print_results(tiers: dict):
    t1    = tiers["tier1"]
    t2    = tiers["tier2"]
    t3    = tiers["tier3"]
    all_r = tiers["all_results"]
    reg   = tiers["regime"]
    br    = tiers["breadth"]

    print(f"\n{'='*68}")
    print(f"  NSE MOMENTUM v5.3    Daily Scan Results")
    print(f"  Regime {reg} ({tiers['regime_name']}) | Breadth {br}/10")
    print(f"{'='*68}")

    if t1:
        print(f"\n  TIER 1    TOP PICKS ({len(t1)} stocks, gate cleared)")
        print(f"  {'  '*62}")
        for r in t1:
            print(f"  {r.ticker:<16} {r.pattern:<22} Score: {r.total_score:>3}  "
                  f"RS: {r.rs_percentile:.0f}th  RVOL: {r.rvol:.1f}x")
            print(f"  {'':16} Entry: {r.entry:.1f}  SL: {r.stop_loss:.1f}  "
                  f"T1: {r.target1:.1f}  T2: {r.target2:.1f}  R:R: {r.rrr:.1f}x")
            for why in r.what_is_working[:2]:
                print(f"  {'':16} OK {why}")
            print()
    else:
        print(f"\n  TIER 1: No stocks cleared the conviction gate.")
        if reg in ["D", "E"]:
            print(f"  Regime {reg} penalty is suppressing scores. Stay in cash.")

    if t2:
        print(f"\n  TIER 2    AGGRESSIVE ({len(t2)} stocks, one condition missing)")
        print(f"  {'  '*62}")
        for r in t2[:5]:
            print(f"  {r.ticker:<16} {r.pattern:<22} Score: {r.total_score:>3}  "
                  f"RS: {r.rs_percentile:.0f}th")
            for miss in r.what_is_missing[:2]:
                print(f"  {'':16}    {miss}")
            if r.trigger_conditions:
                print(f"  {'':16}    {r.trigger_conditions[0]}")
            print()

    if t3:
        print(f"\n  TIER 3    WATCHLIST ({len(t3)} stocks, setup forming)")
        print(f"  {'  '*62}")
        for r in t3[:5]:
            print(f"  {r.ticker:<16} {r.pattern:<22} Score: {r.total_score:>3}")

    if all_r:
        print(f"\n  TOP 20 ACROSS ALL TIERS:")
        print(f"  {'TICKER':<14} {'PATTERN':<22} {'SCORE':>5} {'RS%':>6} {'RVOL':>6} "
              f"{'ENTRY':>8} {'SL':>8} {'T1':>8} {'RR':>5} {'UNI':>6}")
        print(f"  {'  '*100}")
        for r in all_r[:20]:
            print(f"  {r.ticker:<14} {r.pattern:<22} {r.total_score:>5} "
                  f"{r.rs_percentile:>5.0f}% {r.rvol:>6.1f}x "
                  f"{r.entry:>8.1f} {r.stop_loss:>8.1f} {r.target1:>8.1f} "
                  f"{r.rrr:>4.1f}x {r.universe:>6}")

    print(f"\n{'='*68}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Momentum Scanner")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run the full pipeline but skip the email send; "
                              "writes results + pattern distribution to logs/ instead")
    parser.add_argument("--max-tickers", type=int, default=None,
                         help="Cap the universe to the first N tickers, for a fast "
                              "smoke test (e.g. --max-tickers 30) instead of the full 504")
    args = parser.parse_args()
    run_scan(dry_run=args.dry_run, max_tickers=args.max_tickers)
