# -*- coding: utf-8 -*-
"""
NSE Momentum v5.3    Main Scanner
Single pass through 504 stocks. No double scoring.
Outputs 3-tier report + sends combined HTML email.
Run: python scanner.py (or double-click run_scanner.bat)
"""

import sys, logging, json
from datetime import datetime
from pathlib import Path

from data_fetcher import fetch_batch_ohlcv, get_market_context, BhavcopyFetcher, validate_cmp_vs_bhavcopy
from orchestrator import AgentOrchestrator
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from emailer      import send_email_report
from trade_logger import init_tables

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


def run_scan():
    log.info("=" * 60)
    log.info("  NSE MOMENTUM SCANNER v5.3")
    log.info(f"  {datetime.today().strftime('%d %b %Y  %H:%M')}")
    log.info("=" * 60)

    init_tables()

    #      Build de-duplicated universe
    seen, universe = set(), []
    for item in NSE_UNIVERSE:
        if item[0] not in seen:
            seen.add(item[0])
            universe.append(item)

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
    log.info(f"      VIX: {ctx['vix']:.1f}")

    #      Delivery data + full Bhavcopy for NSE-wide breadth
    log.info("\n[2/6] Fetching NSE Bhavcopy (delivery %)...")
    bhav             = BhavcopyFetcher()
    delivery         = bhav.get_delivery_pct()
    bhavcopy_full_df  = getattr(bhav, 'full_df', None)
    bhavcopy_cmp_map  = getattr(bhav, 'bhavcopy_cmp_map', {})
    log.info(f"      {len(delivery)} symbols loaded | {len(bhavcopy_cmp_map)} CMP prices from Bhavcopy")

    #      OHLCV batch fetch
    log.info("\n[3/6] Fetching / loading OHLCV (2yr history)...")
    tickers    = [item[0] for item in universe]
    stock_data = fetch_batch_ohlcv(tickers, period="2y")
    loaded     = sum(1 for df in stock_data.values() if not df.empty)
    log.info(f"      {loaded}/{len(tickers)} tickers loaded")

    # Prepare data_dict for orchestrator
    data_dict = {
        "stock_data":       stock_data,
        "nifty50_data":     ctx["nifty50"],
        "banknifty_data":   ctx["banknifty"],
        "nifty500_data":    ctx["nifty500"],
        "vix":              ctx["vix"],
        "delivery_data":    delivery,
        "universe_meta":    {item[0]: item[2] for item in universe},
        "bhavcopy_full_df": bhavcopy_full_df,
        "bhavcopy_cmp_map": bhavcopy_cmp_map,
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

    tiers = orc.run_universe(universe, stock_data, delivery)
    t1, t2, t3, all_r = tiers["tier1"], tiers["tier2"], tiers["tier3"], tiers["all_results"]

    log.info(f"\n[5/6] Scoring complete:")
    log.info(f"      Tier 1 (Top Picks):   {len(t1)}")
    log.info(f"      Tier 2 (Aggressive):  {len(t2)}")
    log.info(f"      Tier 3 (Watchlist):   {len(t3)}")
    log.info(f"      Rejected/below gate:  {len(universe) - len(t1) - len(t2) - len(t3)}")

    #      Console print
    print_results(tiers)

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
    run_scan()
