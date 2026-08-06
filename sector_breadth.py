"""
sector_breadth.py — P3-01 / P3-02 / P3-06

Computes count-based sector breadth (% of stocks above SMA20/50/100, %
above RSI(14)=50, % with positive 55-day return) from price_history_deep,
and publishes the result to Turso's sector_breadth table (schema from
P1-01) for Portfolio Dashboard to render as a rotation grid.

Deliberately count-based, not market-cap-weighted (P3-05, separate item):
per the backlog's own reasoning, count-based is "arguably more robust,
since MCap weighting lets 2-3 megacaps dominate a sector reading" — and
it's fully buildable today with data already on hand, unlike MCap
weighting which needs a market-cap source not yet wired in.

P3-06 (added 05 Aug 2026): adds 3 volume-quality metrics per sector,
sourced from the SAME BhavcopyFetcher instance scanner.py already runs
once per day (data_fetcher.py) — this module re-instantiates
BhavcopyFetcher and calls get_delivery_pct(), which reads from the local
CACHE_DIR/bhav_DDMMYYYY.csv file written by that earlier run rather than
hitting NSE's archive a second time, PROVIDED it's run after scanner.py
the same day (falls back to the network, same 5-day lookback the class
already does, if run standalone or the cache is missing):
  - pct_above_vwap: % of sector's stocks with CLOSE_PRICE > AVG_PRICE
    (NSE's own official VWAP for the day).
  - pct_high_delivery: % of sector's stocks with DELIV_PER >= 50 — reuses
    the same 50% threshold institutional_proxy_agent.py already uses for
    its own delivery-based accumulation scoring, not a new number invented
    here.
  - avg_turnover_cr: mean daily traded value per stock in the sector,
    converted from TURNOVER_LACS to Rs Crore (÷100), the standard Indian
    market convention.
  Percentages are computed over the subset of each sector's tickers that
  actually matched a Bhavcopy row that day (bhav_coverage), not the
  sector's total stock_count — a ticker with no Bhavcopy match that day
  (rare; NNSE Bhavcopy covers ~3200 EQ symbols) is excluded from the
  Bhavcopy-derived percentages rather than counted as a false negative.
  If Bhavcopy is unavailable entirely (weekend, NSE archive down, no
  cached file), the 3 new fields come back as None for every sector and
  bhav_coverage=0 — the existing price-only metrics are computed and
  published exactly as before. Matches the P1-06 convention used
  throughout this codebase: a Bhavcopy read failure must never break the
  rest of the pipeline.
  Bhavcopy rows are filtered to SERIES=='EQ' before use, so government
  securities / bonds / ETFs in the same CSV (e.g. "1018GS2026") don't
  skew turnover or delivery aggregates. Both SYMBOL and SERIES are
  .str.strip()'d defensively — the raw NSE CSV uses ", " (comma-space)
  as its delimiter, so pandas reads string values with a leading space
  (e.g. " EQ", not "EQ") unless stripped.
  New columns are added to the existing sector_breadth Turso table via
  an idempotent ALTER TABLE (checked against PRAGMA table_info first, so
  safe to re-run) rather than a new table — one source of truth for
  Portfolio Dashboard, which already reads this table.

Definitions (stated explicitly since the backlog doesn't fully spell
these out numerically):
  - Source: price_history (the daily-refreshed 2yr table maintained by
    price_collector.py), NOT price_history_deep. The deep table is capped
    at the Kaggle source file's own ceiling (23 Feb 2026 as of this
    writing) -- correct for backtesting, wrong for "what's the market
    doing right now." price_history is refreshed every trading day and
    comfortably covers the ~150-day lookback SMA100/RSI/RS55 need.
  - RSI: 14-day, matching the RSI convention already used elsewhere in
    this codebase (orchestrator.py's own RSI(14) calc).
  - RS55: 55-trading-day price return (close_today / close_55d_ago - 1).
    This is a per-stock ABSOLUTE momentum measure, distinct from the RS
    PERCENTILE used elsewhere in the scanner (that's a cross-sectional
    rank against the universe; this is StockEdge's own "RS55" window-
    return convention).
  - Sector taxonomy: NSE's own official Nifty 500 constituent list
    (data/ind_nifty500list.csv, https://nsearchives.nseindia.com/content/
    indices/ind_nifty500list.csv), NOT nse_universe.py's own sector field.
    Confirmed 500/500 ticker match, 20 clean non-duplicated sectors --
    nse_universe.py's sector field has real duplication (e.g. "Financials"
    vs "Financial Services", "Auto" vs "Automobile and Auto Components",
    "Metals" vs "Metals & Mining") from the 401->500 universe expansion
    merging two different naming conventions. Falls back to
    nse_universe.py's own sector field only if a ticker isn't found in the
    official list (logged clearly, not silent) -- should be rare given
    today's confirmed 100% match rate, but a ticker added to the universe
    after this CSV was downloaded wouldn't yet be in it.
  - Snapshot as of the latest available date per ticker, not a time
    series -- re-runnable daily, same as everything else in this pipeline.

Usage:
    python sector_breadth.py            # compute + publish to Turso
    python sector_breadth.py --dry-run   # compute + print, skip Turso publish
"""

import argparse
import sys
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from database.schema import get_connection
from nse_universe import NSE_UNIVERSE
from data_fetcher import BhavcopyFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 100   # need at least 100 daily closes for SMA100
HIGH_DELIVERY_THRESHOLD = 50.0  # matches institutional_proxy_agent.py's own threshold

OFFICIAL_SECTOR_CSV = Path(__file__).parent / "data" / "ind_nifty500list.csv"


def _load_official_sectors() -> dict:
    """
    ticker (with .NS) -> official NSE sector, from data/ind_nifty500list.csv
    (NSE's own live-maintained Nifty 500 constituent list -- confirmed
    500/500 ticker match, 20 clean non-duplicated sectors, vs.
    nse_universe.py's own sector field which has real duplication from the
    401->500 universe expansion merging two naming conventions).

    Returns {} if the file is missing -- callers fall back to
    nse_universe.py's own sector field entirely in that case, logged
    clearly, never silently.
    """
    if not OFFICIAL_SECTOR_CSV.exists():
        return {}
    import csv as _csv
    mapping = {}
    with open(OFFICIAL_SECTOR_CSV, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            symbol = row.get("Symbol", "").strip()
            sector = row.get("Industry", "").strip()
            if symbol and sector:
                mapping[f"{symbol}.NS"] = sector
    return mapping


def _load_bhavcopy_metrics() -> dict:
    """
    P3-06: {SYMBOL (no .NS suffix): {"above_vwap": bool, "high_delivery":
    bool, "turnover_lacs": float}} for every EQ-series symbol in today's
    (or the most recent available) Bhavcopy.

    Re-instantiates BhavcopyFetcher and calls get_delivery_pct() -- this
    reads from data/bhavcopy_cache/bhav_DDMMYYYY.csv if scanner.py already
    ran today (the common case), or fetches fresh (same 5-trading-day
    lookback the class already implements) if run standalone. Returns {}
    on any failure -- Bhavcopy unavailability must never break the rest of
    sector breadth (P1-06 pattern, same as everywhere else in this
    codebase).
    """
    try:
        bhav = BhavcopyFetcher()
        bhav.get_delivery_pct()  # populates bhav.full_df as a side effect
        df = bhav.full_df
    except Exception as e:
        log.warning(f"  Bhavcopy fetch failed ({e}) — P3-06 metrics will be unavailable, "
                    f"price-based breadth continues normally")
        return {}

    if df is None or df.empty:
        log.warning("  Bhavcopy full_df unavailable — P3-06 metrics will be unavailable, "
                    "price-based breadth continues normally")
        return {}

    cols = list(df.columns)  # already stripped+uppercased by BhavcopyFetcher._parse()
    close_col = next((c for c in cols if c in ("CLOSE_PRICE", "CLOSE", "CLOSING_PRICE")), None)
    avg_col   = next((c for c in cols if c == "AVG_PRICE"), None)
    deliv_col = next((c for c in cols if "DELIV" in c and "PER" in c), None)
    turn_col  = next((c for c in cols if "TURNOVER" in c), None)
    sym_col   = next((c for c in cols if "SYMBOL" in c), None)
    series_col = next((c for c in cols if c == "SERIES"), None)

    if not all([close_col, avg_col, deliv_col, turn_col, sym_col]):
        log.warning(f"  Bhavcopy missing expected columns for P3-06 (have: {cols[:15]}) — "
                    f"P3-06 metrics will be unavailable, price-based breadth continues normally")
        return {}

    sub = df.copy()
    # Defensive strip: the raw NSE CSV uses ", " as its delimiter, so
    # string values carry a leading space (e.g. " EQ") unless stripped --
    # column NAMES are already stripped by _parse(), values are not.
    sub[sym_col] = sub[sym_col].astype(str).str.strip()
    if series_col:
        sub[series_col] = sub[series_col].astype(str).str.strip()
        sub = sub[sub[series_col] == "EQ"]

    for c in (close_col, avg_col, deliv_col, turn_col):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=[close_col, avg_col, deliv_col, turn_col])

    metrics = {}
    for _, row in sub.iterrows():
        metrics[row[sym_col]] = {
            "above_vwap":     bool(row[close_col] > row[avg_col]),
            "high_delivery":  bool(row[deliv_col] >= HIGH_DELIVERY_THRESHOLD),
            "turnover_lacs":  float(row[turn_col]),
        }
    log.info(f"  Bhavcopy P3-06 metrics loaded for {len(metrics)} EQ symbols "
             f"(date={getattr(bhav, 'trading_date', 'unknown')})")
    return metrics



def _load_mcap_weights() -> dict:
    """
    P3-05: {ticker (with .NS): mcap_cr (float)} from Turso ticker_mcap table.
    Built by fetch_mcap.py from NSE's official daily MCap report.
    Returns {} on any failure — MCap weighting gracefully degrades to None
    for affected sectors, consistent with the P1-06 fallback pattern used
    throughout this codebase (Bhavcopy unavailability never breaks price breadth).
    """
    try:
        from turso_sync import get_client
        client = get_client()
    except (SystemExit, Exception) as e:
        log.warning(f"  MCap weights unavailable ({e}) — pct_above_sma50_mcap will be None")
        return {}
    try:
        rs = client.execute("SELECT ticker, mcap_cr FROM ticker_mcap")
        result = {row[0]: float(row[1]) for row in rs.rows if row[1]}
        log.info(f"  MCap weights loaded for {len(result)} tickers (P3-05)")
        return result
    except Exception as e:
        log.warning(f"  ticker_mcap read failed ({e}) — pct_above_sma50_mcap will be None")
        return {}
    finally:
        client.close()


def _compute_stock_flags(close: np.ndarray) -> dict:
    """
    Per-stock breadth flags as of the LAST value in `close`. Returns None
    if there isn't enough history to compute SMA100 reliably -- mirrors
    the "not enough data" guards used elsewhere in this pipeline
    (compute_holding_stop, RiskAgent) rather than silently zero-filling.
    """
    if len(close) < MIN_HISTORY_DAYS:
        return None

    price  = close[-1]
    sma20  = float(np.mean(close[-20:]))
    sma50  = float(np.mean(close[-50:]))
    sma100 = float(np.mean(close[-100:]))

    delta = np.diff(close[-15:])
    gain  = float(np.mean(np.clip(delta, 0, None)))
    loss  = float(np.mean(np.clip(-delta, 0, None)))
    rsi   = 100.0 - 100.0 / (1 + gain / loss) if loss > 0 else 100.0

    rs55 = (price / close[-56] - 1) if len(close) >= 56 else None

    return {
        "above_sma20":  price > sma20,
        "above_sma50":  price > sma50,
        "above_sma100": price > sma100,
        "above_rsi50":  rsi > 50,
        "above_rs55":   rs55 is not None and rs55 > 0,
    }


def build_ticker_sector_mapping() -> tuple:
    """
    Shared by compute_sector_breadth() and publish_ticker_sector_map()
    (P3-08) -- one mapping, built once, used both for aggregation and for
    publishing the per-ticker lookup Portfolio Dashboard needs to cross-
    reference holdings against sector breadth (it has no access to
    nse_universe.py or the NSE CSV directly -- same "publish once, read via
    Turso" pattern as everything else in this bridge).
    Returns (ticker_to_sector: dict, n_official: int, n_fallback: int).
    """
    fallback_sector = {item[0]: item[2] for item in NSE_UNIVERSE}
    official_sector = _load_official_sectors()
    ticker_to_sector = {}
    n_official, n_fallback = 0, 0
    for ticker in fallback_sector:
        if ticker in official_sector:
            ticker_to_sector[ticker] = official_sector[ticker]
            n_official += 1
        else:
            ticker_to_sector[ticker] = fallback_sector[ticker]
            n_fallback += 1
    return ticker_to_sector, n_official, n_fallback


def compute_sector_breadth(as_of_date: str = None) -> list:
    """
    Returns a list of dicts, one per sector:
    {sector, breadth_date, pct_above_sma20, pct_above_sma50, pct_above_sma100,
     pct_above_rsi50, pct_above_rs55, stock_count, pct_above_vwap,
     pct_high_delivery, avg_turnover_cr, bhav_coverage}

    Reads price_history_deep for every ticker in nse_universe.py, computes
    per-stock flags as of the latest close on or before as_of_date (defaults
    to the latest date actually in the table), and aggregates to %-of-stocks-
    meeting-each-condition per sector. Also folds in P3-06 Bhavcopy-derived
    volume-quality metrics (see module docstring) -- these come back as
    None/0 for every sector if Bhavcopy is unavailable, without affecting
    the price-based metrics.
    """
    ticker_to_sector, n_official, n_fallback = build_ticker_sector_mapping()
    tickers = list(ticker_to_sector.keys())

    if n_fallback:
        log.warning(f"  {n_fallback} ticker(s) not found in official NSE sector list — "
                    f"using nse_universe.py's own (possibly inconsistent) sector field for these")
    log.info(f"  Sector source: {n_official} from official NSE list, {n_fallback} fallback")

    log.info("  Loading Bhavcopy volume-quality metrics (P3-06)...")
    log.info("  Loading MCap weights (P3-05)...")
    mcap_weights = _load_mcap_weights()
    bhav_metrics = _load_bhavcopy_metrics()

    conn = get_connection()
    if as_of_date is None:
        row = conn.execute("SELECT MAX(date) FROM price_history").fetchone()
        as_of_date = row[0] if row and row[0] else date.today().isoformat()

    log.info(f"Computing sector breadth as of {as_of_date} for {len(tickers)} tickers...")

    sector_flags = defaultdict(lambda: {
        "above_sma20": 0, "above_sma50": 0, "above_sma100": 0,
        "above_rsi50": 0, "above_rs55": 0, "count": 0,
        "above_vwap": 0, "high_delivery": 0, "turnover_sum": 0.0, "bhav_coverage": 0,
        "mcap_total": 0.0, "mcap_above_sma50": 0.0,  # P3-05
    })

    n_skipped_no_data = 0
    n_skipped_short_history = 0

    for i, ticker in enumerate(tickers, 1):
        sector = ticker_to_sector[ticker]
        rows = conn.execute(
            "SELECT close FROM price_history WHERE ticker = ? AND date <= ? ORDER BY date",
            (ticker, as_of_date),
        ).fetchall()
        if not rows:
            n_skipped_no_data += 1
            continue

        close = np.array([r[0] for r in rows], dtype=float)
        flags = _compute_stock_flags(close)
        if flags is None:
            n_skipped_short_history += 1
            continue

        s = sector_flags[sector]
        s["count"] += 1
        for key, val in flags.items():
            if val:
                s[key] += 1

        # P3-05: accumulate MCap weighting for SMA50 breadth.
        ticker_mcap = mcap_weights.get(ticker)
        if ticker_mcap and ticker_mcap > 0 and flags is not None:
            s["mcap_total"] += ticker_mcap
            if flags["above_sma50"]:
                s["mcap_above_sma50"] += ticker_mcap

        # P3-06: fold in Bhavcopy metrics for this ticker, if it matched.
        symbol_no_suffix = ticker[:-3] if ticker.endswith(".NS") else ticker
        bm = bhav_metrics.get(symbol_no_suffix)
        if bm is not None:
            s["bhav_coverage"] += 1
            if bm["above_vwap"]:
                s["above_vwap"] += 1
            if bm["high_delivery"]:
                s["high_delivery"] += 1
            s["turnover_sum"] += bm["turnover_lacs"]

        if i % 100 == 0:
            log.info(f"  {i}/{len(tickers)} tickers processed...")

    conn.close()

    if n_skipped_no_data or n_skipped_short_history:
        log.warning(f"  Skipped {n_skipped_no_data} tickers with no price_history rows, "
                    f"{n_skipped_short_history} with < {MIN_HISTORY_DAYS} days of history")

    results = []
    for sector, s in sorted(sector_flags.items()):
        count = s["count"]
        if count == 0:
            continue
        bhav_n = s["bhav_coverage"]
        results.append({
            "sector":            sector,
            "breadth_date":      as_of_date,
            "pct_above_sma20":   round(s["above_sma20"]  / count * 100, 1),
            "pct_above_sma50":   round(s["above_sma50"]  / count * 100, 1),
            "pct_above_sma100":  round(s["above_sma100"] / count * 100, 1),
            "pct_above_rsi50":   round(s["above_rsi50"]  / count * 100, 1),
            "pct_above_rs55":    round(s["above_rs55"]   / count * 100, 1),
            "stock_count":       count,
            # P3-06 — None if Bhavcopy had zero matches for this sector today.
            "pct_above_vwap":    round(s["above_vwap"] / bhav_n * 100, 1) if bhav_n else None,
            "pct_high_delivery": round(s["high_delivery"] / bhav_n * 100, 1) if bhav_n else None,
            "avg_turnover_cr":   round(s["turnover_sum"] / bhav_n / 100.0, 2) if bhav_n else None,
            "bhav_coverage":     bhav_n,
            # P3-05: MCap-weighted SMA50 breadth — None if no MCap data for sector.
            "pct_above_sma50_mcap": (
                round(s["mcap_above_sma50"] / s["mcap_total"] * 100, 1)
                if s["mcap_total"] > 0 else None
            ),
        })
    return results


def print_breadth_table(results: list):
    print(f"\n{'SECTOR':<38}{'COUNT':>6}{'SMA20':>8}{'SMA50':>8}{'SMA100':>8}{'RSI50':>8}{'RS55':>8}"
          f"{'VWAP':>8}{'HIDELIV':>9}{'TURN(Cr)':>10}{'BHAVN':>7}")
    print("-" * 130)
    for r in sorted(results, key=lambda x: -x["pct_above_sma50"]):
        vwap  = f"{r['pct_above_vwap']:>7.1f}%" if r['pct_above_vwap'] is not None else "     n/a"
        hdel  = f"{r['pct_high_delivery']:>8.1f}%" if r['pct_high_delivery'] is not None else "      n/a"
        turn  = f"{r['avg_turnover_cr']:>9.1f}" if r['avg_turnover_cr'] is not None else "      n/a"
        print(f"{r['sector']:<38}{r['stock_count']:>6}"
              f"{r['pct_above_sma20']:>7.1f}%{r['pct_above_sma50']:>7.1f}%"
              f"{r['pct_above_sma100']:>7.1f}%{r['pct_above_rsi50']:>7.1f}%"
              f"{r['pct_above_rs55']:>7.1f}%{vwap}{hdel}{turn}{r['bhav_coverage']:>7}")


def _ensure_bhav_columns(client) -> None:
    """
    P3-06: idempotent ALTER TABLE — adds the 3 new columns to the existing
    sector_breadth table if they aren't already there. Checked against
    PRAGMA table_info first rather than try/except-on-duplicate, so a
    re-run never logs a spurious error.
    """
    rs = client.execute("PRAGMA table_info(sector_breadth)")
    existing = {row[1] for row in rs.rows}  # row[1] is the column name
    new_columns = {
        "pct_above_vwap":    "REAL",
        "pct_high_delivery": "REAL",
        "avg_turnover_cr":   "REAL",
        "bhav_coverage":     "INTEGER",
        "pct_above_sma50_mcap": "REAL",   # P3-05
    }
    for col, coltype in new_columns.items():
        if col not in existing:
            client.execute(f"ALTER TABLE sector_breadth ADD COLUMN {col} {coltype}")
            log.info(f"  Added column {col} ({coltype}) to sector_breadth table")


def publish_to_turso(results: list) -> int:
    from turso_sync import get_client
    now = datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Publish skipped — {e}")
        return 0
    published = 0
    try:
        _ensure_bhav_columns(client)
        for r in results:
            try:
                client.execute(
                    """
                    INSERT INTO sector_breadth (
                        sector, breadth_date, pct_above_sma20, pct_above_sma50,
                        pct_above_sma100, pct_above_rsi50, pct_above_rs55,
                        stock_count, pct_above_vwap, pct_high_delivery,
                        avg_turnover_cr, bhav_coverage, pct_above_sma50_mcap, published_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(sector, breadth_date) DO UPDATE SET
                        pct_above_sma20=excluded.pct_above_sma20,
                        pct_above_sma50=excluded.pct_above_sma50,
                        pct_above_sma100=excluded.pct_above_sma100,
                        pct_above_rsi50=excluded.pct_above_rsi50,
                        pct_above_rs55=excluded.pct_above_rs55,
                        stock_count=excluded.stock_count,
                        pct_above_vwap=excluded.pct_above_vwap,
                        pct_high_delivery=excluded.pct_high_delivery,
                        avg_turnover_cr=excluded.avg_turnover_cr,
                        bhav_coverage=excluded.bhav_coverage,
                        pct_above_sma50_mcap=excluded.pct_above_sma50_mcap,
                        published_at=excluded.published_at
                    """,
                    [r["sector"], r["breadth_date"], r["pct_above_sma20"], r["pct_above_sma50"],
                     r["pct_above_sma100"], r["pct_above_rsi50"], r["pct_above_rs55"],
                     r["stock_count"], r["pct_above_vwap"], r["pct_high_delivery"],
                     r["avg_turnover_cr"], r["bhav_coverage"],
                     r.get("pct_above_sma50_mcap"), now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish {r['sector']}: {e}")
    finally:
        client.close()
    return published


def publish_ticker_sector_map() -> int:
    """
    P3-08: publish the full ticker->sector mapping to Turso so Portfolio
    Dashboard can cross-reference holdings against sector breadth without
    needing its own copy of nse_universe.py or the NSE CSV. Own small table,
    separate from sector_breadth (that one's aggregated per-sector; this
    one's per-ticker) -- upserts on ticker, so safe to re-run.
    """
    from turso_sync import get_client
    official_sector = _load_official_sectors()
    ticker_to_sector, n_official, n_fallback = build_ticker_sector_mapping()
    now = datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Publish skipped — {e}")
        return 0
    published = 0
    try:
        client.execute("""
            CREATE TABLE IF NOT EXISTS ticker_sector_map (
                ticker        TEXT PRIMARY KEY,
                sector        TEXT NOT NULL,
                source        TEXT,
                published_at  TEXT
            )
        """)
        for ticker, sector in ticker_to_sector.items():
            source = "official" if ticker in official_sector else "fallback"
            try:
                client.execute(
                    """
                    INSERT INTO ticker_sector_map (ticker, sector, source, published_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        sector=excluded.sector, source=excluded.source,
                        published_at=excluded.published_at
                    """,
                    [ticker, sector, source, now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish sector map for {ticker}: {e}")
    finally:
        client.close()
    return published


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print only, skip Turso publish")
    ap.add_argument("--date", default=None, help="Compute as of this date (YYYY-MM-DD); defaults to latest available")
    args = ap.parse_args()

    results = compute_sector_breadth(as_of_date=args.date)
    print_breadth_table(results)

    if args.dry_run:
        print(f"\n[DRY RUN] Would publish {len(results)} sector rows to Turso — skipped.")
    else:
        n = publish_to_turso(results)
        print(f"\nPublished {n}/{len(results)} sector rows to Turso sector_breadth table.")
        n_map = publish_ticker_sector_map()
        print(f"Published {n_map} ticker->sector mappings to Turso ticker_sector_map table (P3-08).")
