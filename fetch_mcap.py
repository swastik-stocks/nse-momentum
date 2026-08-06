"""
fetch_mcap.py — P3-05

Fetches current market capitalisation (Rs Crore) for all NSE-listed stocks
from NSE's official daily MCap report (mcap{date}.csv inside the PR Bhavcopy
zip), and publishes to a Turso `ticker_mcap` table so sector_breadth.py can
compute MCap-weighted breadth alongside its existing count-based breadth.

SOURCE:
  https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{DDMMYY}.zip
  Contains: mcap{DDMMYYYY}.csv  — NSE's official MCap file, published daily.
  - No login required (static archive file, same host as Bhavcopy).
  - No cookie session needed (unlike nseindia.com live API endpoints).
  - Covers all ~2600 NSE-listed stocks, not just NIFTY 500.
  - Fields: Symbol, Series, MktCapFullFreeFloat (Cr), MktCapFull (Cr), etc.
    (exact column names vary by date — defensive search used below).

WHY THIS MATTERS (backlog P3-05):
  Count-based breadth treats a Rs 2,000cr and a Rs 2,00,000cr stock
  identically. MCap weighting catches cases where breadth looks healthy on
  count but is actually carried by 2-3 megacaps — a real regime-read risk.

DESIGN:
  - Tries the last 5 trading days (Mon–Fri) with the same 5-day lookback
    pattern already used by BhavcopyFetcher.get_delivery_pct().
  - Caches the zip to data/bhavcopy_cache/mcap_{DDMMYYYY}.csv — same
    cache dir as Bhavcopy, avoids re-downloading on same-day reruns.
  - Filters to EQ series only (same as Bhavcopy delivery% filtering).
  - Falls back gracefully if the archive is unavailable.
  - Idempotent Turso UPSERT — safe to run daily.

Usage:
    python fetch_mcap.py --dry-run    # fetch and print, skip Turso
    python fetch_mcap.py              # fetch + publish to Turso ticker_mcap
"""

import argparse
import io
import logging
import sys
import zipfile
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Optional

import requests
import pandas as pd

BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "data" / "bhavcopy_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_URL  = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get_last_trading_day(from_date: date = None) -> date:
    d = from_date or date.today()
    for i in range(7):
        candidate = d - timedelta(days=i)
        if candidate.weekday() < 5:
            return candidate
    return d


def _fetch_mcap_csv(trading_date: date) -> Optional[pd.DataFrame]:
    """
    Download PR{ddmmyy}.zip, extract mcap{DDMMYYYY}.csv, return as DataFrame.
    Uses cache — returns from disk if already downloaded today.
    Returns None on any failure.
    """
    ddmmyy   = trading_date.strftime("%d%m%y")
    ddmmyyyy = trading_date.strftime("%d%m%Y")
    cache_path = CACHE_DIR / f"mcap_{ddmmyyyy}.csv"

    # Cache hit
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                log.info(f"  MCap: loaded from cache ({cache_path.name})")
                return df
        except Exception:
            pass

    # Download zip
    url = BASE_URL.format(ddmmyy=ddmmyy)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if resp.status_code != 200:
            log.debug(f"  MCap zip {ddmmyy}: HTTP {resp.status_code}")
            return None
        if len(resp.content) < 1000:
            log.debug(f"  MCap zip {ddmmyy}: response too small ({len(resp.content)} bytes)")
            return None
    except requests.RequestException as e:
        log.debug(f"  MCap zip {ddmmyy}: fetch error {e}")
        return None

    # Extract mcap CSV from zip
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the mcap file — naming: mcap06082026.csv
            mcap_files = [f for f in zf.namelist()
                          if f.lower().startswith("mcap") and f.lower().endswith(".csv")]
            if not mcap_files:
                log.warning(f"  No mcap*.csv found in {url}. Files: {zf.namelist()[:10]}")
                return None
            mcap_file = mcap_files[0]
            with zf.open(mcap_file) as f:
                df = pd.read_csv(f)
            # Cache for reuse
            df.to_csv(cache_path, index=False)
            log.info(f"  MCap: downloaded and cached ({mcap_file}, {len(df)} rows)")
            return df
    except Exception as e:
        log.warning(f"  MCap zip parse error: {e}")
        return None


def fetch_nse_mcap() -> Dict[str, float]:
    """
    Fetch MCap for all NSE EQ-series stocks from the official NSE MCap report.
    Tries up to 5 recent trading days (same lookback as BhavcopyFetcher).
    Returns {TICKER.NS: mcap_cr} using full MCap (not free-float).
    Falls back to free-float MCap if full MCap is missing/zero.
    Returns {} on complete failure.
    """
    for offset in range(5):
        candidate = _get_last_trading_day() - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue

        df = _fetch_mcap_csv(candidate)
        if df is None:
            continue

        # Normalise column names
        df.columns = df.columns.str.strip().str.upper()
        cols = list(df.columns)
        log.debug(f"  MCap CSV columns: {cols[:15]}")

        # Find symbol, series, and MCap columns defensively
        sym_col    = next((c for c in cols if "SYMBOL" in c), None)
        series_col = next((c for c in cols if c == "SERIES"), None)
        # Full MCap preferred; free-float as fallback
        mcap_col   = next((c for c in cols if "FULL" in c and "FREE" not in c
                           and "FLOAT" not in c and "MCAP" in c.replace(" ", "")), None)
        if not mcap_col:
            mcap_col = next((c for c in cols if "MCAP" in c.replace(" ", "")
                             or "MARKET" in c.upper() and "CAP" in c.upper()), None)

        if not sym_col or not mcap_col:
            log.warning(f"  MCap CSV missing expected columns. "
                        f"Have: {cols[:15]}. sym={sym_col}, mcap={mcap_col}")
            continue

        # Filter EQ series
        df[sym_col] = df[sym_col].astype(str).str.strip()
        if series_col:
            df[series_col] = df[series_col].astype(str).str.strip()
            df = df[df[series_col] == "EQ"]

        df[mcap_col] = pd.to_numeric(df[mcap_col], errors="coerce")
        df = df.dropna(subset=[mcap_col])
        df = df[df[mcap_col] > 0]

        result = {}
        for _, row in df.iterrows():
            symbol = str(row[sym_col]).strip().upper()
            if symbol:
                # NSE MCap CSV stores values in Rupees — convert to Crore (÷ 1 crore = 1e7)
                result[symbol + ".NS"] = float(row[mcap_col]) / 1e7

        if result:
            log.info(f"  MCap: {len(result)} EQ-series tickers for {candidate}")
            return result

    log.error("MCap fetch failed for last 5 trading days")
    return {}


def _ensure_table(client) -> None:
    """Idempotent table creation — mirrors sector_breadth._ensure_bhav_columns."""
    client.execute("""
        CREATE TABLE IF NOT EXISTS ticker_mcap (
            ticker      TEXT PRIMARY KEY,
            mcap_cr     REAL NOT NULL,
            fetched_at  TEXT NOT NULL
        )
    """)


def publish_to_turso(mcap_map: Dict[str, float]) -> int:
    from turso_sync import get_client
    now = datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Turso unavailable — {e}")
        return 0
    published = 0
    try:
        _ensure_table(client)
        for ticker, mcap_cr in mcap_map.items():
            try:
                client.execute(
                    """
                    INSERT INTO ticker_mcap (ticker, mcap_cr, fetched_at)
                    VALUES (?,?,?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        mcap_cr=excluded.mcap_cr,
                        fetched_at=excluded.fetched_at
                    """,
                    [ticker, mcap_cr, now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed {ticker}: {e}")
    finally:
        client.close()
    return published


def print_mcap_table(mcap_map: Dict[str, float]) -> None:
    sorted_t = sorted(mcap_map.items(), key=lambda x: -x[1])
    print(f"\n{'TICKER':<18} {'MCAP (Rs Cr)':>15}")
    print("-" * 35)
    for ticker, mcap in sorted_t[:30]:
        print(f"{ticker:<18} {mcap:>15,.1f}")
    if len(sorted_t) > 30:
        print(f"  ... and {len(sorted_t) - 30} more tickers")
    total = sum(mcap_map.values())
    print(f"\nTotal: Rs {total/1e5:.1f} lakh crore across {len(mcap_map)} tickers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="P3-05: Fetch NSE MCap from official PR Bhavcopy zip"
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and print, skip Turso publish")
    args = ap.parse_args()

    mcap_map = fetch_nse_mcap()

    if not mcap_map:
        log.error("No MCap data fetched")
        sys.exit(1)

    print_mcap_table(mcap_map)

    if args.dry_run:
        print(f"\n[DRY RUN] Would publish {len(mcap_map)} MCap values to Turso — skipped.")
    else:
        n = publish_to_turso(mcap_map)
        print(f"\nPublished {n}/{len(mcap_map)} MCap values to Turso ticker_mcap table.")
        print("Next: add MCap-weighted breadth toggle to sector_breadth.py (P3-05).")
