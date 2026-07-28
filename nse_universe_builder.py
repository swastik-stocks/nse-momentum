"""
NSE Momentum v6 — Universe Builder
Regenerates nse_universe.py from live NSE index constituent lists instead
of the hand-typed, drifted static list (which had confirmed duplicates —
INDIACEM.NS, TMPV.NS — and an empty Smallcap 250 tier: 0 stocks in every
scan log).

Data source: NSE's own published index constituent CSVs (ind_nifty100list,
ind_niftymidcap150list, ind_niftysmallcap250list), fetched via the same
session-warmup pattern already used in agents/market_breadth_agent.py
(NSE requires a homepage-cookie warmup before serving API/CSV requests).

Sector tags: preserved from the EXISTING nse_universe.py for tickers
already present there. Newly-added tickers get sector="Unknown" and are
printed in a review list at the end — sector tagging for new names should
be filled in by hand or via a follow-up enrichment pass, not guessed here.

Usage:
    python nse_universe_builder.py            # dry run, prints summary only
    python nse_universe_builder.py --write     # regenerates nse_universe.py
                                                 (backs up the old one first)
"""

import sys
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(__file__).parent
UNIVERSE_FILE = BASE_DIR / "nse_universe.py"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

INDEX_CSV_URLS = {
    "LARGE": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "MID":   "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "SMALL": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}


def _get_nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=10)
    except Exception as e:
        log.warning(f"NSE session warmup failed: {e}")
    return s


def _fetch_index_constituents(session: requests.Session, tier: str, url: str) -> pd.DataFrame:
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        log.info(f"  {tier}: {len(df)} constituents fetched from {url.split('/')[-1]}")
        return df
    except Exception as e:
        log.error(f"  {tier}: fetch FAILED ({e}) — this tier will be empty this run")
        return pd.DataFrame()


def _load_existing_sector_map() -> dict:
    """Parse the CURRENT nse_universe.py (if present) to preserve sector
    tags for tickers we already know about, rather than re-guessing them."""
    sector_map = {}
    if not UNIVERSE_FILE.exists():
        return sector_map
    try:
        ns = {}
        exec(UNIVERSE_FILE.read_text(encoding="utf-8"), ns)
        for row in ns.get("NSE_UNIVERSE", []):
            ticker, name, sector, tier = row[0], row[1], row[2], row[3]
            sector_map[ticker] = sector
    except Exception as e:
        log.warning(f"Could not parse existing nse_universe.py for sector reuse: {e}")
    return sector_map


def build_universe() -> tuple[list, list]:
    """
    Returns (universe_rows, new_tickers_needing_sector_review).
    universe_rows: list of (ticker, name, sector, tier) tuples, deduped,
    ready to write into NSE_UNIVERSE.
    """
    session = _get_nse_session()
    sector_map = _load_existing_sector_map()

    # Priority order matters: a stock could technically appear in more
    # than one list during index reshuffles — LARGE takes priority over
    # MID over SMALL, since NIFTY100 members shouldn't be double-counted
    # in Midcap150 in practice, but this guards against transition noise.
    seen: dict[str, str] = {}   # symbol -> tier already assigned
    rows: list = []
    new_tickers: list = []

    for tier in ["LARGE", "MID", "SMALL"]:
        df = _fetch_index_constituents(session, tier, INDEX_CSV_URLS[tier])
        if df.empty:
            continue

        symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
        name_col   = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        industry_col = next((c for c in df.columns if "industry" in c.lower()), None)

        if not symbol_col:
            log.error(f"  {tier}: no SYMBOL column found in CSV — columns were {list(df.columns)}")
            continue

        for _, r in df.iterrows():
            symbol = str(r[symbol_col]).strip()
            if not symbol or symbol in seen:
                continue
            seen[symbol] = tier
            ticker = f"{symbol}.NS"
            name   = str(r[name_col]).strip() if name_col else symbol
            sector = sector_map.get(ticker)
            if sector is None:
                sector = str(r[industry_col]).strip() if industry_col else "Unknown"
                new_tickers.append((ticker, name, sector))
            rows.append((ticker, name, sector, tier))

    return rows, new_tickers


def write_universe_file(rows: list) -> None:
    """Regenerate nse_universe.py, preserving UNIVERSE_CONFIG/UNIVERSE_SEED
    exactly as they are and only replacing the NSE_UNIVERSE list itself."""
    if not UNIVERSE_FILE.exists():
        log.error("nse_universe.py not found — cannot preserve UNIVERSE_CONFIG/SEED. Aborting write.")
        return

    original = UNIVERSE_FILE.read_text(encoding="utf-8")

    # Backup
    backup_path = BASE_DIR / f"nse_universe.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path.write_text(original, encoding="utf-8")
    log.info(f"Backed up existing nse_universe.py to {backup_path.name}")

    # Preserve everything up to (not including) the NSE_UNIVERSE = [ ... ] block
    marker = "NSE_UNIVERSE = ["
    idx = original.find(marker)
    if idx == -1:
        log.error("Could not find 'NSE_UNIVERSE = [' marker in existing file — aborting write.")
        return
    header = original[:idx]

    lines = [header.rstrip() + "\n\nNSE_UNIVERSE = ["]
    for ticker, name, sector, tier in rows:
        # Basic escaping for any stray quotes in names
        safe_name = name.replace('"', "'")
        lines.append(f'    ("{ticker}", "{safe_name}", "{sector}", "{tier}"),')
    lines.append("]\n")

    UNIVERSE_FILE.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {len(rows)} deduped tickers to nse_universe.py")


def print_summary(rows: list, new_tickers: list) -> None:
    from collections import Counter
    tier_counts = Counter(r[3] for r in rows)

    print("\n" + "=" * 60)
    print("  UNIVERSE BUILD SUMMARY")
    print("=" * 60)
    for tier in ["LARGE", "MID", "SMALL"]:
        print(f"  {tier:<8} {tier_counts.get(tier, 0):>4} stocks")
    print(f"  {'TOTAL':<8} {len(rows):>4} stocks  (deduped)")
    print("=" * 60)

    if new_tickers:
        print(f"\n  {len(new_tickers)} NEW tickers not in the previous universe "
              f"(sector needs manual review):")
        for ticker, name, sector in new_tickers[:30]:
            print(f"    {ticker:<16} {name:<35} sector='{sector}'")
        if len(new_tickers) > 30:
            print(f"    ... and {len(new_tickers) - 30} more")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                         help="Actually regenerate nse_universe.py (default: dry run)")
    args = parser.parse_args()

    rows, new_tickers = build_universe()
    print_summary(rows, new_tickers)

    if args.write:
        write_universe_file(rows)
        print("nse_universe.py updated. Restart any running scanner/backtest processes.")
    else:
        print("Dry run only — no files changed. Re-run with --write to apply.")
