"""
publish_symbol_master.py — P1-05

Publishes the NSE equity+ETF symbol master (downloaded via
symbol_resolver.load_nse_master()) to a Turso `nse_symbol_master` table, so
Portfolio Dashboard — a separate repo/app on `libsql`, not `libsql_client` —
can look up resolved symbols by querying the shared DB directly instead of
needing this repo's Python module or its local CSV cache.

Run as a step in daily_scan.yml (after fetch_mcap.py) so the table refreshes
nightly. Idempotent UPSERT — safe to run more than once a day.

Usage:
    python publish_symbol_master.py --dry-run    # fetch and print, skip Turso
    python publish_symbol_master.py              # fetch + publish to Turso
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from symbol_resolver import load_nse_master


def _ensure_table(client) -> None:
    client.execute("""
        CREATE TABLE IF NOT EXISTS nse_symbol_master (
            symbol        TEXT PRIMARY KEY,
            company_name  TEXT,
            isin          TEXT,
            published_at  TEXT
        )
    """)


def publish_to_turso(name_map: dict, isin_map: dict) -> int:
    from turso_sync import get_client

    # name_map is name -> symbol; isin_map is isin -> symbol. Need the
    # reverse (symbol -> isin) to publish one row per symbol with both
    # its company name and ISIN — build that once here rather than a
    # per-row linear scan over isin_map for every symbol.
    symbol_to_isin = {sym: isin for isin, sym in isin_map.items()}
    symbol_to_name = {sym: name for name, sym in name_map.items()}
    all_symbols = set(symbol_to_name) | set(symbol_to_isin)

    now = datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Turso unavailable — {e}")
        return 0

    published = 0
    try:
        _ensure_table(client)
        for symbol in all_symbols:
            try:
                client.execute(
                    """
                    INSERT INTO nse_symbol_master (symbol, company_name, isin, published_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        company_name=excluded.company_name,
                        isin=excluded.isin,
                        published_at=excluded.published_at
                    """,
                    [symbol, symbol_to_name.get(symbol), symbol_to_isin.get(symbol), now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed {symbol}: {e}")
    finally:
        client.close()
    return published


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="P1-05: Publish NSE symbol master (equities + ETFs) to Turso"
    )
    ap.add_argument("--dry-run", action="store_true",
                     help="Fetch and print, skip Turso publish")
    args = ap.parse_args()

    isin_map, name_map = load_nse_master()
    if not isin_map and not name_map:
        log.error("No NSE master data fetched")
        sys.exit(1)

    n_symbols = len(set(name_map.values()) | set(isin_map.values()))
    log.info(f"Loaded {n_symbols} symbols ({len(name_map)} with company names, "
              f"{len(isin_map)} with ISINs)")

    if args.dry_run:
        for name, symbol in list(name_map.items())[:20]:
            print(f"  {symbol:<15} {name}")
        print(f"  ... and {max(0, len(name_map) - 20)} more")
        sys.exit(0)

    published = publish_to_turso(name_map, isin_map)
    log.info(f"Published {published} symbols to nse_symbol_master")
