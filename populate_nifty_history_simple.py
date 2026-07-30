"""
Minimal market_regime_history populator — unblocks pipeline_replay_hourly.py
without requiring the full regime_backfill_deep.py chain (which needs
price_history_deep, currently unavailable — see today's session notes).

WHAT THIS DOES: fetches real NIFTY 50 daily close history via yfinance
(same reliable source regime_backfill_deep.py itself successfully used
before it crashed on the LATER breadth-proxy step) and writes it into
market_regime_history with nifty_close populated for every trading day.

WHAT THIS DELIBERATELY DOES NOT DO: compute a real breadth-based regime
letter (A-E). That requires price_history_deep's close matrix, which
isn't available right now. Every date gets regime='C' (neutral) instead
of a fabricated classification — honest placeholder, not a real signal.
This means pipeline_replay_hourly.py's regime penalty will be a flat,
uninformative -5 for every trade rather than a genuine per-date signal.
That's a real, known limitation of today's run — not hidden, and does
NOT affect the RS benchmark comparison (which only needs nifty_close,
not the regime letter) or the core pattern-edge question this replay
exists to answer.

Usage:
    python populate_nifty_history_simple.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, init_all_tables

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run():
    init_all_tables()  # ensures market_regime_history exists (v4.0 schema)

    log.info("Fetching full available NIFTY 50 history (period=max)...")
    nifty = yf.Ticker("^NSEI").history(period="max", interval="1d")
    if nifty.empty:
        raise RuntimeError("yfinance returned no NIFTY data — check network/ticker symbol.")

    log.info(f"NIFTY history: {nifty.index[0].date()} to {nifty.index[-1].date()} "
             f"({len(nifty)} days)")

    conn = get_connection()
    rows = [
        (d.strftime("%Y-%m-%d"), "C", 0, 0.0, 0.0, 0, 0, float(row["Close"]), 15.0)
        for d, row in nifty.iterrows()
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO market_regime_history
        (date, regime, breadth_score, ad_ratio, above_50_pct, new_highs, new_lows, nifty_close, vix)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM market_regime_history").fetchone()[0]
    conn.close()

    log.info(f"market_regime_history now has {count:,} rows "
             f"(nifty_close populated, regime='C' neutral placeholder for all dates — "
             f"see module docstring for why)")


if __name__ == "__main__":
    run()
