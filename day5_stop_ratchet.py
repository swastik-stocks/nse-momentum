"""
NSE Momentum v5.0 - Day+5 Stop Ratchet
Run each morning after market open to update stops on open trades.

Logic:
  For each OPEN trade in trades_v4 that is 5+ sessions old:
  - Fetch current EMA21 from Yahoo Finance
  - If EMA21 > original stop_loss: ratchet stop up to EMA21
  - Log the update to notes

Usage:
    python day5_stop_ratchet.py

Run this as part of the morning routine after dhan_morning_scanner.py.
"""

import sqlite3
import logging
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / "data" / "momentum_v5.db"


def get_ema21(ticker: str) -> float:
    """Fetch current EMA21 from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 21:
            return 0.0
        close = df["Close"].squeeze()
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        return round(ema21, 2)
    except Exception as e:
        log.debug("EMA21 fetch failed for %s: %s", ticker, e)
        return 0.0


def run_ratchet(min_hold_days: int = 5) -> int:
    """
    Update stops on open trades that have held for min_hold_days.
    Returns number of stops updated.
    """
    if not DB_PATH.exists():
        print("No database found.")
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    cutoff = (date.today() - timedelta(days=min_hold_days)).isoformat()
    cur.execute("""
        SELECT id, ticker, entry_price, stop_loss, entry_date, notes
        FROM trades_v4
        WHERE status = 'open'
          AND entry_date <= ?
        ORDER BY entry_date ASC
    """, (cutoff,))

    trades = [dict(r) for r in cur.fetchall()]
    updated = 0

    print(f"\nDay+5 Stop Ratchet - {date.today().isoformat()}")
    print(f"Checking {len(trades)} open trade(s) >= {min_hold_days} days old...")
    print()

    for trade in trades:
        ticker     = trade["ticker"]
        orig_stop  = float(trade["stop_loss"])
        entry      = float(trade["entry_price"])
        trade_id   = trade["id"]
        entry_date = trade["entry_date"]

        ema21 = get_ema21(ticker)
        if ema21 <= 0:
            print(f"  {ticker:<20} EMA21 fetch failed — skip")
            continue

        hold = (date.today() - date.fromisoformat(entry_date)).days

        if ema21 > orig_stop and ema21 < entry:
            # Ratchet: EMA21 is above original stop but below entry
            new_stop = round(ema21 * 0.993, 2)   # 0.7% buffer below EMA21
            note = (f"Day+{hold} ratchet {date.today().isoformat()}: "
                    f"stop {orig_stop} -> {new_stop} (EMA21={ema21})")
            cur.execute("""
                UPDATE trades_v4
                SET stop_loss = ?,
                    notes = COALESCE(notes, '') || ' | ' || ?
                WHERE id = ?
            """, (new_stop, note, trade_id))
            print(f"  {ticker:<20} Stop RATCHETED: {orig_stop} -> {new_stop}"
                  f"  (EMA21={ema21}, hold={hold}d)")
            updated += 1
        elif ema21 >= entry:
            # EMA21 caught up to entry — trail at EMA21
            new_stop = round(ema21 * 0.993, 2)
            note = (f"Day+{hold} trail {date.today().isoformat()}: "
                    f"stop {orig_stop} -> {new_stop} (EMA21={ema21} >= entry)")
            cur.execute("""
                UPDATE trades_v4
                SET stop_loss = ?,
                    notes = COALESCE(notes, '') || ' | ' || ?
                WHERE id = ?
            """, (new_stop, note, trade_id))
            print(f"  {ticker:<20} Stop TRAILED:   {orig_stop} -> {new_stop}"
                  f"  (EMA21={ema21} >= entry, hold={hold}d)")
            updated += 1
        else:
            print(f"  {ticker:<20} No change  "
                  f"(EMA21={ema21} <= stop={orig_stop}, hold={hold}d)")

    conn.commit()
    conn.close()

    print(f"\nUpdated {updated} of {len(trades)} open trade(s).")
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_ratchet(min_hold_days=5)
