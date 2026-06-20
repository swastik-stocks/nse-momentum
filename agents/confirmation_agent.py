"""
NSE Momentum v5.0 - ConfirmationAgent
Distinguishes SETUP_READY from BREAKOUT_CONFIRMED.
Checks trades_v4 for stocks already logged to see if breakout held.
New triggers default to SETUP_READY (3 pts).
Confirmed breakouts score 6 pts.
"""

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "momentum_v5.db"


class ConfirmationAgent:
    def __init__(self, ticker: str, entry: float, stop: float,
                 breakout_level: float):
        self.ticker   = ticker
        self.entry    = entry
        self.stop     = stop
        self.breakout = breakout_level
        self._state   = "SETUP_READY"
        self._score   = 3
        self._check()

    def _check(self):
        """
        Look up trades_v4 for this ticker.
        If logged in last 3 sessions and breakout held -> CONFIRMED.
        """
        try:
            if not DB_PATH.exists():
                return
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cutoff = (date.today() - timedelta(days=5)).isoformat()
            cur.execute("""
                SELECT entry, stop, scan_date, confirmation_state
                FROM trades_v4
                WHERE ticker=? AND scan_date >= ? AND outcome='OPEN'
                ORDER BY scan_date DESC LIMIT 1
            """, (self.ticker, cutoff))
            row = cur.fetchone()
            conn.close()

            if not row:
                return   # New trigger today — SETUP_READY

            logged_entry, logged_stop, logged_date, conf_state = row

            # If already confirmed in DB, use that
            if conf_state == "BREAKOUT_CONFIRMED":
                self._state = "BREAKOUT_CONFIRMED"
                self._score = 6
                return

            # Check if current price is above breakout and stop not violated
            if (self.entry > self.breakout > 0 and
                    self.entry > logged_stop):
                self._state = "BREAKOUT_CONFIRMED"
                self._score = 6

        except Exception as e:
            log.debug("ConfirmationAgent error for %s: %s", self.ticker, e)

    def get_state(self) -> str:
        return self._state

    def get_score(self) -> int:
        return self._score
