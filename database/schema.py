"""
NSE Momentum v4.3 — Database Layer
SQLite schema with full v4.0 tables: price_history, factor_store,
pattern_occurrences, pattern_statistics, market_regime_history,
corporate_events, trades_v4, adaptive_weights, portfolio
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "data" / "momentum_v4.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_all_tables():
    """Create all v4.0 tables if they don't exist."""
    BASE_DIR.joinpath("data").mkdir(exist_ok=True)
    conn = get_connection()
    conn.executescript("""
    -- ─────────────────────────────────────────────────────────
    -- PRICE HISTORY (local OHLCV cache — 238,795 rows, 2yr)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS price_history (
        ticker      TEXT NOT NULL,
        date        TEXT NOT NULL,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        volume      INTEGER,
        adj_close   REAL,
        PRIMARY KEY (ticker, date)
    );
    CREATE INDEX IF NOT EXISTS idx_ph_ticker ON price_history(ticker);
    CREATE INDEX IF NOT EXISTS idx_ph_date   ON price_history(date);

    -- ─────────────────────────────────────────────────────────
    -- FACTOR STORE (pre-computed features per symbol per date)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS factor_store (
        ticker          TEXT NOT NULL,
        date            TEXT NOT NULL,
        rs_4w           REAL,
        rs_12w          REAL,
        rs_26w          REAL,
        rs_percentile   REAL,
        rvol            REAL,
        delivery_pct    REAL,
        ema10           REAL,
        ema21           REAL,
        ema50           REAL,
        ema200          REAL,
        macd            REAL,
        macd_signal     REAL,
        rsi14           REAL,
        atr14           REAL,
        liquidity_adt   REAL,
        sector_rank     INTEGER,
        final_score     INTEGER,
        regime          TEXT,
        PRIMARY KEY (ticker, date)
    );

    -- ─────────────────────────────────────────────────────────
    -- PATTERN OCCURRENCES (historical pattern detection log)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS pattern_occurrences (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        date            TEXT,
        pattern         TEXT,
        breakout_level  REAL,
        score           INTEGER,
        regime          TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_po_ticker  ON pattern_occurrences(ticker);
    CREATE INDEX IF NOT EXISTS idx_po_pattern ON pattern_occurrences(pattern);

    -- ─────────────────────────────────────────────────────────
    -- PATTERN STATISTICS (win rates from backtesting)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS pattern_statistics (
        pattern         TEXT PRIMARY KEY,
        total_signals   INTEGER DEFAULT 0,
        wins            INTEGER DEFAULT 0,
        losses          INTEGER DEFAULT 0,
        total_r         REAL    DEFAULT 0,
        win_rate        REAL    DEFAULT 0,
        avg_r           REAL    DEFAULT 0,
        profit_factor   REAL    DEFAULT 0,
        last_updated    TEXT
    );

    -- ─────────────────────────────────────────────────────────
    -- MARKET REGIME HISTORY
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS market_regime_history (
        date            TEXT PRIMARY KEY,
        regime          TEXT,
        breadth_score   INTEGER,
        ad_ratio        REAL,
        above_50_pct    REAL,
        new_highs       INTEGER,
        new_lows        INTEGER,
        nifty_close     REAL,
        vix             REAL
    );

    -- ─────────────────────────────────────────────────────────
    -- CORPORATE EVENTS (NSE announcements / earnings)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS corporate_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        event_date      TEXT,
        event_type      TEXT,
        headline        TEXT,
        score_impact    INTEGER DEFAULT 0,
        fetched_at      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ce_ticker ON corporate_events(ticker);

    -- ─────────────────────────────────────────────────────────
    -- TRADES v4 (live trade log — same as trade_logger.py)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS trades_v4 (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        name            TEXT,
        sector          TEXT,
        universe        TEXT,
        pattern         TEXT,
        entry_date      TEXT,
        exit_date       TEXT,
        entry_price     REAL,
        stop_loss       REAL,
        target1         REAL,
        target2         REAL,
        rrr             REAL,
        exit_price      REAL,
        exit_type       TEXT,
        r_multiple      REAL,
        pnl_pct         REAL,
        total_score     INTEGER,
        confidence_pct  REAL,
        regime          TEXT,
        breadth_score   INTEGER,
        status          TEXT DEFAULT 'OPEN',
        notes           TEXT
    );

    -- ─────────────────────────────────────────────────────────
    -- DYNAMIC WEIGHTS (pattern weight evolution)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS dynamic_weights (
        pattern         TEXT PRIMARY KEY,
        weight          INTEGER,
        win_rate        REAL,
        expectancy_r    REAL,
        profit_factor   REAL,
        sample_count    INTEGER,
        last_updated    TEXT,
        cycles          INTEGER DEFAULT 0,
        reason          TEXT
    );

    -- ─────────────────────────────────────────────────────────
    -- ADAPTIVE WEIGHTS HISTORY (per-cycle audit trail)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS adaptive_weights (
        pattern         TEXT,
        cycle           INTEGER,
        weight          INTEGER,
        win_rate        REAL,
        expectancy_r    REAL,
        updated_at      TEXT,
        PRIMARY KEY (pattern, cycle)
    );

    -- ─────────────────────────────────────────────────────────
    -- PORTFOLIO (open positions tracker)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS portfolio (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        name            TEXT,
        entry_date      TEXT,
        entry_price     REAL,
        stop_loss       REAL,
        target1         REAL,
        target2         REAL,
        position_size   REAL,
        position_pct    REAL,
        sector          TEXT,
        universe        TEXT,
        pattern         TEXT,
        total_score     INTEGER,
        status          TEXT DEFAULT 'OPEN'
    );
    """)
    conn.commit()
    conn.close()
    log.info(f"All v4.0 tables initialised at {DB_PATH}")


def get_table_stats() -> dict:
    """Return row counts for all tables."""
    conn = get_connection()
    tables = [
        "price_history", "factor_store", "pattern_occurrences",
        "pattern_statistics", "market_regime_history", "corporate_events",
        "trades_v4", "dynamic_weights", "portfolio"
    ]
    stats = {}
    for t in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
            stats[t] = row[0]
        except Exception:
            stats[t] = 0
    conn.close()
    return stats


if __name__ == "__main__":
    init_all_tables()
    stats = get_table_stats()
    print("\nDatabase table row counts:")
    for t, n in stats.items():
        print(f"  {t:<30} {n:>8,} rows")
