"""
NSE Momentum v5.0 - Trade Logger & Learning Agent
SQLite persistence for all trades. Dynamic reweighting every 25 closed trades.
Pattern leaderboard with win rate + expectancy + profit factor.
DB: data/momentum_v5.db (unified — no more v4/v5 split)
"""

import sqlite3, logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "data" / "momentum_v5.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_tables():
    BASE_DIR.joinpath("data").mkdir(exist_ok=True)
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades_v4 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, name TEXT, sector TEXT, universe TEXT,
        pattern TEXT,
        entry_date TEXT, exit_date TEXT,
        entry_price REAL, stop_loss REAL, target1 REAL, target2 REAL, rrr REAL,
        exit_price REAL, exit_type TEXT,
        r_multiple REAL, pnl_pct REAL,
        total_score INTEGER, confidence_pct REAL,
        regime TEXT, breadth_score INTEGER,
        status TEXT DEFAULT 'open',
        notes TEXT,
        breakout_quality TEXT DEFAULT '',
        macro_state TEXT DEFAULT 'MIXED',
        event_risk TEXT DEFAULT 'NORMAL',
        confirmation_state TEXT DEFAULT 'SETUP_READY',
        headroom_pct REAL DEFAULT 0.0,
        vcp_w4 REAL DEFAULT 0.0,
        earnings_flag INTEGER DEFAULT 0,
        asymmetry_risk_pct REAL DEFAULT 0.0,
        asymmetry_reward_pct REAL DEFAULT 0.0,
        asymmetry_rr REAL DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS dynamic_weights (
        pattern TEXT PRIMARY KEY,
        weight INTEGER,
        win_rate REAL,
        expectancy_r REAL,
        profit_factor REAL,
        sample_count INTEGER,
        last_updated TEXT,
        cycles INTEGER DEFAULT 0,
        reason TEXT
    );

    CREATE TABLE IF NOT EXISTS pattern_statistics (
        pattern TEXT PRIMARY KEY,
        total_signals INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        total_r REAL DEFAULT 0,
        last_updated TEXT
    );

    CREATE TABLE IF NOT EXISTS market_regime_history (
        date TEXT PRIMARY KEY,
        regime TEXT,
        breadth_score INTEGER,
        ad_ratio REAL,
        above_50_pct REAL
    );

    CREATE TABLE IF NOT EXISTS adaptive_weights (
        pattern TEXT, cycle INTEGER, weight INTEGER, win_rate REAL,
        expectancy_r REAL, PRIMARY KEY (pattern, cycle)
    );

    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, entry_date TEXT, entry_price REAL,
        stop_loss REAL, target1 REAL, target2 REAL,
        position_size REAL, sector TEXT, universe TEXT,
        status TEXT DEFAULT 'open'
    );
    """)
    conn.commit()
    conn.close()
    _seed_weights()


def _seed_weights():
    SEED = {
        "High Tight Flag":      (17, 0.78, 0.55, 2.8),
        "Swing High Breakout":  (17, 0.72, 0.51, 2.4),
        "VCP":                  (16, 0.68, 0.48, 2.2),
        "3-Weeks-Tight":        (15, 0.67, 0.45, 2.1),
        "High Base":            (14, 0.65, 0.43, 2.0),
        "Base Breakout":        (14, 0.64, 0.42, 1.9),
        "Rounded Base":         (13, 0.63, 0.41, 1.9),
        "Double Bottom":        (15, 0.62, 0.38, 1.8),
        "Cup & Handle":         (11, 0.61, 0.37, 1.8),
        "Ascending Triangle":   (13, 0.60, 0.36, 1.7),
        "IPO Base":             (15, 0.60, 0.40, 1.8),
        "Diamond Bottom":       (14, 0.54, 0.28, 1.5),
        "Flat Base":            (10, 0.58, 0.33, 1.7),
        "Volume Expansion":     (12, 0.57, 0.30, 1.6),
        "52W Momentum":         (13, 0.56, 0.29, 1.6),
        "Bull Flag":            (12, 0.55, 0.25, 1.5),
        "Symmetrical Triangle": (11, 0.53, 0.22, 1.4),
        "Descending Wedge":     (12, 0.52, 0.20, 1.4),
        "Falling Wedge":         (8, 0.50, 0.15, 1.3),
    }
    conn = _conn()
    for pat, (wt, wr, exp, pf) in SEED.items():
        exists = conn.execute(
            "SELECT 1 FROM dynamic_weights WHERE pattern=?", (pat,)
        ).fetchone()
        if not exists:
            conn.execute("""
                INSERT INTO dynamic_weights
                (pattern, weight, win_rate, expectancy_r, profit_factor,
                 sample_count, last_updated, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pat, wt, wr, exp, pf, 264,
                  datetime.today().strftime("%Y-%m-%d"),
                  "Seed: 264 training examples"))
    conn.commit()
    conn.close()


def get_dynamic_weight(pattern: str) -> int:
    conn = _conn()
    row = conn.execute(
        "SELECT weight FROM dynamic_weights WHERE pattern=?", (pattern,)
    ).fetchone()
    conn.close()
    from agents.pattern_agent import DEFAULT_WEIGHTS
    default = DEFAULT_WEIGHTS.get(pattern, 12)
    return int(row["weight"]) if row else default


def log_entry(ticker: str, name: str, sector: str, universe: str,
              pattern: str, entry: float, stop: float,
              t1: float, t2: float, rrr: float,
              score: int, confidence: float,
              regime: str = "C", breadth: int = 5,
              notes: str = "") -> int:
    conn = _conn()
    cur = conn.execute("""
        INSERT INTO trades_v4
        (ticker, name, sector, universe, pattern, entry_date, entry_price,
         stop_loss, target1, target2, rrr, total_score, confidence_pct,
         regime, breadth_score, status, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)
    """, (ticker, name, sector, universe, pattern,
          datetime.today().strftime("%Y-%m-%d"),
          entry, stop, t1, t2, rrr, score, confidence, regime, breadth, notes))
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("Trade logged: %s #%d | Entry: %s | SL: %s | T1: %s",
             ticker, trade_id, entry, stop, t1)
    return trade_id


def log_exit(trade_id: int, exit_price: float, exit_type: str = "Manual") -> dict:
    conn = _conn()
    trade = conn.execute(
        "SELECT * FROM trades_v4 WHERE id=? AND status='open'", (trade_id,)
    ).fetchone()
    if not trade:
        conn.close()
        return {"error": f"Trade {trade_id} not found or already closed"}

    entry  = float(trade["entry_price"])
    stop   = float(trade["stop_loss"])
    risk   = entry - stop
    r_mult = round((exit_price - entry) / risk, 2) if risk > 0 else 0.0
    pnl    = round((exit_price - entry) / entry * 100, 2)

    conn.execute("""
        UPDATE trades_v4
        SET exit_date=?, exit_price=?, exit_type=?, r_multiple=?, pnl_pct=?, status='CLOSED'
        WHERE id=?
    """, (datetime.today().strftime("%Y-%m-%d"), exit_price, exit_type,
          r_mult, pnl, trade_id))
    conn.commit()

    closed_count = conn.execute(
        "SELECT COUNT(*) FROM trades_v4 WHERE status='CLOSED'"
    ).fetchone()[0]
    conn.close()

    if closed_count % 25 == 0:
        _run_dynamic_weighting()

    return {"trade_id": trade_id, "r_multiple": r_mult, "pnl_pct": pnl}


def _run_dynamic_weighting():
    conn = _conn()
    patterns = conn.execute(
        "SELECT DISTINCT pattern FROM trades_v4 WHERE status='CLOSED'"
    ).fetchall()
    for row in patterns:
        pat = row["pattern"]
        trades = conn.execute("""
            SELECT r_multiple FROM trades_v4
            WHERE pattern=? AND status='CLOSED' AND r_multiple IS NOT NULL
        """, (pat,)).fetchall()
        if len(trades) < 5:
            continue
        rs     = [t["r_multiple"] for t in trades]
        wins   = sum(1 for r in rs if r > 0)
        wr     = wins / len(rs)
        exp_r  = sum(rs) / len(rs)
        wins_r = sum(r for r in rs if r > 0)
        loss_r = abs(sum(r for r in rs if r < 0))
        pf     = wins_r / loss_r if loss_r > 0 else 2.0
        base   = 10
        new_wt = base + int((wr - 0.50) / 0.05)
        new_wt = max(5, min(18, new_wt))
        cur_row = conn.execute(
            "SELECT weight FROM dynamic_weights WHERE pattern=?", (pat,)
        ).fetchone()
        if cur_row:
            cur_wt = int(cur_row["weight"])
            cap    = int(cur_wt * 0.30)
            new_wt = max(cur_wt - cap, min(cur_wt + cap, new_wt))
        reason = f"n={len(rs)} | WR={wr:.0%} | Exp={exp_r:.2f}R | PF={pf:.1f}"
        conn.execute("""
            INSERT OR REPLACE INTO dynamic_weights
            (pattern, weight, win_rate, expectancy_r, profit_factor,
             sample_count, last_updated, reason)
            VALUES (?,?,?,?,?,?,?,?)
        """, (pat, new_wt, wr, exp_r, pf, len(rs),
              datetime.today().strftime("%Y-%m-%d"), reason))
    conn.commit()
    conn.close()
    log.info("Dynamic weights updated from live trade data")


def auto_log_t1_picks(t1_results: list, regime: str = "B") -> int:
    """
    Auto-log Tier 1 StockResult objects to trades_v4 after each evening scan.
    Column names match the actual trades_v4 schema exactly.
    """
    if not t1_results:
        return 0

    today  = date.today().isoformat()
    logged = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        # Add any missing v5 columns to existing table
        v5_cols = [
            ("breakout_quality",     "TEXT    DEFAULT ''"),
            ("macro_state",          "TEXT    DEFAULT 'MIXED'"),
            ("event_risk",           "TEXT    DEFAULT 'NORMAL'"),
            ("confirmation_state",   "TEXT    DEFAULT 'SETUP_READY'"),
            ("headroom_pct",         "REAL    DEFAULT 0.0"),
            ("vcp_w4",               "REAL    DEFAULT 0.0"),
            ("earnings_flag",        "INTEGER DEFAULT 0"),
            ("asymmetry_risk_pct",   "REAL    DEFAULT 0.0"),
            ("asymmetry_reward_pct", "REAL    DEFAULT 0.0"),
            ("asymmetry_rr",         "REAL    DEFAULT 0.0"),
        ]
        existing = {r[1] for r in cur.execute("PRAGMA table_info(trades_v4)")}
        for col_name, col_def in v5_cols:
            if col_name not in existing:
                cur.execute(f"ALTER TABLE trades_v4 ADD COLUMN {col_name} {col_def}")

        for r in t1_results:
            # Skip duplicates
            cur.execute(
                "SELECT id FROM trades_v4 WHERE ticker=? AND entry_date=?",
                (r.ticker, today)
            )
            if cur.fetchone():
                log.debug("Already logged %s for %s", r.ticker, today)
                continue

            cur.execute("""
                INSERT INTO trades_v4 (
                    ticker, name, sector, universe, pattern,
                    entry_date, entry_price, stop_loss, target1, target2, rrr,
                    total_score, confidence_pct, regime, breadth_score,
                    status, notes,
                    breakout_quality, macro_state, event_risk,
                    confirmation_state, headroom_pct, vcp_w4,
                    earnings_flag, asymmetry_risk_pct,
                    asymmetry_reward_pct, asymmetry_rr
                ) VALUES (
                    ?,?,?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,
                    ?,?,?,
                    ?,?,?,
                    ?,?,?,?
                )
            """, (
                r.ticker,
                r.name,
                r.sector,
                getattr(r, "universe", "LARGE"),
                r.pattern,
                today,
                r.entry,
                r.stop_loss,
                r.target1,
                r.target2,
                r.rrr,
                r.total_score,
                r.confidence_pct,
                regime,
                getattr(r, "breadth_score", 0),
                "open",
                f"Auto-logged v5 | {r.pattern} | Score {r.total_score}",
                getattr(r, "breakout_quality", "MINOR"),
                getattr(r, "macro_state", "MIXED"),
                getattr(r, "event_risk", "NORMAL"),
                getattr(r, "confirmation_state", "SETUP_READY"),
                getattr(r, "headroom_pct", 0.0),
                getattr(r, "vcp_w4_pct", 0.0),
                1 if getattr(r, "earnings_flag", False) else 0,
                getattr(r, "asymmetry_risk_pct", 0.0),
                getattr(r, "asymmetry_reward_pct", 0.0),
                getattr(r, "asymmetry_rr", 0.0),
            ))
            logged += 1

        conn.commit()
        log.info("auto_log_t1_picks: logged %d picks for %s", logged, today)

    except Exception as e:
        log.error("auto_log_t1_picks DB error: %s", e)
        raise
    finally:
        if 'conn' in locals():
            conn.close()

    return logged


def print_stats():
    conn = _conn()
    patterns = conn.execute("""
        SELECT pattern,
               COUNT(*) as n,
               SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(AVG(r_multiple), 2) as avg_r,
               ROUND(SUM(CASE WHEN r_multiple > 0 THEN r_multiple ELSE 0 END) /
                     MAX(ABS(SUM(CASE WHEN r_multiple < 0 THEN r_multiple ELSE 0 END)), 0.001), 2) as pf
        FROM trades_v4
        WHERE status='CLOSED' AND r_multiple IS NOT NULL
        GROUP BY pattern ORDER BY avg_r DESC
    """).fetchall()
    conn.close()
    print("\n" + "="*70)
    print(f"  PATTERN LEADERBOARD - {datetime.today().strftime('%d %b %Y')}")
    print("="*70)
    print(f"  {'PATTERN':<25} {'N':>4} {'WIN%':>6} {'EXP R':>7} {'PF':>5}")
    print("  " + "-"*60)
    for p in patterns:
        wr = (p["wins"] / p["n"] * 100) if p["n"] > 0 else 0
        print(f"  {p['pattern']:<25} {p['n']:>4} {wr:>5.0f}% {p['avg_r']:>7.2f}R {p['pf']:>5.1f}")
    print("="*70 + "\n")


def get_open_trades() -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM trades_v4 WHERE status='open' ORDER BY entry_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_summary() -> dict:
    conn = _conn()
    row = conn.execute("""
        SELECT COUNT(*) as n,
               ROUND(AVG(r_multiple), 2) as avg_r,
               SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(pnl_pct), 2) as total_pnl_pct
        FROM trades_v4 WHERE status='CLOSED' AND r_multiple IS NOT NULL
    """).fetchone()
    conn.close()
    if not row or row["n"] == 0:
        return {"n": 0, "avg_r": 0, "win_rate": 0, "total_pnl_pct": 0}
    return {
        "n": row["n"],
        "avg_r": row["avg_r"],
        "win_rate": round(row["wins"] / row["n"] * 100, 1),
        "total_pnl_pct": row["total_pnl_pct"],
    }
