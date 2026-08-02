"""
turso_sync.py — P1-02

Publishes NSE Momentum's evening-scan output to Turso, the shared data
bridge with Portfolio Dashboard (see INTEGRATION_BACKLOG1.md Phase 1).

Uses libsql_client (pure-Python, HTTP-based) rather than the libsql
package Portfolio Dashboard's db.py uses. Deliberate choice, not a
downgrade: libsql has no Windows wheel for Python 3.14 yet (compiles
from Rust source, which crashes on this machine — same class of issue
as the numba/pandas-ta problem tracked in P4-10). libsql_client ships
a pure-Python wheel (no native extension), and for a write-only batch
publisher with no need for a local embedded replica, an HTTP client is
actually the more appropriate tool anyway — not a compromise.

Note: libsql_client's own GitHub repo is archived (successor work has
moved to tursodatabase/libsql-python, i.e. the same libsql package that
doesn't build here) -- still functional and installable from PyPI today,
but worth re-checking if this ever needs replacing.

Usage:
    python turso_sync.py --test        # verify credentials, no writes
    python turso_sync.py --init-schema # create P1-01 tables (idempotent)
"""

import argparse
import os
import sys

import libsql_client
from dotenv import load_dotenv

load_dotenv()

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS scanner_signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        scan_date       TEXT NOT NULL,
        pattern         TEXT,
        total_score     INTEGER,
        confidence_pct  REAL,
        entry_price     REAL,
        pivot           REAL,
        stop_loss       REAL,
        target1         REAL,
        target2         REAL,
        rrr             REAL,
        rs_percentile   REAL,
        regime          TEXT,
        breadth_score   INTEGER,
        tier            TEXT,
        published_at    TEXT,
        UNIQUE(ticker, scan_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS position_actions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        action_date     TEXT NOT NULL,
        action_type     TEXT NOT NULL,
        reason          TEXT,
        trigger_price   REAL,
        blended_cost    REAL,
        new_stop        REAL,
        published_at    TEXT,
        UNIQUE(ticker, action_date, action_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sector_breadth (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        sector              TEXT NOT NULL,
        breadth_date        TEXT NOT NULL,
        pct_above_sma20     REAL,
        pct_above_sma50     REAL,
        pct_above_sma100    REAL,
        pct_above_rsi50     REAL,
        pct_above_rs55      REAL,
        stock_count         INTEGER,
        published_at        TEXT,
        UNIQUE(sector, breadth_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS industry_breadth (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        industry            TEXT NOT NULL,
        breadth_date        TEXT NOT NULL,
        pct_above_sma20     REAL,
        pct_above_sma50     REAL,
        pct_above_sma100    REAL,
        pct_above_rsi50     REAL,
        pct_above_rs55      REAL,
        stock_count         INTEGER,
        published_at        TEXT,
        UNIQUE(industry, breadth_date)
    )
    """,
]


def get_client():
    if not TURSO_URL or not TURSO_TOKEN:
        sys.exit(
            "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not found in .env — "
            "add them before running turso_sync.py."
        )
    # libsql_client wants the http(s) scheme for the sync client, not libsql://
    url = TURSO_URL.replace("libsql://", "https://")
    return libsql_client.create_client_sync(url, auth_token=TURSO_TOKEN)


def test_connection():
    client = get_client()
    try:
        rs = client.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        print(f"Connected OK. {len(rs.rows)} existing tables in this Turso database:")
        for row in rs.rows:
            print(f"   {row[0]}")
    finally:
        client.close()


def init_schema():
    client = get_client()
    try:
        for stmt in SCHEMA_STATEMENTS:
            client.execute(stmt)
        print("Schema created (or already existed) — scanner_signals, position_actions, "
              "sector_breadth, industry_breadth.")
    finally:
        client.close()


def publish_signals(results, regime: str, scan_date: str = None):
    """
    P1-03: publish today's T1+T2 picks to scanner_signals.

    `results` — list of StockResult objects (or anything with matching
    attributes) from orchestrator.py's t1_accepted + t2. Upserts on
    (ticker, scan_date), so safe to call more than once for the same day
    (e.g. a re-run) without creating duplicates.

    Never raises past the caller if an individual row fails to publish —
    logs and skips that row, continues with the rest. Per P1-06: a bridge
    write failure must never break the scan or block the evening email.
    Returns the count of rows successfully published.
    """
    import datetime as _dt

    if not results:
        return 0

    scan_date = scan_date or _dt.date.today().isoformat()
    now = _dt.datetime.now().isoformat()
    client = get_client()
    published = 0
    try:
        for r in results:
            try:
                pivot = r.breakout_level if r.breakout_level > 0 else r.entry
                client.execute(
                    """
                    INSERT INTO scanner_signals (
                        ticker, scan_date, pattern, total_score, confidence_pct,
                        entry_price, pivot, stop_loss, target1, target2, rrr,
                        rs_percentile, regime, breadth_score, tier, published_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(ticker, scan_date) DO UPDATE SET
                        pattern=excluded.pattern, total_score=excluded.total_score,
                        confidence_pct=excluded.confidence_pct, entry_price=excluded.entry_price,
                        pivot=excluded.pivot, stop_loss=excluded.stop_loss,
                        target1=excluded.target1, target2=excluded.target2, rrr=excluded.rrr,
                        rs_percentile=excluded.rs_percentile, regime=excluded.regime,
                        breadth_score=excluded.breadth_score, tier=excluded.tier,
                        published_at=excluded.published_at
                    """,
                    [
                        r.ticker, scan_date, r.pattern, r.total_score, r.confidence_pct,
                        r.entry, pivot, r.stop_loss, r.target1, r.target2, r.rrr,
                        r.rs_percentile, regime, r.breadth_score, r.tier, now,
                    ],
                )
                published += 1
            except Exception as e:
                print(f"  [turso_sync] Failed to publish {getattr(r, 'ticker', '?')}: {e}")
    finally:
        client.close()
    return published


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="Verify credentials/connection only, no writes")
    ap.add_argument("--init-schema", action="store_true", help="Create the P1-01 bridge tables (idempotent)")
    args = ap.parse_args()

    if args.test:
        test_connection()
    elif args.init_schema:
        init_schema()
    else:
        print("Nothing to do. Use --test or --init-schema.")
