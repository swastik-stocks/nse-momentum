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


def migrate_position_actions_columns(client):
    """
    P2-03/P2-04: position_actions (created by P1-01) doesn't yet have
    tranche-sizing columns. Rather than define a second, conflicting table
    (a real mistake worth avoiding), extend the existing one -- reusing
    trigger_price/blended_cost/new_stop/reason for concepts that already
    map cleanly, adding only the 3 genuinely new columns. Same
    check-before-ALTER pattern trade_logger.py already uses elsewhere in
    this codebase (SQLite has no "ADD COLUMN IF NOT EXISTS").
    """
    rs = client.execute("PRAGMA table_info(position_actions)")
    existing = {row[1] for row in rs.rows}
    new_columns = {
        "tranche_number":          "INTEGER",
        "recommended_qty":         "INTEGER",
        "recommended_qty_inr":     "REAL",
    }
    for col_name, col_type in new_columns.items():
        if col_name not in existing:
            client.execute(f"ALTER TABLE position_actions ADD COLUMN {col_name} {col_type}")
            print(f"  [turso_sync] Added column position_actions.{col_name}")


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
        migrate_position_actions_columns(client)
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
    try:
        client = get_client()
    except SystemExit as e:
        print(f"  [turso_sync] publish_signals skipped — {e}")
        return 0
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


def get_holdings() -> dict:
    """
    P1-04: read current holdings from Turso, consolidated by symbol across
    accounts — deliberately mirrors Portfolio Dashboard's own
    get_consolidated() in db.py, so the scanner sees the same "what do I
    own" picture the dashboard shows you, not a re-derived one that could
    drift out of sync.

    Returns {ticker: {"qty": float, "avg_price": float, "company_name": str,
    "accounts": [str, ...]}}, keyed by ticker (e.g. "RELIANCE.NS") for O(1)
    lookup during a scan. Empty dict on any failure — never raises, since a
    holdings-read failure must not be able to break the scan (P1-06).
    """
    try:
        client = get_client()
    except SystemExit:
        return {}

    try:
        rs = client.execute("SELECT symbol, company_name, account, qty, avg_price FROM holdings")
        rows = rs.rows
    except Exception as e:
        print(f"  [turso_sync] get_holdings failed: {e}")
        return {}
    finally:
        client.close()

    by_symbol = {}
    for symbol, company_name, account, qty, avg_price in rows:
        if symbol not in by_symbol:
            by_symbol[symbol] = {"company_name": company_name, "qty": 0.0,
                                  "total_invested": 0.0, "accounts": []}
        entry = by_symbol[symbol]
        entry["qty"] += qty
        entry["total_invested"] += qty * avg_price
        entry["accounts"].append(account)

    holdings = {}
    for symbol, entry in by_symbol.items():
        weighted_avg = entry["total_invested"] / entry["qty"] if entry["qty"] else 0
        holdings[symbol] = {
            "qty": entry["qty"],
            "avg_price": weighted_avg,
            "company_name": entry["company_name"],
            "accounts": sorted(set(entry["accounts"])),
        }
    return holdings


def get_tranche_count(ticker: str) -> int:
    """
    P2-03: how many ADD_ON tranches have already been recommended for this
    ticker, so a fresh candidate knows whether it's tranche 1, 2, or 3 (and
    whether MAX_TRANCHES has already been hit). Counts existing rows in
    position_actions with action_type='ADD_ON' for this ticker. Returns 0
    (never raises) on any failure -- a read failure here should block a new
    recommendation from firing, not crash the scan; treating it as "no
    prior tranches" is the conservative choice (worst case: recommends
    tranche 1 again, which the UNIQUE(ticker, action_date, action_type)
    constraint plus daily cadence makes harmless, rather than silently
    skipping a real add-on opportunity).
    """
    try:
        client = get_client()
    except SystemExit:
        return 0
    try:
        rs = client.execute(
            "SELECT COUNT(*) FROM position_actions WHERE ticker = ? AND action_type = 'ADD_ON'",
            [ticker],
        )
        return rs.rows[0][0] if rs.rows else 0
    except Exception as e:
        print(f"  [turso_sync] get_tranche_count failed for {ticker}: {e}")
        return 0
    finally:
        client.close()


def publish_position_action(ticker: str, action_type: str, trigger_price: float,
                             blended_cost: float, new_stop: float, reason: str,
                             tranche_number: int = None, recommended_qty: int = None,
                             recommended_qty_inr: float = None, action_date: str = None):
    """
    P2-03/P2-04: publish a position action (ADD_ON/EXIT/TRIM) to the
    existing position_actions table (schema from P1-01, tranche columns
    added by migrate_position_actions_columns above). Same failure
    isolation as publish_signals/get_holdings -- never raises past the
    caller, returns False on failure so the scan continues regardless.
    """
    import datetime as _dt
    action_date = action_date or _dt.date.today().isoformat()
    now = _dt.datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        print(f"  [turso_sync] publish_position_action skipped — {e}")
        return False
    try:
        client.execute(
            """
            INSERT INTO position_actions (
                ticker, action_date, action_type, reason, trigger_price,
                blended_cost, new_stop, published_at, tranche_number,
                recommended_qty, recommended_qty_inr
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker, action_date, action_type) DO UPDATE SET
                reason=excluded.reason, trigger_price=excluded.trigger_price,
                blended_cost=excluded.blended_cost, new_stop=excluded.new_stop,
                published_at=excluded.published_at, tranche_number=excluded.tranche_number,
                recommended_qty=excluded.recommended_qty,
                recommended_qty_inr=excluded.recommended_qty_inr
            """,
            [ticker, action_date, action_type, reason, trigger_price,
             blended_cost, new_stop, now, tranche_number, recommended_qty,
             recommended_qty_inr],
        )
        return True
    except Exception as e:
        print(f"  [turso_sync] Failed to publish position_action for {ticker}: {e}")
        return False
    finally:
        client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="Verify credentials/connection only, no writes")
    ap.add_argument("--init-schema", action="store_true", help="Create the P1-01 bridge tables (idempotent)")
    ap.add_argument("--holdings", action="store_true", help="Fetch and print current holdings from Turso")
    ap.add_argument("--migrate", action="store_true", help="Run position_actions column migration only (P2-03/04)")
    args = ap.parse_args()

    if args.test:
        test_connection()
    elif args.init_schema:
        init_schema()
    elif args.migrate:
        client = get_client()
        try:
            migrate_position_actions_columns(client)
            print("Migration complete.")
        finally:
            client.close()
    elif args.holdings:
        h = get_holdings()
        print(f"{len(h)} distinct tickers held:")
        for ticker, info in sorted(h.items()):
            print(f"   {ticker:<15} qty={info['qty']:>8.1f}  avg={info['avg_price']:>9.2f}  "
                  f"accounts={','.join(info['accounts'])}")
    else:
        print("Nothing to do. Use --test or --init-schema.")
