"""
db.py — Portfolio Dashboard storage, local OR hosted (Turso).

CORRECTED VERSION: the previous version used the `libsql_client` package
(websocket/"hrana" protocol), which Turso deprecated after migrating
their free-tier infra to AWS — that's what caused the
WSServerHandshakeError. This version uses Turso's current official
`libsql` package instead, which uses the "Embedded Replicas" model:
- A local SQLite file (portfolio_replica.db) is kept in sync with
  your remote Turso database.
- Writes go to the local file AND are pushed to the cloud primary.
- conn.sync() pulls any changes made elsewhere (e.g. via the Turso
  dashboard's Edit Data tab) into the local replica.

Same design principle as before: ONE codebase, driven by env vars.
  TURSO_DATABASE_URL + TURSO_AUTH_TOKEN set -> embedded replica + Turso sync
  neither set -> plain local file, no sync

Each row is one LOT (one purchase, in one account) — NOT pre-consolidated.
Consolidation across accounts happens at query time in get_consolidated().

P1-05: get_scanner_signals() reads scanner_signals, published by
NSE Momentum's daily_scan.yml via turso_sync.py (P1-03). Lives in the
SAME Turso database as holdings — no new connection logic needed, just
another query against the existing get_conn().
"""

import os
import libsql
from pathlib import Path
from datetime import date

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

TURSO_URL   = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

LOCAL_DB_PATH      = Path(__file__).parent / "portfolio.db"
LOCAL_REPLICA_PATH = Path(__file__).parent / "portfolio_replica.db"


def get_conn():
    if TURSO_URL:
        conn = libsql.connect(str(LOCAL_REPLICA_PATH), sync_url=TURSO_URL, auth_token=TURSO_TOKEN)
        conn.sync()  # pull latest remote changes before we read/write
        return conn
    return libsql.connect(str(LOCAL_DB_PATH))


def _sync_if_remote(conn):
    if TURSO_URL:
        conn.sync()  # push our writes to the cloud primary immediately


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            company_name  TEXT,
            account       TEXT NOT NULL,
            qty           REAL NOT NULL,
            avg_price     REAL NOT NULL,
            added_date    TEXT,
            notes         TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_cache (
            symbol             TEXT PRIMARY KEY,
            verdict            TEXT,
            fundamental_score  REAL,
            news_sentiment     INTEGER,
            data_completeness  TEXT,
            computed_at        TEXT
        )
    """)
    conn.commit()
    _sync_if_remote(conn)
    conn.close()


def get_signal_cache() -> dict:
    """{symbol: {verdict, fundamental_score, news_sentiment, data_completeness, computed_at}}"""
    conn = get_conn()
    cursor = conn.execute("SELECT * FROM signal_cache")
    rows = _rows_to_dicts(cursor)
    conn.close()
    return {r["symbol"]: r for r in rows}


def save_signal_cache(symbol: str, verdict: str, fundamental_score: float,
                       news_sentiment: int, data_completeness: str, computed_at: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO signal_cache (symbol, verdict, fundamental_score, news_sentiment, data_completeness, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            verdict=excluded.verdict,
            fundamental_score=excluded.fundamental_score,
            news_sentiment=excluded.news_sentiment,
            data_completeness=excluded.data_completeness,
            computed_at=excluded.computed_at
    """, (symbol, verdict, fundamental_score, news_sentiment, data_completeness, computed_at))
    conn.commit()
    _sync_if_remote(conn)
    conn.close()


def add_holding(symbol: str, company_name: str, account: str, qty: float,
                 avg_price: float, notes: str = "") -> int:
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO holdings (symbol, company_name, account, qty, avg_price, added_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (symbol.strip().upper(), company_name.strip(), account.strip(),
          qty, avg_price, date.today().isoformat(), notes))
    conn.commit()
    new_id = cursor.lastrowid
    _sync_if_remote(conn)
    conn.close()
    return new_id


def update_holding(holding_id: int, symbol: str, company_name: str, account: str,
                    qty: float, avg_price: float, notes: str = ""):
    conn = get_conn()
    conn.execute("""
        UPDATE holdings
        SET symbol=?, company_name=?, account=?, qty=?, avg_price=?, notes=?
        WHERE id=?
    """, (symbol.strip().upper(), company_name.strip(), account.strip(),
          qty, avg_price, notes, holding_id))
    conn.commit()
    _sync_if_remote(conn)
    conn.close()


def delete_holding(holding_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM holdings WHERE id=?", (holding_id,))
    conn.commit()
    _sync_if_remote(conn)
    conn.close()


def _rows_to_dicts(cursor) -> list:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_all_lots() -> list:
    conn = get_conn()
    cursor = conn.execute("SELECT * FROM holdings ORDER BY symbol, account")
    result = _rows_to_dicts(cursor)
    conn.close()
    return result


def get_accounts() -> list:
    conn = get_conn()
    cursor = conn.execute("SELECT DISTINCT account FROM holdings ORDER BY account")
    result = [row[0] for row in cursor.fetchall()]
    conn.close()
    return result


def get_consolidated() -> list:
    """
    Groups all lots by symbol across ALL accounts: sums quantity, computes
    a quantity-weighted average purchase price.
    """
    lots = get_all_lots()
    by_symbol = {}
    for lot in lots:
        sym = lot["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {
                "symbol": sym,
                "company_name": lot["company_name"],
                "qty": 0.0,
                "total_invested": 0.0,
                "accounts": [],
            }
        entry = by_symbol[sym]
        entry["qty"] += lot["qty"]
        entry["total_invested"] += lot["qty"] * lot["avg_price"]
        entry["accounts"].append(lot["account"])

    result = []
    for sym, entry in by_symbol.items():
        weighted_avg = entry["total_invested"] / entry["qty"] if entry["qty"] else 0
        result.append({
            "symbol": sym,
            "company_name": entry["company_name"],
            "qty": entry["qty"],
            "avg_price": weighted_avg,
            "total_invested": entry["total_invested"],
            "accounts": sorted(set(entry["accounts"])),
            "num_accounts": len(set(entry["accounts"])),
        })
    return sorted(result, key=lambda x: x["symbol"])


def get_scanner_signals(scan_date: str = None) -> list:
    """
    P1-05: read scanner signals published by NSE Momentum's evening scan.
    Defaults to the most recent scan_date present in the table (there may
    be a gap of days if the scan hasn't run — e.g. weekends — so "most
    recent" is deliberately not "today"). Returns [] gracefully if the
    table doesn't exist yet or the read fails for any reason — a bridge
    read failure must never break the dashboard (same principle as
    P1-06 on the NSE Momentum side).
    """
    try:
        conn = get_conn()
        if scan_date:
            cursor = conn.execute(
                "SELECT * FROM scanner_signals WHERE scan_date = ? ORDER BY total_score DESC",
                (scan_date,)
            )
        else:
            cursor = conn.execute("""
                SELECT * FROM scanner_signals
                WHERE scan_date = (SELECT MAX(scan_date) FROM scanner_signals)
                ORDER BY total_score DESC
            """)
        result = _rows_to_dicts(cursor)
        conn.close()
        return result
    except Exception:
        return []


if __name__ == "__main__":
    init_db()
    mode = "Turso (embedded replica)" if TURSO_URL else f"local file ({LOCAL_DB_PATH})"
    print(f"Initialized DB in {mode} mode")
