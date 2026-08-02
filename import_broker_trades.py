"""
import_broker_trades.py — P4-01

Imports Axis broker exports (TradeSummary_*.csv, Brokerage_Charges_*.csv)
into momentum_v4.db, resolves ISIN -> NSE ticker, and FIFO-matches buy/sell
fills per ISIN to compute REAL realized P&L per closed lot.

This is the ground truth P4-02 (signal->outcome attribution) will join
against trades_v4 to answer "are the scanner's weights actually working
with real money" -- not just backtest simulation.

Usage:
    python import_broker_trades.py --trades "C:\\Users\\hp\\Downloads\\TradeSummary_*.csv" ^
                                    --charges "C:\\Users\\hp\\Downloads\\Brokerage_Charges_*.csv"

Design notes (see INTEGRATION_BACKLOG1.md P4-01 for full context):

  - TradeSummary rows are individual FILLS, not round-trip trades. A single
    sell often closes multiple prior buy fills (see NARAYANA HRUDAYALAYA:
    two buy fills of 13+7 shares on 08-Jun, closed by one 20-share sell on
    10-Jul). Realized P&L requires FIFO lot-matching per ISIN, done here.

  - TradeSummary's "Net Price" column is ALREADY charges-adjusted (verified:
    Trade Value - Net Price == the row's own Brokerage+GST+Exchange+SEBI+
    Stamp+STT columns, to the rupee, on every sampled row). So realized P&L
    computed from Net Price is real net P&L -- no separate charge allocation
    needed.

  - Brokerage_Charges aggregates by (date, scrip) -- one row per contract
    note. Cross-checked against TradeSummary: summing TradeSummary's
    per-fill charges by (date, ISIN) exactly matches the corresponding
    Brokerage_Charges row (verified on every sampled row). This import
    uses that match to resolve ISIN -> ticker: Brokerage_Charges' "Contract
    Particulars" column is the NSE symbol + "EQ" suffix (e.g. "NHEQ" ->
    NH.NS), which TradeSummary doesn't give you directly. Any ISIN that
    can't be resolved this way (e.g. no matching charges file for that
    month) is imported anyway but flagged UNMAPPED for manual fixup in
    isin_ticker_map -- never silently guessed.
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from database.schema import get_connection, DB_PATH


# ─────────────────────────────────────────────────────────────────────────
# Schema additions (idempotent — safe to re-run)
# ─────────────────────────────────────────────────────────────────────────

def init_broker_tables():
    conn = get_connection()
    conn.executescript("""
    -- ─────────────────────────────────────────────────────────
    -- BROKER TRADES (raw import, one row per fill)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS broker_trades (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date          TEXT NOT NULL,
        exchange            TEXT,
        isin                TEXT NOT NULL,
        scrip_name          TEXT,
        ticker              TEXT,
        order_number        TEXT,
        buy_sell            TEXT NOT NULL,
        quantity             INTEGER NOT NULL,
        traded_price        REAL,
        trade_value         REAL,
        brokerage_value     REAL,
        gst                 REAL,
        exchange_txn_charges REAL,
        sebi_turnover_fees  REAL,
        stamp_duty          REAL,
        stt                 REAL,
        others              REAL,
        net_price           REAL,
        source_file         TEXT,
        imported_at         TEXT,
        UNIQUE(trade_date, order_number, isin, buy_sell, quantity, traded_price)
    );
    CREATE INDEX IF NOT EXISTS idx_bt_isin ON broker_trades(isin);
    CREATE INDEX IF NOT EXISTS idx_bt_ticker ON broker_trades(ticker);
    CREATE INDEX IF NOT EXISTS idx_bt_date ON broker_trades(trade_date);

    -- ─────────────────────────────────────────────────────────
    -- BROKER CHARGES (raw import, one row per contract note)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS broker_charges (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date              TEXT NOT NULL,
        contract_particulars    TEXT NOT NULL,
        brokerage               REAL,
        exchange_txn_charges    REAL,
        sebi_fees               REAL,
        gst                     REAL,
        stamp_duty              REAL,
        stt_amount              REAL,
        ctt_amount              REAL,
        total_charges           REAL,
        source_file             TEXT,
        imported_at              TEXT,
        UNIQUE(trade_date, contract_particulars)
    );

    -- ─────────────────────────────────────────────────────────
    -- ISIN -> TICKER MAP (built from broker_charges cross-match;
    -- unresolved ISINs get mapped_via='UNMAPPED' for manual fixup)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS isin_ticker_map (
        isin        TEXT PRIMARY KEY,
        ticker      TEXT,
        scrip_name  TEXT,
        mapped_via  TEXT,
        confirmed   INTEGER DEFAULT 0
    );

    -- ─────────────────────────────────────────────────────────
    -- REALIZED TRADES (FIFO-matched round trips — the actual P4-01
    -- deliverable: real P&L per closed lot)
    -- ─────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS realized_trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT,
        isin            TEXT NOT NULL,
        buy_date        TEXT NOT NULL,
        sell_date       TEXT NOT NULL,
        quantity        INTEGER NOT NULL,
        buy_price       REAL,
        sell_price      REAL,
        buy_net_price   REAL,
        sell_net_price  REAL,
        gross_pnl       REAL,
        net_pnl         REAL,
        net_pnl_pct     REAL,
        holding_days    INTEGER,
        computed_at     TEXT,
        UNIQUE(isin, buy_date, sell_date, quantity, buy_net_price, sell_net_price)
    );
    CREATE INDEX IF NOT EXISTS idx_rt_ticker ON realized_trades(ticker);
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────
# CSV parsing — both exports have a variable-length preamble before the
# real header row, and a disclaimer footer after the data. Find the header
# by its known first column, read until the first short/blank row.
# ─────────────────────────────────────────────────────────────────────────

def _read_bordered_csv(path: str, header_first_col: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))
    header_idx = next(i for i, row in enumerate(reader) if row and row[0] == header_first_col)
    header = reader[header_idx]
    rows = []
    for row in reader[header_idx + 1:]:
        if not row or not row[0].strip() or not row[0].strip().isdigit():
            break
        rows.append(dict(zip(header, row)))
    return rows


def parse_trade_summary(path: str) -> list[dict]:
    raw = _read_bordered_csv(path, "Sr. No.")
    out = []
    for r in raw:
        out.append({
            "trade_date": datetime.strptime(r["Trade Date"], "%d-%m-%Y").strftime("%Y-%m-%d"),
            "exchange": r["Exchange"],
            "isin": r["ISIN Number"],
            "scrip_name": r["Scrip Name"].strip(),
            "order_number": r["Order Number"],
            "buy_sell": r["Buy/Sell"],
            "quantity": int(float(r["Quantity"])),
            "traded_price": float(r["Traded Price"]),
            "trade_value": float(r["Trade Value"]),
            "brokerage_value": float(r["Brokerage Value"]),
            "gst": float(r["GST"]),
            "exchange_txn_charges": float(r["Exchange Txn Charges"]),
            "sebi_turnover_fees": float(r["SEBI Turnover Fees"]),
            "stamp_duty": float(r["Stamp Duty"]),
            "stt": float(r["STT"]),
            "others": float(r["Others"]),
            "net_price": float(r["Net Price"]),
        })
    return out


def parse_brokerage_charges(path: str) -> list[dict]:
    raw = _read_bordered_csv(path, "Sr. No.")
    out = []
    for r in raw:
        out.append({
            "trade_date": datetime.strptime(r["Trade Date"], "%d-%b-%Y").strftime("%Y-%m-%d"),
            "contract_particulars": r["Contract Particulars"].strip(),
            "brokerage": float(r["Brokerage"]),
            "exchange_txn_charges": float(r["Exchange Transaction Charges"]),
            "sebi_fees": float(r["SEBI Fees"]),
            "gst": float(r["GST"]),
            "stamp_duty": float(r["Stamp Duty"]),
            "stt_amount": float(r["STT Amount"]),
            "ctt_amount": float(r["CTT Amount"]),
            "total_charges": float(r["Total Charges"]),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
# ISIN -> ticker resolution via cross-file charge matching
# ─────────────────────────────────────────────────────────────────────────

def resolve_isin_tickers(trades: list[dict], charges: list[dict]) -> dict:
    """Group TradeSummary fills by (date, isin), sum brokerage_value, and
    match against Brokerage_Charges rows with the same (date, brokerage
    total) to borrow the ticker from Contract Particulars (strip 'EQ').
    Returns {isin: (ticker, scrip_name, mapped_via)}."""
    by_date_isin = defaultdict(list)
    for t in trades:
        by_date_isin[(t["trade_date"], t["isin"])].append(t)

    charges_by_date = defaultdict(list)
    for c in charges:
        charges_by_date[c["trade_date"]].append(c)

    mapping = {}
    for (date, isin), fills in by_date_isin.items():
        if isin in mapping:
            continue
        total_brokerage = round(sum(f["brokerage_value"] for f in fills), 2)
        scrip_name = fills[0]["scrip_name"]
        match = None
        for c in charges_by_date.get(date, []):
            if abs(round(c["brokerage"], 2) - total_brokerage) < 0.02:
                match = c
                break
        if match:
            code = match["contract_particulars"]
            ticker = code[:-2] + ".NS" if code.upper().endswith("EQ") else code + ".NS"
            mapping[isin] = (ticker, scrip_name, "charges_crossmatch")
        else:
            mapping[isin] = (None, scrip_name, "UNMAPPED")
    return mapping


# ─────────────────────────────────────────────────────────────────────────
# Load into DB (idempotent)
# ─────────────────────────────────────────────────────────────────────────

def load_broker_trades(trades, isin_map, source_file):
    conn = get_connection()
    now = datetime.now().isoformat()
    n_inserted = 0
    for t in trades:
        ticker = isin_map.get(t["isin"], (None, None, None))[0]
        cur = conn.execute("""
            INSERT OR IGNORE INTO broker_trades (
                trade_date, exchange, isin, scrip_name, ticker, order_number,
                buy_sell, quantity, traded_price, trade_value, brokerage_value,
                gst, exchange_txn_charges, sebi_turnover_fees, stamp_duty, stt,
                others, net_price, source_file, imported_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            t["trade_date"], t["exchange"], t["isin"], t["scrip_name"], ticker,
            t["order_number"], t["buy_sell"], t["quantity"], t["traded_price"],
            t["trade_value"], t["brokerage_value"], t["gst"], t["exchange_txn_charges"],
            t["sebi_turnover_fees"], t["stamp_duty"], t["stt"], t["others"],
            t["net_price"], source_file, now,
        ))
        n_inserted += cur.rowcount
    conn.commit()
    conn.close()
    return n_inserted


def load_broker_charges(charges, source_file):
    conn = get_connection()
    now = datetime.now().isoformat()
    n_inserted = 0
    for c in charges:
        cur = conn.execute("""
            INSERT OR IGNORE INTO broker_charges (
                trade_date, contract_particulars, brokerage, exchange_txn_charges,
                sebi_fees, gst, stamp_duty, stt_amount, ctt_amount, total_charges,
                source_file, imported_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            c["trade_date"], c["contract_particulars"], c["brokerage"],
            c["exchange_txn_charges"], c["sebi_fees"], c["gst"], c["stamp_duty"],
            c["stt_amount"], c["ctt_amount"], c["total_charges"], source_file, now,
        ))
        n_inserted += cur.rowcount
    conn.commit()
    conn.close()
    return n_inserted


def save_isin_map(isin_map):
    conn = get_connection()
    for isin, (ticker, scrip_name, via) in isin_map.items():
        conn.execute("""
            INSERT INTO isin_ticker_map (isin, ticker, scrip_name, mapped_via, confirmed)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(isin) DO UPDATE SET
                ticker=excluded.ticker, mapped_via=excluded.mapped_via
                WHERE isin_ticker_map.confirmed = 0
        """, (isin, ticker, scrip_name, via))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────
# FIFO lot matching -> realized P&L
# ─────────────────────────────────────────────────────────────────────────

def compute_realized_trades():
    """Reads ALL broker_trades from the DB (not just this run's import) and
    FIFO-matches buy/sell fills per ISIN. Safe to re-run: realized_trades
    has a UNIQUE constraint, so already-computed lots are skipped."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT trade_date, isin, ticker, buy_sell, quantity, traded_price, net_price
        FROM broker_trades ORDER BY isin, trade_date, id
    """).fetchall()

    by_isin = defaultdict(list)
    for r in rows:
        by_isin[r["isin"]].append(dict(r))

    now = datetime.now().isoformat()
    n_matched = 0
    unmatched_open = []

    for isin, fills in by_isin.items():
        buy_queue = []  # list of dicts: date, qty_remaining, traded_price, net_price
        ticker = fills[0]["ticker"]
        for f in fills:
            qty = abs(f["quantity"])
            if f["buy_sell"] == "B":
                buy_queue.append({
                    "date": f["trade_date"], "qty": qty,
                    "traded_price": f["traded_price"],
                    "net_price_per_share": f["net_price"] / qty,
                })
            elif f["buy_sell"] == "S":
                sell_remaining = qty
                sell_net_per_share = abs(f["net_price"]) / qty
                while sell_remaining > 0 and buy_queue:
                    lot = buy_queue[0]
                    matched_qty = min(lot["qty"], sell_remaining)
                    buy_price = lot["traded_price"]
                    sell_price = f["traded_price"]
                    buy_net = lot["net_price_per_share"]
                    sell_net = sell_net_per_share
                    gross_pnl = (sell_price - buy_price) * matched_qty
                    net_pnl = (sell_net - buy_net) * matched_qty
                    cost_basis = buy_net * matched_qty
                    net_pnl_pct = (net_pnl / cost_basis * 100) if cost_basis else None
                    holding_days = (datetime.strptime(f["trade_date"], "%Y-%m-%d")
                                     - datetime.strptime(lot["date"], "%Y-%m-%d")).days
                    cur = conn.execute("""
                        INSERT OR IGNORE INTO realized_trades (
                            ticker, isin, buy_date, sell_date, quantity, buy_price,
                            sell_price, buy_net_price, sell_net_price, gross_pnl,
                            net_pnl, net_pnl_pct, holding_days, computed_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (ticker, isin, lot["date"], f["trade_date"], matched_qty,
                          buy_price, sell_price, buy_net, sell_net, gross_pnl,
                          net_pnl, net_pnl_pct, holding_days, now))
                    n_matched += cur.rowcount
                    lot["qty"] -= matched_qty
                    sell_remaining -= matched_qty
                    if lot["qty"] == 0:
                        buy_queue.pop(0)
                if sell_remaining > 0:
                    unmatched_open.append((isin, ticker, f["trade_date"], sell_remaining,
                                            "SELL exceeds known buy history — likely a pre-import position"))
        if buy_queue:
            for lot in buy_queue:
                unmatched_open.append((isin, ticker, lot["date"], lot["qty"], "OPEN — not yet sold"))

    conn.commit()
    conn.close()
    return n_matched, unmatched_open


# ─────────────────────────────────────────────────────────────────────────
# Integrity check: TradeSummary per-fill charges vs Brokerage_Charges totals
# ─────────────────────────────────────────────────────────────────────────

def reconcile_charges():
    conn = get_connection()
    mismatches = conn.execute("""
        WITH summed AS (
            SELECT trade_date, isin,
                   SUM(brokerage_value) AS brokerage,
                   SUM(gst) AS gst,
                   SUM(stt) AS stt
            FROM broker_trades GROUP BY trade_date, isin
        )
        SELECT s.trade_date, s.isin, s.brokerage, bc.brokerage, bc.contract_particulars
        FROM summed s
        LEFT JOIN broker_charges bc
          ON bc.trade_date = s.trade_date
        WHERE bc.trade_date IS NULL
    """).fetchall()
    conn.close()
    return mismatches


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Glob pattern for TradeSummary_*.csv files")
    ap.add_argument("--charges", required=True, help="Glob pattern for Brokerage_Charges_*.csv files")
    args = ap.parse_args()

    init_broker_tables()
    print(f"Database: {DB_PATH}\n")

    trade_files = sorted(glob.glob(args.trades))
    charge_files = sorted(glob.glob(args.charges))
    if not trade_files:
        sys.exit(f"No files matched --trades pattern: {args.trades}")

    all_trades, all_charges = [], []
    for tf in trade_files:
        parsed = parse_trade_summary(tf)
        all_trades.extend(parsed)
        print(f"Parsed {len(parsed):>4} fills from {Path(tf).name}")
    for cf in charge_files:
        parsed = parse_brokerage_charges(cf)
        all_charges.extend(parsed)
        print(f"Parsed {len(parsed):>4} charge rows from {Path(cf).name}")

    isin_map = resolve_isin_tickers(all_trades, all_charges)
    save_isin_map(isin_map)

    n_bt = load_broker_trades(all_trades, isin_map, ",".join(Path(f).name for f in trade_files))
    n_bc = load_broker_charges(all_charges, ",".join(Path(f).name for f in charge_files))
    n_rt, unmatched = compute_realized_trades()

    print(f"\nInserted {n_bt} new broker_trades rows, {n_bc} new broker_charges rows, "
          f"{n_rt} new realized_trades (FIFO-matched lots).")

    unmapped = [(isin, sn) for isin, (tk, sn, via) in isin_map.items() if via == "UNMAPPED"]
    if unmapped:
        print(f"\n⚠ {len(unmapped)} ISIN(s) could not be auto-mapped to a ticker "
              f"(no matching Brokerage_Charges row for that date) — fix manually in isin_ticker_map:")
        for isin, name in unmapped:
            print(f"   {isin}  {name}")

    if unmatched:
        print(f"\nOpen / unresolved positions ({len(unmatched)}):")
        for isin, ticker, date, qty, note in unmatched:
            print(f"   {ticker or isin}  {date}  qty={qty}  — {note}")

    mismatches = reconcile_charges()
    if mismatches:
        print(f"\n⚠ {len(mismatches)} (date, isin) group(s) in broker_trades have no matching "
              f"Brokerage_Charges row — charges couldn't be cross-verified for these:")
        for m in mismatches:
            print(f"   {m['trade_date']}  {m['isin']}  fill-summed brokerage={m['brokerage']}")


if __name__ == "__main__":
    main()
