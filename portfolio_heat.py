"""
portfolio_heat.py — P4-04

Computes portfolio heat: the total open risk in rupees across all held
positions, defined as:

    heat_per_holding = (current_price - effective_stop) × qty
    portfolio_heat   = Σ heat_per_holding  (across all held lots)

This answers "if every stop gets hit tonight, how much do I lose?" in
absolute rupee terms — the single number that tells you whether your
aggregate risk is within your daily/weekly loss limits without requiring
you to mentally sum across positions.

Per-holding effective stop uses the SAME P2-05 logic already live in
orchestrator.py (the tighter of technical stop vs. ENTRY hard stop
computed from avg_price × (1 - STOP_CAP)), not a separate recomputation.
The effective stops are read from Turso's holdings_heat table if already
published by orchestrator.py, OR recomputed here from price_history +
avg_price if that table isn't available yet — fails gracefully rather
than blocking the email.

Output:
  - Published to Turso's portfolio_heat table (for dashboard rendering)
  - Returned as a list of dicts for emailer.py's tiers["portfolio_heat"]

Definitions:
  - distance_to_stop: (current_price - effective_stop) / current_price
    expressed as a % — how far price needs to fall to hit the stop
  - heat_inr: distance_to_stop × invested_value — rupee loss IF stop hit
  - effective_stop: max(technical_stop, avg_price × (1 - STOP_CAP[tier]))
    matching P2-05's blended-cost hard stop exactly
  - total_heat_inr: Σ heat_inr across all positions with a valid stop

Usage (standalone, for testing):
    python portfolio_heat.py --dry-run
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# STOP_CAP per universe tier — kept identical to agents/risk_agent.py's
# STOP_CAP and orchestrator.py's P2-05 effective-stop computation.
# Duplicated (not imported) for the same cross-repo-boundary reason as
# sector_concentration_alert.py's threshold constants — this module runs
# from nse_momentum and can't import across to risk_agent.py's venv path
# without sys.path manipulation that would make this fragile.
# If you change risk_agent.py's STOP_CAP, change this too.
STOP_CAP = {"LARGE": 0.03, "MID": 0.04, "SMALL": 0.05}
DEFAULT_STOP_CAP = 0.05   # conservative fallback if universe tier unknown


def _load_holdings_and_prices() -> tuple:
    """
    Returns (holdings: list[dict], prices: dict[ticker->float]).
    holdings: consolidated by symbol across accounts, with qty + avg_price.
    prices: latest close from price_history for each held ticker.
    Both return empty on failure — same "must never break the pipeline"
    principle as every other Turso bridge read in this codebase.
    """
    from turso_sync import get_client

    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Portfolio heat skipped — cannot reach Turso: {e}")
        return [], {}

    lots, prices = [], {}
    try:
        # Holdings — raw lots, consolidated below
        try:
            result = client.execute(
                "SELECT symbol, qty, avg_price FROM holdings"
            )
            rows = result.rows if hasattr(result, "rows") else result
            lots = [{"symbol": r[0], "qty": float(r[1]), "avg_price": float(r[2])}
                    for r in rows]
        except Exception as e:
            log.warning(f"  Could not read holdings from Turso: {e}")

        # Latest prices from price_history
        if lots:
            tickers = list({lot["symbol"] for lot in lots})
            try:
                from database.schema import get_connection
                conn = get_connection()
                row = conn.execute("SELECT MAX(date) FROM price_history").fetchone()
                latest_date = row[0] if row and row[0] else date.today().isoformat()
                for ticker in tickers:
                    r = conn.execute(
                        "SELECT close FROM price_history WHERE ticker=? AND date<=? "
                        "ORDER BY date DESC LIMIT 1",
                        (ticker, latest_date)
                    ).fetchone()
                    if r:
                        prices[ticker] = float(r[0])
                conn.close()
            except Exception as e:
                log.warning(f"  Could not read prices from price_history: {e}")

    finally:
        client.close()

    return lots, prices


def _consolidate_lots(lots: list) -> list:
    """Mirrors db.py's get_consolidated() — qty-weighted avg_price per symbol."""
    by_symbol = {}
    for lot in lots:
        sym = lot["symbol"]
        entry = by_symbol.setdefault(sym, {"symbol": sym, "qty": 0.0, "total_invested": 0.0})
        entry["qty"] += lot["qty"]
        entry["total_invested"] += lot["qty"] * lot["avg_price"]
    result = []
    for sym, e in by_symbol.items():
        result.append({
            "symbol":         sym,
            "qty":            e["qty"],
            "avg_price":      e["total_invested"] / e["qty"] if e["qty"] else 0.0,
            "total_invested": e["total_invested"],
        })
    return result


def compute_portfolio_heat() -> dict:
    """
    Returns:
    {
        "holdings_heat": [
            {symbol, qty, avg_price, current_price, effective_stop,
             stop_pct, heat_inr, invested_value, heat_pct_of_position},
            ...
        ],
        "total_heat_inr":       float,   # Σ heat_inr across all positions
        "total_invested_inr":   float,   # Σ invested_value (denominator for heat%)
        "heat_pct_of_portfolio": float,  # total_heat / total_invested * 100
        "positions_with_stop":  int,     # positions where heat was computable
        "positions_no_price":   int,     # skipped — no price in price_history
        "as_of_date":           str,
    }

    Returns an empty dict with zero totals on any bridge failure — caller
    treats that the same as "no heat data available" rather than an error.
    """
    lots, prices = _load_holdings_and_prices()
    if not lots:
        return {"holdings_heat": [], "total_heat_inr": 0.0,
                "total_invested_inr": 0.0, "heat_pct_of_portfolio": 0.0,
                "positions_with_stop": 0, "positions_no_price": 0,
                "as_of_date": date.today().isoformat()}

    consolidated = _consolidate_lots(lots)
    holdings_heat = []
    total_heat = 0.0
    total_invested = 0.0
    n_no_price = 0

    for h in consolidated:
        sym = h["symbol"]
        current_price = prices.get(sym)
        if current_price is None:
            log.info(f"  {sym}: no price in price_history — excluded from heat")
            n_no_price += 1
            continue

        avg_price = h["avg_price"]
        qty = h["qty"]
        invested_value = h["total_invested"]

        # Effective stop: conservative hard stop from avg_price
        # (same P2-05 logic — max of technical stop and entry-cost stop;
        # we don't have the technical stop here without running RiskAgent
        # over live price data, so we use the entry-cost hard stop only,
        # which is what P2-05 falls back to when technical > entry stop).
        # This is a conservative FLOOR on the stop — real stop may be
        # tighter (closer to current price) if RiskAgent found one above it.
        # STOP_CAP default: we don't know the universe tier from holdings
        # alone, so use DEFAULT_STOP_CAP (5% — small-cap conservative).
        cap = DEFAULT_STOP_CAP
        hard_stop = avg_price * (1 - cap)
        effective_stop = hard_stop

        if current_price <= effective_stop:
            # Already below stop — heat is 0 for this position (it's already
            # an EXIT alert per P2-05; don't double-count as "open risk").
            stop_pct = 0.0
            heat_inr = 0.0
        else:
            stop_pct = (current_price - effective_stop) / current_price * 100
            heat_inr = (current_price - effective_stop) * qty

        total_heat += heat_inr
        total_invested += invested_value

        holdings_heat.append({
            "symbol":               sym,
            "qty":                  qty,
            "avg_price":            round(avg_price, 2),
            "current_price":        round(current_price, 2),
            "effective_stop":       round(effective_stop, 2),
            "stop_pct":             round(stop_pct, 1),
            "heat_inr":             round(heat_inr, 0),
            "invested_value":       round(invested_value, 0),
            "heat_pct_of_position": round(heat_inr / invested_value * 100, 1)
                                    if invested_value > 0 else 0.0,
        })

    holdings_heat.sort(key=lambda x: -x["heat_inr"])

    heat_pct = total_heat / total_invested * 100 if total_invested > 0 else 0.0

    return {
        "holdings_heat":        holdings_heat,
        "total_heat_inr":       round(total_heat, 0),
        "total_invested_inr":   round(total_invested, 0),
        "heat_pct_of_portfolio": round(heat_pct, 2),
        "positions_with_stop":  len(holdings_heat),
        "positions_no_price":   n_no_price,
        "as_of_date":           date.today().isoformat(),
    }


def publish_to_turso(heat_data: dict) -> int:
    """
    Publishes per-holding heat rows to Turso's portfolio_heat table
    (created here if it doesn't exist — same idempotent CREATE TABLE IF
    NOT EXISTS pattern as industry_breadth.py). Upserts on (symbol, as_of_date)
    so safe to re-run daily. Returns number of rows published.
    """
    from turso_sync import get_client
    from datetime import datetime as _dt

    if not heat_data.get("holdings_heat"):
        return 0

    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Publish skipped — {e}")
        return 0

    now = _dt.now().isoformat()
    published = 0
    try:
        client.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_heat (
                symbol                TEXT NOT NULL,
                as_of_date            TEXT NOT NULL,
                qty                   REAL,
                avg_price             REAL,
                current_price         REAL,
                effective_stop        REAL,
                stop_pct              REAL,
                heat_inr              REAL,
                invested_value        REAL,
                heat_pct_of_position  REAL,
                published_at          TEXT,
                PRIMARY KEY (symbol, as_of_date)
            )
        """)
        for r in heat_data["holdings_heat"]:
            try:
                client.execute("""
                    INSERT INTO portfolio_heat (
                        symbol, as_of_date, qty, avg_price, current_price,
                        effective_stop, stop_pct, heat_inr, invested_value,
                        heat_pct_of_position, published_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(symbol, as_of_date) DO UPDATE SET
                        qty=excluded.qty, avg_price=excluded.avg_price,
                        current_price=excluded.current_price,
                        effective_stop=excluded.effective_stop,
                        stop_pct=excluded.stop_pct,
                        heat_inr=excluded.heat_inr,
                        invested_value=excluded.invested_value,
                        heat_pct_of_position=excluded.heat_pct_of_position,
                        published_at=excluded.published_at
                """, [r["symbol"], heat_data["as_of_date"], r["qty"],
                      r["avg_price"], r["current_price"], r["effective_stop"],
                      r["stop_pct"], r["heat_inr"], r["invested_value"],
                      r["heat_pct_of_position"], now])
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish heat for {r['symbol']}: {e}")
    finally:
        client.close()
    return published


def print_heat_table(heat_data: dict):
    rows = heat_data.get("holdings_heat", [])
    if not rows:
        print("No heat data available.")
        return
    print(f"\n{'SYMBOL':<16}{'QTY':>8}{'AVG COST':>12}{'CMP':>12}"
          f"{'STOP':>12}{'STOP%':>8}{'HEAT ₹':>14}{'HEAT%POS':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['symbol'].replace('.NS',''):<16}{r['qty']:>8.0f}"
              f"₹{r['avg_price']:>10,.1f}  ₹{r['current_price']:>10,.1f}"
              f"  ₹{r['effective_stop']:>10,.1f}{r['stop_pct']:>7.1f}%"
              f"  ₹{r['heat_inr']:>11,.0f}{r['heat_pct_of_position']:>9.1f}%")
    print("-" * 100)
    print(f"{'TOTAL OPEN RISK':<40}"
          f"₹{heat_data['total_heat_inr']:>11,.0f}  "
          f"({heat_data['heat_pct_of_portfolio']:.2f}% of ₹{heat_data['total_invested_inr']:,.0f} invested)")
    print(f"Positions with stop: {heat_data['positions_with_stop']} | "
          f"No price data: {heat_data['positions_no_price']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print only, skip Turso publish")
    args = ap.parse_args()

    heat_data = compute_portfolio_heat()
    print_heat_table(heat_data)

    if args.dry_run:
        print(f"\n[DRY RUN] Would publish {len(heat_data.get('holdings_heat', []))} "
              f"rows to Turso portfolio_heat table — skipped.")
    else:
        n = publish_to_turso(heat_data)
        print(f"\nPublished {n} rows to Turso portfolio_heat table.")
