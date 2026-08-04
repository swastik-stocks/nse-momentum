"""
sector_concentration_alert.py — P4-04

Email-alert version of Portfolio Dashboard's Sector Concentration section
(P3-08, app.py). Same question -- "are you unknowingly making one big
sector bet?" -- but computed here so orchestrator.py can feed the result
into emailer.py's tiers dict as Section 6's fourth subsection, instead of
requiring you to open the dashboard to see it.

Deliberately mirrors app.py's P3-08 block line-for-line in logic (same
thresholds, same 🔴/🟡 flag rule, same unmapped-ticker handling) so the
dashboard and the email can never quietly disagree about what counts as
"concentrated." The two threshold constants below are duplicated from
app.py rather than imported -- nse_momentum and PortfolioDashboard are
separate repos/deployments with no shared import path today, and every
other cross-cutting value in emailer.py (EXIT/TRIM/ADDON colors, etc.) is
already self-contained per-file rather than shared. If you ever change
CONCENTRATION_THRESHOLD_PCT or WEAK_SECTOR_SMA50_PCT here, change the
matching constant in PortfolioDashboard/app.py too -- they're meant to
stay identical, just not mechanically linked.

Reads Turso directly (holdings, ticker_sector_map, sector_breadth) via
turso_sync.py, the same bridge pattern sector_breadth.py and
industry_breadth.py already use -- no dependency on db.py or app.py,
which live in the other repo.

Usage (standalone, for testing):
    python sector_concentration_alert.py --dry-run
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Kept identical to app.py's CONCENTRATION_THRESHOLD_PCT / WEAK_SECTOR_SMA50_PCT
# (PortfolioDashboard repo) -- see module docstring above.
CONCENTRATION_THRESHOLD_PCT = 25.0
WEAK_SECTOR_SMA50_PCT = 50.0


def _load_from_turso() -> tuple:
    """
    Returns (holdings: list[dict], ticker_sector: dict, sector_breadth_rows: list[dict]).
    Each independently returns empty on failure -- same "a bridge read
    must never break the pipeline" principle as db.py's get_* functions
    and sector_breadth.py -- so a partial Turso outage degrades this
    feature to "no alert" rather than crashing the evening scan.

    holdings dicts here are LOTS (one row per purchase/account), matching
    Turso's holdings table schema (db.py's add_holding/get_all_lots) --
    NOT the consolidated-by-symbol shape app.py builds locally via
    get_consolidated(). Consolidation (summing qty, weighting avg_price
    across accounts) happens here in _consolidate_lots() instead, since
    that logic lives in db.py (PortfolioDashboard repo) and isn't
    reachable from here.
    """
    from turso_sync import get_client

    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Sector concentration check skipped — cannot reach Turso: {e}")
        return [], {}, []

    holdings, ticker_sector, sector_breadth_rows = [], {}, []
    try:
        try:
            result = client.execute("SELECT symbol, qty, avg_price FROM holdings")
            rows = result.rows if hasattr(result, "rows") else result
            holdings = [{"symbol": r[0], "qty": r[1], "avg_price": r[2]} for r in rows]
        except Exception as e:
            log.warning(f"  Could not read holdings: {e}")

        try:
            result = client.execute("SELECT ticker, sector FROM ticker_sector_map")
            rows = result.rows if hasattr(result, "rows") else result
            ticker_sector = {r[0]: r[1] for r in rows}
        except Exception as e:
            log.warning(f"  Could not read ticker_sector_map: {e}")

        try:
            result = client.execute("""
                SELECT sector, pct_above_sma50 FROM sector_breadth
                WHERE breadth_date = (SELECT MAX(breadth_date) FROM sector_breadth)
            """)
            rows = result.rows if hasattr(result, "rows") else result
            sector_breadth_rows = [{"sector": r[0], "pct_above_sma50": r[1]} for r in rows]
        except Exception as e:
            log.warning(f"  Could not read sector_breadth: {e}")
    finally:
        client.close()

    return holdings, ticker_sector, sector_breadth_rows


def _consolidate_lots(lots: list) -> list:
    """
    Sums qty and total invested value per symbol across accounts --
    mirrors db.py's get_consolidated() (PortfolioDashboard repo), just
    re-implemented here since that function isn't importable across the
    repo boundary. Only the fields this module needs (symbol, qty,
    avg_price, total_invested) -- not the full consolidated shape
    app.py's dashboard view uses (accounts list, num_accounts, etc.),
    which isn't needed for a concentration percentage.
    """
    by_symbol = {}
    for lot in lots:
        sym = lot["symbol"]
        entry = by_symbol.setdefault(sym, {"symbol": sym, "qty": 0.0, "total_invested": 0.0})
        entry["qty"] += lot["qty"]
        entry["total_invested"] += lot["qty"] * lot["avg_price"]
    return list(by_symbol.values())


def compute_sector_concentration() -> list:
    """
    Returns a list of dicts, one per sector currently held, sorted by
    invested value descending -- same shape/order as app.py's P3-08
    concentration_rows, minus the pre-formatted display strings (this is
    for the email pipeline, not Streamlit rendering):
    {sector, invested_value, pct_of_portfolio, sector_sma50, flag}
    where flag is one of "" / "concentrated" / "concentrated_weak".

    Returns [] if holdings or the ticker->sector map aren't available --
    the caller (orchestrator.py, feeding emailer.py's tiers dict) should
    treat that the same as "nothing to alert on," not an error.
    """
    lots, ticker_sector, sector_breadth_rows = _load_from_turso()
    if not lots or not ticker_sector:
        return []

    consolidated = _consolidate_lots(lots)
    sector_breadth_lookup = {r["sector"]: r for r in sector_breadth_rows}

    sector_value = {}
    total_value = 0.0
    unmapped_tickers = []
    for h in consolidated:
        sector = ticker_sector.get(h["symbol"])
        value = h["total_invested"]
        total_value += value
        if sector:
            sector_value[sector] = sector_value.get(sector, 0.0) + value
        else:
            unmapped_tickers.append(h["symbol"])

    if total_value <= 0:
        return []

    if unmapped_tickers:
        log.info(f"  {len(unmapped_tickers)} held ticker(s) with no sector mapping, "
                 f"excluded from concentration %: {', '.join(unmapped_tickers)}")

    results = []
    for sector, value in sorted(sector_value.items(), key=lambda x: -x[1]):
        pct_of_portfolio = value / total_value * 100
        breadth = sector_breadth_lookup.get(sector)
        sma50 = breadth.get("pct_above_sma50") if breadth else None
        is_concentrated = pct_of_portfolio >= CONCENTRATION_THRESHOLD_PCT
        is_weak = sma50 is not None and sma50 < WEAK_SECTOR_SMA50_PCT

        flag = ""
        if is_concentrated and is_weak:
            flag = "concentrated_weak"
        elif is_concentrated:
            flag = "concentrated"

        results.append({
            "sector": sector,
            "invested_value": value,
            "pct_of_portfolio": pct_of_portfolio,
            "sector_sma50": sma50,
            "flag": flag,
        })

    return results


def print_concentration_table(results: list):
    print(f"\n{'SECTOR':<38}{'INVESTED':>14}{'% PORT':>10}{'SMA50':>10}  FLAG")
    print("-" * 90)
    for r in results:
        sma50_str = f"{r['sector_sma50']:.1f}%" if r["sector_sma50"] is not None else "—"
        flag_str = {"concentrated_weak": "🔴 concentrated + weak",
                    "concentrated": "🟡 concentrated",
                    "": ""}[r["flag"]]
        print(f"{r['sector']:<38}₹{r['invested_value']:>12,.0f}{r['pct_of_portfolio']:>9.1f}%"
              f"{sma50_str:>10}  {flag_str}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print only")
    args = ap.parse_args()

    results = compute_sector_concentration()
    if not results:
        print("No concentration data — holdings, ticker_sector_map, or sector_breadth "
              "unavailable, or portfolio has zero invested value.")
    else:
        print_concentration_table(results)
        flagged = [r for r in results if r["flag"]]
        print(f"\n{len(flagged)} sector(s) flagged out of {len(results)}.")
        if args.dry_run:
            print("[DRY RUN] Nothing published — this module only computes, "
                  "orchestrator.py wires the result into emailer.py's tiers dict.")
