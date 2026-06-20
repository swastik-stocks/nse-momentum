"""
NSE Momentum v4.0 — Portfolio Construction Engine
Position sizing with sector, universe, and correlation controls.
Reads open positions from SQLite portfolio table.

Usage:
    python portfolio_engine.py                     # show current portfolio
    python portfolio_engine.py size RELIANCE 82    # compute position size
    python portfolio_engine.py add RELIANCE.NS     # add to portfolio (interactive)
    python portfolio_engine.py review              # full portfolio review
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ── Portfolio Rules ──────────────────────────────────────────────────────────
MAX_POSITIONS           = 10       # max concurrent open trades
MAX_SECTOR_EXPOSURE_PCT = 30.0     # max % of portfolio in any one sector
MAX_UNIVERSE_EXPOSURE   = {        # max positions per universe
    "LARGE": 4,
    "MID":   4,
    "SMALL": 3,
}
BASE_RISK_PER_TRADE_PCT = 1.0      # % of capital at risk per trade
CAPITAL_INR             = 1_000_000  # default ₹10L capital (overridable)

# Score-based position multipliers (conviction-based sizing)
SCORE_SIZING = {
    (90, 100): 1.5,   # Score 90-100 → 1.5× standard size
    (82, 90):  1.25,  # Score 82-89 → 1.25× standard size
    (78, 82):  1.0,   # Score 78-81 → 1.0× standard size (gate)
    (60, 78):  0.75,  # Score 60-77 (Tier 2) → 0.75× size
    (0,  60):  0.5,   # Below 60 → half size (Tier 3)
}


def _get_multiplier(score: int) -> float:
    for (lo, hi), mult in SCORE_SIZING.items():
        if lo <= score < hi:
            return mult
    return 1.0


def compute_position_size(
    entry: float,
    stop_loss: float,
    score: int = 78,
    capital: float = CAPITAL_INR
) -> dict:
    """
    Compute position size using fixed-fractional risk model.

    Risk per trade = capital × BASE_RISK_PER_TRADE_PCT × score_multiplier
    Position size  = risk_amount / (entry - stop_loss)
    """
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return {"error": "Stop loss must be below entry"}

    mult          = _get_multiplier(score)
    risk_amount   = capital * (BASE_RISK_PER_TRADE_PCT / 100) * mult
    shares        = int(risk_amount / risk_per_share)
    position_value = shares * entry
    position_pct  = (position_value / capital * 100) if capital > 0 else 0

    return {
        "shares":          shares,
        "position_value":  round(position_value, 2),
        "position_pct":    round(position_pct, 2),
        "risk_amount":     round(risk_amount, 2),
        "risk_per_share":  round(risk_per_share, 2),
        "score_mult":      mult,
    }


def get_open_positions() -> list:
    """Return all open positions from portfolio table."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM portfolio WHERE status='OPEN' ORDER BY entry_date DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def check_portfolio_rules(
    ticker: str,
    sector: str,
    universe: str,
    score: int
) -> dict:
    """
    Check if adding this position violates portfolio construction rules.
    Returns {allowed: bool, reason: str, warnings: list}
    """
    positions = get_open_positions()
    warnings  = []
    reason    = ""

    # Rule 1: Max total positions
    if len(positions) >= MAX_POSITIONS:
        return {"allowed": False, "reason": f"Max {MAX_POSITIONS} positions reached", "warnings": []}

    # Rule 2: Universe cap
    universe_count = sum(1 for p in positions if p.get("universe") == universe)
    if universe_count >= MAX_UNIVERSE_EXPOSURE.get(universe, 4):
        return {
            "allowed": False,
            "reason": f"Universe {universe} cap ({MAX_UNIVERSE_EXPOSURE[universe]}) reached",
            "warnings": []
        }

    # Rule 3: Sector concentration
    sector_count = sum(1 for p in positions if p.get("sector") == sector)
    sector_pct   = (sector_count + 1) / MAX_POSITIONS * 100
    if sector_pct > MAX_SECTOR_EXPOSURE_PCT:
        return {
            "allowed": False,
            "reason": f"Sector '{sector}' would exceed {MAX_SECTOR_EXPOSURE_PCT}% exposure",
            "warnings": []
        }

    # Rule 4: Duplicate ticker
    if any(p.get("ticker") == ticker for p in positions):
        return {"allowed": False, "reason": f"{ticker} already in portfolio", "warnings": []}

    # Warnings (non-blocking)
    if sector_count >= 2:
        warnings.append(f"Already have {sector_count} positions in {sector}")
    if len(positions) >= MAX_POSITIONS - 2:
        warnings.append(f"Approaching max positions ({len(positions)}/{MAX_POSITIONS})")

    return {"allowed": True, "reason": "OK", "warnings": warnings}


def add_position(
    ticker: str, name: str, sector: str, universe: str, pattern: str,
    entry: float, stop_loss: float, target1: float, target2: float,
    total_score: int, capital: float = CAPITAL_INR
) -> dict:
    """Add a new position to the portfolio table."""
    check = check_portfolio_rules(ticker, sector, universe, total_score)
    if not check["allowed"]:
        return {"error": check["reason"]}

    sizing = compute_position_size(entry, stop_loss, total_score, capital)
    if "error" in sizing:
        return sizing

    conn = get_connection()
    conn.execute("""
        INSERT INTO portfolio
        (ticker, name, entry_date, entry_price, stop_loss, target1, target2,
         position_size, position_pct, sector, universe, pattern, total_score, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')
    """, (
        ticker, name, datetime.today().strftime("%Y-%m-%d"),
        entry, stop_loss, target1, target2,
        sizing["position_value"], sizing["position_pct"],
        sector, universe, pattern, total_score
    ))
    conn.commit()
    conn.close()
    return {"added": True, "sizing": sizing, "warnings": check["warnings"]}


def close_position(ticker: str) -> dict:
    """Mark position as closed."""
    conn = get_connection()
    conn.execute(
        "UPDATE portfolio SET status='CLOSED' WHERE ticker=? AND status='OPEN'",
        (ticker,)
    )
    n = conn.total_changes
    conn.commit()
    conn.close()
    return {"closed": n > 0, "ticker": ticker}


def print_portfolio() -> None:
    """Print current portfolio with exposure analysis."""
    positions = get_open_positions()
    if not positions:
        print("\n  Portfolio is empty.\n")
        return

    print("\n" + "="*75)
    print(f"  PORTFOLIO — {datetime.today().strftime('%d %b %Y')}")
    print(f"  {len(positions)} open positions")
    print("="*75)
    print(f"  {'TICKER':<14} {'SECTOR':<14} {'UNI':<6} {'ENTRY':>8} {'SL':>8} {'T1':>8} {'SCORE':>5}")
    print("  " + "-"*68)
    for p in positions:
        print(f"  {p['ticker']:<14} {(p['sector'] or '')[:13]:<14} "
              f"{(p['universe'] or ''):>6} {p['entry_price']:>8.1f} "
              f"{p['stop_loss']:>8.1f} {p['target1']:>8.1f} {p.get('total_score',0):>5}")

    # Sector breakdown
    sectors = {}
    for p in positions:
        s = p.get("sector", "Unknown")
        sectors[s] = sectors.get(s, 0) + 1
    print("\n  Sector exposure:")
    for s, n in sorted(sectors.items(), key=lambda x: -x[1]):
        pct = n / MAX_POSITIONS * 100
        bar = "█" * n
        print(f"    {s:<20} {bar} {n} ({pct:.0f}%)")

    print("="*75 + "\n")


def print_sizing(entry: float, stop_loss: float, score: int = 78) -> None:
    """Print position sizing recommendation."""
    s = compute_position_size(entry, stop_loss, score)
    print(f"\n  Position Sizing (Score {score} | Capital ₹{CAPITAL_INR:,})")
    print(f"  {'Entry':<25} ₹{entry:.2f}")
    print(f"  {'Stop Loss':<25} ₹{stop_loss:.2f}")
    print(f"  {'Risk/Share':<25} ₹{s['risk_per_share']:.2f}")
    print(f"  {'Score Multiplier':<25} {s['score_mult']}×")
    print(f"  {'Risk Amount':<25} ₹{s['risk_amount']:,.0f}")
    print(f"  {'Shares':<25} {s['shares']}")
    print(f"  {'Position Value':<25} ₹{s['position_value']:,.0f}")
    print(f"  {'Position %':<25} {s['position_pct']:.1f}%\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "review":
        print_portfolio()
    elif args[0] == "size" and len(args) >= 3:
        entry     = float(args[1])
        stop_loss = float(args[2])
        score     = int(args[3]) if len(args) > 3 else 78
        print_sizing(entry, stop_loss, score)
    elif args[0] == "close" and len(args) >= 2:
        res = close_position(args[1].upper())
        print(f"Closed: {res}")
    else:
        print("Usage:")
        print("  python portfolio_engine.py                     # show portfolio")
        print("  python portfolio_engine.py size 2500 2350 82  # sizing for entry=2500 SL=2350 score=82")
        print("  python portfolio_engine.py close RELIANCE      # close position")
        print("  python portfolio_engine.py review              # full review")
