"""
NSE Momentum — Defensive / Relative-Strength Agent
====================================================
Answers the question the rest of the pipeline deliberately does NOT answer
in a falling market: orchestrator.py forces t1_cap=0 in Regime E (always)
and Regime D (when confidence=HIGH) — "Stay in cash. All setups suppressed."
That is a correct, deliberate risk-off design. This module does not change
that behaviour or generate new bullish entry signals during D/E.

Instead, when regime is D or E, this surfaces a SEPARATE, clearly labeled
"Defensive Watchlist": stocks whose relative strength (already computed
once per scan by rs_agent.py — this module never recomputes RS) shows
genuine outperformance against the falling/weak index, cross-checked
against their own drawdown vs Nifty's drawdown over the same window.

This is capital-preservation/triage information for a cash-and-delivery
trader — "these are relatively less damaged, worth a closer look or worth
holding if already owned" — NOT a new-entry buy signal the way a Tier 1
pattern pick is. Output is kept structurally and visually separate from
tier1/tier2/tier3 for that reason.

Integration
-----------
Called once per scan, AFTER AgentOrchestrator.__init__() has already run
(so self.universe_rs_ranks / self.sector_rs_ranks / self.regime exist),
and using the SAME universe_items / stock_data / nifty50_data / delivery_data
already loaded for the main scan — nothing is re-fetched.

    from agents.defensive_agent import run_defensive_scan

    orch = AgentOrchestrator(data_dict)
    ... existing per-stock loop using orch.run(...) ...

    defensive_picks = run_defensive_scan(
        orch, universe_items, data_dict.get("stock_data", {}),
        data_dict.get("nifty50_data", pd.DataFrame()),
        data_dict.get("delivery_data", {}),
    )
    # defensive_picks is [] unless orch.regime in ("D", "E") — safe to always call.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import pandas as pd

from agents.liquidity_agent import LiquidityAgent

log = logging.getLogger(__name__)

RS_UNIVERSE_PCT_MIN = 60.0   # must beat at least 60% of the universe on RS
RS_SECTOR_PCT_MIN = 50.0     # and be at least mid-pack within its own sector
DD_WINDOW = 60                # bars, matches rs_agent's 12w leg approximately

# Same soft prior as documented in rs_agent.py / pattern_agent.py style —
# small, labeled, and meant to be recalibrated once enough D/E-regime
# pattern_occurrences-equivalent data exists to test it properly.
DEFENSIVE_SECTOR_TILT = {
    "FMCG":         +3,
    "Pharma":       +2,
    "Financials":    0,
    "IT":            0,
    "Energy":        0,
    "Industrials":  -1,
    "ConsumerDisc": -1,
    "Auto":         -2,
    "Metals":       -3,
    "Realty":       -3,
    "Cement":       -1,
    "Chemicals":     0,
    "Telecom":       0,
}


@dataclass
class DefensiveResult:
    ticker: str
    name: str
    sector: str
    tier: str
    price: float
    rs_universe_pct: float
    rs_sector_pct: float
    stock_dd_pct: float
    nifty_dd_pct: float
    adt_cr: float
    defensive_score: float
    note: str = ""


def _max_drawdown_pct(closes: np.ndarray, window: int = DD_WINDOW) -> float:
    """Max drawdown (negative %) over the last `window` bars."""
    if len(closes) < 2:
        return 0.0
    seg = closes[-window:] if len(closes) >= window else closes
    peak = seg[0]
    max_dd = 0.0
    for p in seg:
        if p > peak:
            peak = p
        dd = (p - peak) / peak * 100 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def scan_defensive_candidates(
    orchestrator,
    universe_items: list,
    stock_data: Dict[str, pd.DataFrame],
    nifty50_data: pd.DataFrame,
    delivery_data: dict = None,
) -> List[DefensiveResult]:
    """
    Core scan logic — separated from run_defensive_scan() so it can be
    unit-tested or re-run with --force outside the regime gate.
    """
    if nifty50_data is None or nifty50_data.empty or len(nifty50_data) < DD_WINDOW + 1:
        log.warning("  [Defensive] Insufficient NIFTY history for drawdown comparison — skipping.")
        return []

    nifty_close = nifty50_data["Close"].squeeze().to_numpy(dtype=float)
    nifty_dd = _max_drawdown_pct(nifty_close)

    universe_rs = getattr(orchestrator, "universe_rs_ranks", {}) or {}
    sector_rs = getattr(orchestrator, "sector_rs_ranks", {}) or {}
    delivery_data = delivery_data or {}

    results = []
    seen = set()

    for ticker, name, sector, tier in universe_items:
        if ticker in seen:
            continue
        seen.add(ticker)

        df = stock_data.get(ticker)
        if df is None or df.empty or len(df) < DD_WINDOW + 1:
            continue

        rs_pct = universe_rs.get(ticker)
        if rs_pct is None or rs_pct < RS_UNIVERSE_PCT_MIN:
            continue

        sec_pct = sector_rs.get(ticker, 50.0)
        if sec_pct < RS_SECTOR_PCT_MIN:
            continue

        closes = df["Close"].squeeze().to_numpy(dtype=float)
        stock_dd = _max_drawdown_pct(closes)

        # Must have lost LESS than the index over the same window
        # (both are <= 0; "less negative" = shallower drawdown = better).
        if stock_dd < nifty_dd:
            continue

        liq = LiquidityAgent(df, universe=tier)
        if not liq.passes():
            continue

        score = round(rs_pct - 50, 1) + round((sec_pct - 50) * 0.3, 1)
        score += DEFENSIVE_SECTOR_TILT.get(sector, 0)

        note = ""
        del_pct = delivery_data.get(ticker.replace(".NS", ""))
        if del_pct is not None and del_pct >= 55:
            note = f"High delivery {del_pct:.0f}% — holders not selling into weakness"

        results.append(DefensiveResult(
            ticker=ticker, name=name, sector=sector, tier=tier,
            price=float(closes[-1]),
            rs_universe_pct=round(rs_pct, 1),
            rs_sector_pct=round(sec_pct, 1),
            stock_dd_pct=round(stock_dd, 1),
            nifty_dd_pct=round(nifty_dd, 1),
            adt_cr=round(liq.get_adt(), 1),
            defensive_score=round(score, 1),
            note=note,
        ))

    results.sort(key=lambda r: r.defensive_score, reverse=True)
    return results


def print_defensive_report(results: List[DefensiveResult], regime: str) -> None:
    print("\n" + "=" * 96)
    print(f"  DEFENSIVE / RELATIVE-STRENGTH WATCHLIST  —  Regime {regime}")
    print("  Capital-preservation triage — NOT a Tier 1/2 buy signal. See module docstring.")
    print("=" * 96)
    if not results:
        print("  No qualifying candidates this scan.")
        print("=" * 96 + "\n")
        return

    print(f"  {'TICKER':<16}{'SECTOR':<14}{'TIER':<7}{'RS(U)':>7}{'RS(Sec)':>9}"
          f"{'StkDD':>8}{'NfyDD':>8}{'ADT':>7}{'Score':>7}")
    print("  " + "-" * 90)
    for r in results[:25]:
        print(f"  {r.ticker:<16}{r.sector:<14}{r.tier:<7}"
              f"{r.rs_universe_pct:>6.1f}{r.rs_sector_pct:>8.1f}"
              f"{r.stock_dd_pct:>7.1f}%{r.nifty_dd_pct:>7.1f}%"
              f"{r.adt_cr:>7.1f}{r.defensive_score:>7.1f}")
        if r.note:
            print(f"      └─ {r.note}")
    print("=" * 96 + "\n")


def run_defensive_scan(
    orchestrator,
    universe_items: list,
    stock_data: Dict[str, pd.DataFrame],
    nifty50_data: pd.DataFrame,
    delivery_data: dict = None,
    force: bool = False,
) -> List[dict]:
    """
    Main entry point. Safe to call unconditionally every scan — it is a
    no-op unless orchestrator.regime is D or E, matching the same gate
    the rest of the pipeline already uses for suppressing new long entries.

    Returns a list of plain dicts (not dataclasses) ready to json.dump
    alongside tier1/tier2/tier3/near_breakout in picks_latest.json, each
    tagged "list_type": "DEFENSIVE" so downstream templates/emails can
    render it in its own clearly separate section.
    """
    regime = getattr(orchestrator, "regime", "UNKNOWN")
    t1_cap = getattr(orchestrator, "t1_cap", None)
    macro_state = getattr(orchestrator, "macro_state", "MIXED")

    # Trigger on EITHER a D/E regime OR t1_cap==0 for any other reason
    # (e.g. Regime C but MacroAgent independently forced t1_cap=0 on
    # HOSTILE VIX/breadth/FII — no new buys are happening regardless of
    # which layer caused it, so the defensive view should still run).
    is_regime_bad = regime in ("D", "E")
    is_capped_out = t1_cap == 0

    if not force and not is_regime_bad and not is_capped_out:
        log.info(f"  [Defensive] Regime={regime}, t1_cap={t1_cap} — defensive scan "
                 f"not triggered (runs when regime is D/E OR t1_cap=0). Skipping.")
        return []

    if not is_regime_bad and is_capped_out:
        log.info(f"  [Defensive] Regime={regime} but t1_cap=0 (macro={macro_state}) "
                 f"— running defensive scan on the macro-suppression trigger.")

    log.info(f"  [Defensive] Regime={regime} — running defensive/RS scan...")
    results = scan_defensive_candidates(
        orchestrator, universe_items, stock_data, nifty50_data, delivery_data
    )
    print_defensive_report(results, regime)

    return [
        {
            "list_type":        "DEFENSIVE",
            "ticker":            r.ticker.replace(".NS", ""),
            "ticker_raw":        r.ticker,
            "name":              r.name,
            "sector":            r.sector,
            "tier":              r.tier,
            "price":             round(r.price, 2),
            "rs_universe_pct":   r.rs_universe_pct,
            "rs_sector_pct":     r.rs_sector_pct,
            "stock_dd_pct":      r.stock_dd_pct,
            "nifty_dd_pct":      r.nifty_dd_pct,
            "adt_cr":            r.adt_cr,
            "defensive_score":   r.defensive_score,
            "note":              r.note,
            "regime":            regime,
        }
        for r in results
    ]
