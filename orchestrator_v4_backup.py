"""
NSE Momentum v4.3 — Orchestrator
Wires all 12 agents. 3-tier output system.
Score computed directly (NOT via ca.total()) to avoid double-penalisation.
Tier 1: gate cleared (78/80/82) — Top Picks
Tier 2: 60 to gate-1 — Aggressive / Aggressive
Tier 3: 48-59 — Watchlist / Setup Forming
"""

import sys, logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.pattern_agent              import PatternAgent
from agents.rs_agent                   import RSAgent, compute_universe_ranks
from agents.volume_agent               import VolumeAgent
from agents.market_agent               import MarketAgent
from agents.market_breadth_agent       import MarketBreadthAgent
from agents.sector_agent               import SectorAgent
from agents.risk_agent                 import RiskAgent
from agents.liquidity_agent            import LiquidityAgent
from agents.conviction_agent           import ConvictionAgent
from agents.fundamental_proxy_agent    import FundamentalProxyAgent
from agents.institutional_proxy_agent  import InstitutionalProxyAgent
from trade_logger                      import get_dynamic_weight
from nse_universe                      import UNIVERSE_CONFIG, UNIVERSE_SEED

log = logging.getLogger(__name__)


@dataclass
class StockResult:
    ticker: str;    name: str;    sector: str;    price: float
    universe: str = "LARGE"
    tier: int = 0              # 1=Top Pick, 2=Aggressive, 3=Watchlist, 0=rejected
    pattern: str = ""
    breakout_level: float = 0.0
    entry_low: float = 0.0;       entry_high: float = 0.0
    entry: float = 0.0;           stop_loss: float = 0.0
    target1: float = 0.0;         target2: float = 0.0
    rrr: float = 0.0
    stop_pct: float = 0.0;        gain_pct_t1: float = 0.0
    pattern_score: int = 0;       rs_score: int = 0
    volume_score: int = 0;        market_score: int = 0
    sector_score: int = 0;        rsi_score: int = 0
    ema_score: int = 0;           macd_score: int = 0
    liq_score: int = 0;           bonus_score: int = 0
    fundamental_score: int = 0;   institutional_score: int = 0
    raw_score: int = 0;           total_score: int = 0
    confidence_pct: float = 0.0
    rs_percentile: float = 0.0;   market_regime: str = ""
    regime: str = "";               regime_name: str = ""
    adt_cr: float = 0.0;          mcap_cr: float = 0.0
    mcap_tier: str = "";          part_rate: float = 0.0
    breadth_score: int = 0;       rsi_val: float = 0.0
    rvol: float = 0.0;            del_pct: float = 0.0
    # 3-tier intelligence
    what_is_working: List[str] = field(default_factory=list)
    what_is_missing: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    # Rejection
    rejected: bool = False;       reject_reason: str = ""


class AgentOrchestrator:
    def __init__(self, data_dict: dict):
        self.data = data_dict

        # ── Step 1: Breadth first — feeds regime ─────────────────────
        breadth_agent = MarketBreadthAgent(
            stock_data=data_dict.get("stock_data", {}),
            nifty_df=data_dict.get("nifty50_data", pd.DataFrame()),
        )
        breadth_result       = breadth_agent.compute()
        self.breadth_score   = breadth_result["breadth_score"]
        self.breadth_detail  = breadth_result
        data_dict["breadth_score"] = self.breadth_score

        # ── Step 2: Market regime (breadth-adjusted) ──────────────────
        self.market_agent    = MarketAgent(data_dict)
        self.regime          = self.market_agent.get_regime()
        self.regime_name     = self.market_agent.get_regime_name()
        self.market_score    = self.market_agent.score()
        self.regime_penalty  = self.market_agent.get_penalty()

        # ── Step 3: Sector rotation (once per scan) ───────────────────
        self.sector_agent    = SectorAgent(data_dict)
        self.sector_ranks    = self.sector_agent.get_ranks()

        # ── Step 4: RS ranks (once per scan) ─────────────────────────
        self.universe_rs_ranks = compute_universe_ranks(data_dict)

        log.info(f"Regime: {self.regime} ({self.regime_name}) | "
                 f"Breadth: {self.breadth_score}/10 | Penalty: {self.regime_penalty}")

    def run(self, ticker: str, name: str, sector: str, df: pd.DataFrame,
            delivery_data: dict = None, universe: str = "LARGE") -> StockResult:
        """
        Score a single stock through all 12 agents.
        Returns StockResult with tier assignment.
        """
        r = StockResult(
            ticker=ticker, name=name, sector=sector,
            price=float(df["Close"].iloc[-1]) if not df.empty else 0.0,
            universe=universe,
            market_regime=self.regime,
            breadth_score=self.breadth_score,
        )

        if df.empty or len(df) < 60:
            r.rejected = True; r.reject_reason = "Insufficient price history"
            return r

        cfg         = UNIVERSE_CONFIG[universe]
        del_pct     = (delivery_data or {}).get(ticker.replace(".NS",""), 0.0)
        r.del_pct   = del_pct

        # ── Agent 7: Liquidity gate ───────────────────────────────────
        liq = LiquidityAgent(df, universe=universe)
        if not liq.passes():
            r.rejected = True; r.reject_reason = liq.reject_reason()
            return r
        r.liq_score  = liq.score()
        r.adt_cr     = liq.get_adt()
        r.part_rate  = liq.get_part_rate()
        r.mcap_tier  = liq.get_mcap_tier()

        # ── Agent 1: Pattern detection ────────────────────────────────
        pa = PatternAgent(df)
        if not pa.pattern:
            r.rejected = True; r.reject_reason = "No pattern detected"
            return r
        dyn_weight       = get_dynamic_weight(pa.pattern)
        r.pattern        = pa.pattern
        r.breakout_level = pa.breakout_level
        r.entry_low      = pa.entry_low
        r.entry_high     = pa.entry_high
        r.pattern_score  = min(dyn_weight, 18)
        r.ema_score      = pa.get_ema_score()
        r.macd_score     = pa.get_macd_score()
        r.rsi_score      = pa.get_rsi_score()

        # RSI value for display
        try:
            import ta
            r.rsi_val = float(
                ta.momentum.RSIIndicator(df["Close"].squeeze(), 14).rsi().iloc[-1]
            )
        except Exception:
            pass

        # RVOL
        vol = df["Volume"].squeeze().to_numpy(dtype=float)
        avg20v = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1
        r.rvol = round(float(vol[-1]) / avg20v, 2)

        # ── Agent 2: Relative Strength ────────────────────────────────
        rsa = RSAgent(
            df, self.data.get("nifty50_data", pd.DataFrame()),
            universe_ranks=self.universe_rs_ranks, ticker=ticker
        )
        r.rs_score      = min(int(rsa.score() * cfg["rs_weight_mult"]), 20)
        r.rs_percentile = rsa.get_percentile()

        if r.rs_percentile < 40:
            r.rejected = True; r.reject_reason = f"RS Percentile {r.rs_percentile:.0f} < 40"
            return r

        # ── Agent 3: Volume ───────────────────────────────────────────
        va = VolumeAgent(df, del_pct, universe=universe)
        r.volume_score = min(int(va.score() * cfg["vol_weight_mult"]), 12)

        # ── Agent 4+5: Market regime already computed ─────────────────
        r.market_score  = self.market_score
        r.regime        = self.regime
        r.regime_name   = self.regime_name
        r.market_regime = self.regime

        # ── Agent 6: Sector ───────────────────────────────────────────
        r.sector_score = self.sector_agent.score_for_sector(sector, self.sector_ranks)

        # ── Agent 8: Risk gate ────────────────────────────────────────
        risk = RiskAgent(df, r.breakout_level, r.entry_low, r.entry_high, universe=universe)
        if not risk.passes():
            r.rejected = True; r.reject_reason = risk.reject_reason()
            return r
        r.entry     = risk.entry
        r.stop_loss = risk.stop
        r.target1   = risk.target1
        r.target2   = risk.target2
        r.rrr       = risk.rrr
        r.stop_pct  = risk.stop_pct
        r.gain_pct_t1 = risk.gain_pct

        # ── Agent 9: Fundamental Proxy ────────────────────────────────
        fp = FundamentalProxyAgent(ticker, df, del_pct, r.rs_percentile,
                                    self.sector_ranks.get(sector, 7))
        fp_result          = fp.evaluate()
        r.fundamental_score = fp_result["fundamental_proxy_score"]

        # ── Agent 10: Institutional Proxy ─────────────────────────────
        ip = InstitutionalProxyAgent(ticker, df, del_pct)
        ip_result             = ip.evaluate()
        r.institutional_score = ip_result["institutional_proxy_score"]

        # ── Bonus score ───────────────────────────────────────────────
        r.bonus_score = min(
            r.liq_score // 2
            + r.fundamental_score // 4
            + r.institutional_score // 4, 5
        )

        # ── MASTER SCORE — direct computation (bypasses ca.total()) ──
        # Formula: RS(20) + Pattern(18) + RSI(15) + Volume(12) + EMA(10)
        #        + Market(5) + MACD(8) + Sector(7) + Bonus(5) = 100
        r.raw_score = (
            r.rs_score       +   # 0-20
            r.pattern_score  +   # 0-18
            r.rsi_score      +   # 0-15
            r.volume_score   +   # 0-12
            r.ema_score      +   # 0-10
            r.market_score   +   # 0-5
            r.macd_score     +   # 0-8
            r.sector_score   +   # 0-7
            r.bonus_score        # 0-5
        )                        # Total: 0-100

        # Regime penalty (universe-aware)
        penalty = int(self.regime_penalty * cfg["regime_penalty_mult"])
        r.total_score = max(0, r.raw_score + penalty)

        # ── Agent 11: Conviction calibration ─────────────────────────
        ca = ConvictionAgent()
        r.confidence_pct = ca.calibrate_confidence(r.pattern, r.total_score, universe)

        # ── Tier assignment ───────────────────────────────────────────
        gate = cfg["score_gate"]
        if r.total_score >= gate:
            r.tier = 1
            r.what_is_working = self._why_working(r)
        elif r.total_score >= 55:
            r.tier = 2
            r.what_is_working = self._why_working(r)
            r.what_is_missing = self._what_missing(r, gate)
            r.trigger_conditions = self._triggers(r)
        elif r.total_score >= 42:
            r.tier = 3
            r.what_is_working = self._why_working(r)[:2]
            r.trigger_conditions = self._triggers(r)
        else:
            r.rejected = True; r.reject_reason = f"Score {r.total_score} below watchlist threshold"
            return r

        r.risk_factors = self._risk_factors(r)
        return r

    def run_universe(self, universe_items: list,
                     stock_data: dict, delivery_data: dict) -> dict:
        """
        Score all stocks and return tier dict.
        Returns {'tier1': [...], 'tier2': [...], 'tier3': [...], 'regime': ..., 'breadth': ...}
        """
        all_results = []
        reject_reasons = {}
        for item in universe_items:
            ticker, name, sector, universe = item
            df = stock_data.get(ticker, pd.DataFrame())
            if df.empty:
                continue
            try:
                result = self.run(ticker, name, sector, df, delivery_data, universe)
                if not result.rejected:
                    all_results.append(result)
                else:
                    reason = result.reject_reason
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            except Exception as e:
                log.warning(f"{ticker} CRASH: {type(e).__name__}: {e}")
                reject_reasons[f"Exception: {type(e).__name__}"] = reject_reasons.get(f"Exception: {type(e).__name__}", 0) + 1
        
        # Log rejection breakdown for diagnostics
        if reject_reasons:
            # Bucket similar reasons for cleaner reporting
            buckets = {}
            for reason, count in reject_reasons.items():
                if reason.startswith("R:R"):
                    key = "R:R below minimum"
                elif reason.startswith("ADT"):
                    key = "ADT below minimum"
                elif "pattern" in reason.lower():
                    key = "No pattern detected"
                elif reason.startswith("Score"):
                    key = "Score below T3 threshold (42)"
                elif "Insufficient" in reason:
                    key = "Insufficient price history"
                else:
                    key = reason
                buckets[key] = buckets.get(key, 0) + count
            log.info("  Rejection breakdown:")
            for reason, count in sorted(buckets.items(), key=lambda x: -x[1])[:10]:
                log.info(f"    {count:>3} x {reason}")

        # Sort by total score descending
        all_results.sort(key=lambda x: x.total_score, reverse=True)

        t1 = [r for r in all_results if r.tier == 1]
        t2 = [r for r in all_results if r.tier == 2]
        t3 = [r for r in all_results if r.tier == 3]

        return {
            "tier1": t1, "tier2": t2, "tier3": t3,
            "all_results": all_results,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "breadth": self.breadth_score,
            "breadth_detail": self.breadth_detail,
        }

    # ── Intelligence text generators ─────────────────────────────────

    def _why_working(self, r: StockResult) -> List[str]:
        reasons = []
        if r.rs_percentile >= 80:
            reasons.append(f"RS Rank {r.rs_percentile:.0f}th — outperforming {r.rs_percentile:.0f}% of market")
        if r.pattern:
            reasons.append(f"{r.pattern} pattern detected — entry zone Rs.{r.entry_low:.1f}–{r.entry_high:.1f}")
        if r.rvol >= 1.5:
            reasons.append(f"Volume surge {r.rvol:.1f}x avg — institutional buying signal")
        if r.del_pct >= 50:
            reasons.append(f"Delivery {r.del_pct:.0f}% — large holders not selling")
        if r.sector_score >= 5:
            top = self.sector_agent.get_top_sectors(3)
            sector_names = [s for s, _ in top]
            if r.sector in sector_names:
                reasons.append(f"Sector leadership — {r.sector} is top-3 in rotation")
        if r.rsi_val > 0:
            reasons.append(f"RSI {r.rsi_val:.0f} — momentum in constructive zone")
        return reasons[:4]

    def _what_missing(self, r: StockResult, gate: int) -> List[str]:
        missing = []
        gap = gate - r.total_score
        missing.append(f"Score {r.total_score} — needs {gap} more points to clear Tier 1 gate")
        if r.rs_percentile < 75:
            missing.append(f"RS Rank {r.rs_percentile:.0f}th — top-tier setups need RS >= 75")
        if r.rvol < 1.3:
            missing.append(f"Volume {r.rvol:.1f}x — needs convincing breakout volume >= 1.5x")
        if r.regime in ["C", "D", "E"]:
            missing.append(f"Market regime {r.regime} ({r.regime_name}) — regime penalty applied")
        return missing[:3]

    def _triggers(self, r: StockResult) -> List[str]:
        triggers = []
        if r.breakout_level > 0:
            triggers.append(f"Price closes above Rs.{r.breakout_level:.1f} on 1.5x avg volume")
        triggers.append(f"RS rank improves above 75th percentile")
        if r.regime in ["C", "D"]:
            triggers.append("Market regime shifts to B (Bull) or better")
        triggers.append(f"Stop loss: Rs.{r.stop_loss:.1f} ({r.stop_pct:.1f}% risk)")
        return triggers[:3]

    def _risk_factors(self, r: StockResult) -> List[str]:
        risks = []
        if r.regime in ["D", "E"]:
            risks.append(f"Market in {r.regime_name} — elevated failure rate on breakouts")
        if r.rrr < 2.0:
            risks.append(f"R:R {r.rrr:.1f}x — below 2x ideal; size smaller")
        if r.stop_pct > 8:
            risks.append(f"Wide stop {r.stop_pct:.1f}% — consider smaller position size")
        if r.universe == "SMALL":
            risks.append("Small-cap: higher volatility, wider spreads, lower liquidity")
        if r.breadth_score <= 3:
            risks.append(f"Market breadth weak ({r.breadth_score}/10) — sector-specific risk elevated")
        return risks[:3]
