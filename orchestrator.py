"""
NSE Momentum v5.0 - Orchestrator (complete)
All 14 agents wired. Full P1+P2 implementation.
"""

import sys, logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date

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
from agents.asymmetry_gate             import AsymmetryGate
from agents.vcp_gate                   import VCPContractionGate
from agents.macro_agent                import MacroAgent
from agents.event_risk_agent           import EventRiskAgent
from agents.confirmation_agent         import ConfirmationAgent
from agents.near_breakout              import find_near_breakout_stocks
from trade_logger                      import get_dynamic_weight
from nse_universe                      import UNIVERSE_CONFIG, UNIVERSE_SEED

log = logging.getLogger(__name__)


@dataclass
class StockResult:
    ticker: str;  name: str;  sector: str;  price: float
    universe: str = "LARGE"
    tier: int = 0
    pattern: str = ""
    breakout_level: float = 0.0
    entry_low: float = 0.0;      entry_high: float = 0.0
    entry: float = 0.0;          stop_loss: float = 0.0
    target1: float = 0.0;        target2: float = 0.0
    rrr: float = 0.0
    stop_pct: float = 0.0;       gain_pct_t1: float = 0.0
    pattern_score: int = 0;      rs_score: int = 0
    volume_score: int = 0;       market_score: int = 0
    sector_score: int = 0;       rsi_score: int = 0
    ema_score: int = 0;          macd_score: int = 0
    liq_score: int = 0;          bonus_score: int = 0
    fundamental_score: int = 0;  institutional_score: int = 0
    raw_score: int = 0;          total_score: int = 0
    confidence_pct: float = 0.0
    rs_percentile: float = 0.0;  rs_persistence: int = 0
    market_regime: str = ""
    regime: str = "";            regime_name: str = ""
    adt_cr: float = 0.0;         mcap_cr: float = 0.0
    mcap_tier: str = "";         part_rate: float = 0.0
    breadth_score: int = 0;      rsi_val: float = 0.0
    rvol: float = 0.0;           del_pct: float = 0.0
    what_is_working: List[str] = field(default_factory=list)
    what_is_missing: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    rejected: bool = False;      reject_reason: str = ""
    # v5 fields
    asymmetry_risk_pct:   float = 0.0
    asymmetry_reward_pct: float = 0.0
    asymmetry_rr:         float = 0.0
    asymmetry_fail_stage: str   = ""
    vcp_w4_pct:           float = 0.0
    vcp_contracting:      bool  = False
    vcp_penalty:          int   = 0
    breakout_quality:     str   = ""
    earnings_flag:        bool  = False
    earnings_score:       int   = 0
    headroom_pct:         float = 0.0
    macro_state:          str   = "MIXED"
    macro_score:          int   = 3
    event_risk:           str   = "NORMAL"
    confirmation_state:   str   = "SETUP_READY"
    confirmation_score:   int   = 3


class AgentOrchestrator:
    def __init__(self, data_dict: dict):
        self.data = data_dict

        # Breadth
        breadth_agent = MarketBreadthAgent(
            stock_data=data_dict.get("stock_data", {}),
            nifty_df=data_dict.get("nifty50_data", pd.DataFrame()),
        )
        breadth_result           = breadth_agent.compute()
        self.breadth_score       = breadth_result["breadth_score"]
        self.breadth_detail      = breadth_result
        data_dict["breadth_score"] = self.breadth_score

        # Market regime
        self.market_agent   = MarketAgent(data_dict)
        self.regime         = self.market_agent.get_regime()
        self.regime_name    = self.market_agent.get_regime_name()
        self.market_score   = self.market_agent.score()
        self.regime_penalty = self.market_agent.get_penalty()

        # Sector
        self.sector_agent   = SectorAgent(data_dict)
        self.sector_ranks   = self.sector_agent.get_ranks()

        # RS ranks
        self.universe_rs_ranks = compute_universe_ranks(data_dict)

        # MacroAgent (v5)
        vix     = data_dict.get("vix", 15.0)
        adv_dec = breadth_result.get("adv_dec_ratio", 1.0)
        fii     = data_dict.get("fii_flow", 0.0)
        self.macro_agent = MacroAgent(
            vix=vix, breadth_score=self.breadth_score,
            fii_flow=fii, adv_dec_ratio=adv_dec
        )
        self.macro_state = self.macro_agent.get_state()
        self.macro_score = self.macro_agent.get_score()
        self.t1_cap      = self.macro_agent.get_t1_cap(self.regime)

        # EventRiskAgent (v5)
        self.event_agent = EventRiskAgent()
        self.event_state = self.event_agent.get_state()
        self.event_penalty = self.event_agent.get_score_penalty()

        log.info(f"Regime: {self.regime} ({self.regime_name}) | "
                 f"Breadth: {self.breadth_score}/10 | Macro: {self.macro_state} | "
                 f"Event: {self.event_state} | T1 cap: {self.t1_cap}")

    def run(self, ticker: str, name: str, sector: str, df: pd.DataFrame,
            delivery_data: dict = None, universe: str = "LARGE") -> StockResult:

        r = StockResult(
            ticker=ticker, name=name, sector=sector,
            price=float(df["Close"].iloc[-1]) if not df.empty else 0.0,
            universe=universe, market_regime=self.regime,
            breadth_score=self.breadth_score,
            macro_state=self.macro_state,
            macro_score=self.macro_score,
            event_risk=self.event_state,
        )

        if df.empty or len(df) < 60:
            r.rejected = True; r.reject_reason = "Insufficient price history"
            return r

        cfg     = UNIVERSE_CONFIG[universe]
        del_pct = (delivery_data or {}).get(ticker.replace(".NS",""), 0.0)
        r.del_pct = del_pct

        # G1: Liquidity
        liq = LiquidityAgent(df, universe=universe)
        if not liq.passes():
            r.rejected = True; r.reject_reason = liq.reject_reason()
            return r
        r.liq_score = liq.score(); r.adt_cr = liq.get_adt()
        r.part_rate = liq.get_part_rate(); r.mcap_tier = liq.get_mcap_tier()

        # G2: Pattern
        pa = PatternAgent(df)
        if not pa.pattern:
            r.rejected = True; r.reject_reason = "No pattern detected"
            return r
        dyn_weight        = get_dynamic_weight(pa.pattern)
        r.pattern         = pa.pattern
        r.breakout_level  = pa.breakout_level
        r.entry_low       = pa.entry_low
        r.entry_high      = pa.entry_high
        r.pattern_score   = min(dyn_weight, 18)
        r.ema_score       = pa.get_ema_score()
        r.macd_score      = pa.get_macd_score()
        r.rsi_score       = pa.get_rsi_score()
        r.breakout_quality = getattr(pa, "breakout_quality", "MINOR")

        try:
            import ta
            r.rsi_val = float(
                ta.momentum.RSIIndicator(df["Close"].squeeze(), 14).rsi().iloc[-1])
        except Exception:
            # Fallback: pure pandas RSI calculation
            try:
                c     = df["Close"].squeeze()
                delta = c.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain / loss.replace(0, float('nan'))
                rsi_s = 100 - 100 / (1 + rs)
                r.rsi_val = float(rsi_s.iloc[-1])
            except Exception:
                r.rsi_val = 0.0

        vol    = df["Volume"].squeeze().to_numpy(dtype=float)
        avg20v = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1
        r.rvol = round(float(vol[-1]) / avg20v, 2)

        # G3: RS (gate now 30th percentile)
        rsa = RSAgent(
            df, self.data.get("nifty50_data", pd.DataFrame()),
            universe_ranks=self.universe_rs_ranks, ticker=ticker
        )
        r.rs_score       = min(int(rsa.score() * cfg["rs_weight_mult"]), 20)
        r.rs_percentile  = rsa.get_percentile()
        r.rs_persistence = rsa.get_persistence()

        if not rsa.passes_gate():
            r.rejected = True
            r.reject_reason = f"RS Percentile {r.rs_percentile:.0f} < 30"
            return r

        # Volume score
        va = VolumeAgent(df, del_pct, universe=universe)
        r.volume_score = min(int(va.score() * cfg["vol_weight_mult"]), 12)

        # Market
        r.market_score  = self.market_score
        r.regime        = self.regime
        r.regime_name   = self.regime_name
        r.market_regime = self.regime

        # Sector
        r.sector_score = self.sector_agent.score_for_sector(sector, self.sector_ranks)

        # G4: Risk
        risk = RiskAgent(df, r.breakout_level, r.entry_low, r.entry_high, universe=universe)
        if not risk.passes():
            r.rejected = True; r.reject_reason = risk.reject_reason()
            return r
        r.entry = risk.entry; r.stop_loss = risk.stop
        r.target1 = risk.target1; r.target2 = risk.target2
        r.rrr = risk.rrr; r.stop_pct = risk.stop_pct; r.gain_pct_t1 = risk.gain_pct

        # G5: AsymmetryGate
        ag = AsymmetryGate(entry=r.entry_high, stop=r.stop_loss,
                           target1=r.target1, universe=universe)
        ag_result = ag.check()
        r.asymmetry_risk_pct   = ag_result["risk_pct"]
        r.asymmetry_reward_pct = ag_result["reward_pct"]
        r.asymmetry_rr         = ag_result["rr_ratio"]
        r.asymmetry_fail_stage = ag_result["fail_stage"]
        if not ag_result["qualified"]:
            r.rejected = True; r.reject_reason = ag_result["fail_reason"]
            return r

        # G6: VCP
        vcpg = VCPContractionGate(df=df)
        vcp  = vcpg.check()
        r.vcp_w4_pct = vcp["w4_pct"]; r.vcp_contracting = vcp["contracting"]
        r.vcp_penalty = vcp["penalty"]
        if vcp["hard_reject"]:
            r.rejected = True; r.reject_reason = vcp["fail_reason"]
            return r

        # G7: Headroom
        if r.entry_high > 0 and r.target1 > r.entry_high:
            r.headroom_pct = round((r.target1 - r.entry_high) / r.entry_high * 100, 2)
        if r.headroom_pct < 4.5:
            r.rejected = True
            r.reject_reason = (f"Headroom {r.headroom_pct:.1f}% < 4.5% "
                               f"(T1 {r.target1:.0f} vs entry {r.entry_high:.0f})")
            return r

        # Fundamental proxy
        fp = FundamentalProxyAgent(ticker, df, del_pct, r.rs_percentile,
                                    self.sector_ranks.get(sector, 7))
        r.fundamental_score = fp.evaluate()["fundamental_proxy_score"]

        # Institutional proxy
        ip = InstitutionalProxyAgent(ticker, df, del_pct)
        r.institutional_score = ip.evaluate()["institutional_proxy_score"]

        r.bonus_score = min(r.liq_score // 2
                            + r.fundamental_score // 4
                            + r.institutional_score // 4, 5)

        # Earnings catalyst
        try:
            from agents.earnings_catalyst_agent import EarningsCatalystAgent
            eca = EarningsCatalystAgent(ticker=ticker, df=df).analyze()
            r.earnings_flag  = eca.get("catalyst_found", False)
            r.earnings_score = eca.get("score", 0)
        except Exception as e:
            log.debug("EarningsCatalystAgent %s: %s", ticker, e)

        # ConfirmationAgent (v5)
        conf = ConfirmationAgent(ticker, r.entry, r.stop_loss, r.breakout_level)
        r.confirmation_state = conf.get_state()
        r.confirmation_score = conf.get_score()

        # Master score (v5 rebalanced)
        # RS(18) + Pattern(16) + RSI(10) + Volume(12) + EMA(8)
        # + Market(6-macro) + MACD(4) + Sector(8) + Confirmation(6)
        # + Bonus(2) + Earnings(4) - VCP penalty = ~94 base + extras
        r.raw_score = (
            r.rs_score       +   # 0-18 (rebalanced from 20)
            r.pattern_score  +   # 0-16 (rebalanced from 18)
            r.rsi_score      +   # 0-10 (rebalanced from 15)
            r.volume_score   +   # 0-12
            r.ema_score      +   # 0-8  (rebalanced from 10)
            r.market_score   +   # 0-5
            r.macd_score     +   # 0-4  (rebalanced from 8)
            r.sector_score   +   # 0-8  (rebalanced from 7)
            r.bonus_score        # 0-2  (compressed from 5)
        )

        penalty = int(self.regime_penalty * cfg["regime_penalty_mult"])
        r.total_score = max(0, (
            r.raw_score
            + penalty
            + r.earnings_score       # 0 or 4
            + self.macro_score        # 0, 3, or 6
            + r.confirmation_score    # 3 or 6
            - r.vcp_penalty           # 0 or 5
            - self.event_penalty      # 0, 2, or 5
        ))

        # Conviction
        ca = ConvictionAgent()
        r.confidence_pct = ca.calibrate_confidence(r.pattern, r.total_score, universe)

        # Tier assignment
        gate = cfg["score_gate"]
        # Event risk raises the effective gate
        effective_gate = gate + self.event_penalty
        if r.total_score >= effective_gate:
            r.tier = 1
            r.what_is_working = self._why_working(r)
        elif r.total_score >= 55:
            r.tier = 2
            r.what_is_working    = self._why_working(r)
            r.what_is_missing    = self._what_missing(r, effective_gate)
            r.trigger_conditions = self._triggers(r)
        elif r.total_score >= 42:
            r.tier = 3
            r.what_is_working    = self._why_working(r)[:2]
            r.trigger_conditions = self._triggers(r)
        else:
            r.rejected = True
            r.reject_reason = f"Score {r.total_score} below watchlist threshold"
            return r

        r.risk_factors = self._risk_factors(r)
        return r

    def run_universe(self, universe_items: list,
                     stock_data: dict, delivery_data: dict) -> dict:
        all_results    = []
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
                key = f"Exception: {type(e).__name__}"
                reject_reasons[key] = reject_reasons.get(key, 0) + 1

        # Rejection buckets
        if reject_reasons:
            buckets = {}
            for reason, count in reject_reasons.items():
                if "AsymmetryGate" in reason and "risk" in reason:
                    key = "AsymmetryGate: stop too wide"
                elif "AsymmetryGate" in reason and "reward" in reason:
                    key = "AsymmetryGate: reward too small"
                elif "AsymmetryGate" in reason:
                    key = "AsymmetryGate: R:R below minimum"
                elif "VCPGate" in reason:
                    key = "VCPGate: W4 too loose"
                elif "Headroom" in reason:
                    key = "Headroom < 4.5%"
                elif reason.startswith("R:R"):
                    key = "R:R below minimum (RiskAgent)"
                elif reason.startswith("ADT"):
                    key = "ADT below minimum"
                elif "pattern" in reason.lower():
                    key = "No pattern detected"
                elif reason.startswith("Score"):
                    key = "Score below T3 threshold (42)"
                elif "Insufficient" in reason:
                    key = "Insufficient price history"
                elif "RS Percentile" in reason:
                    key = reason
                else:
                    key = reason
                buckets[key] = buckets.get(key, 0) + count
            log.info("  Rejection breakdown:")
            for reason, count in sorted(buckets.items(), key=lambda x: -x[1])[:12]:
                log.info(f"    {count:>3} x {reason}")

        all_results.sort(key=lambda x: x.total_score, reverse=True)

        t1 = [r for r in all_results if r.tier == 1]
        t2 = [r for r in all_results if r.tier == 2]
        t3 = [r for r in all_results if r.tier == 3]

        # Ranking funnel - apply T1 cap (P1 item)
        if len(t1) > self.t1_cap:
            # Sort by: confirmation state first, then RS persistence, then score
            t1.sort(key=lambda r: (
                0 if r.confirmation_state == "BREAKOUT_CONFIRMED" else 1,
                -r.rs_persistence,
                -r.total_score
            ))
            overflow = t1[self.t1_cap:]
            t1 = t1[:self.t1_cap]
            # Demote overflow to T2
            for r in overflow:
                r.tier = 2
            t2 = overflow + t2
            t2.sort(key=lambda r: -r.total_score)

        log.info(f"  Tier 1 (Top Picks):  {len(t1)} (cap={self.t1_cap})")
        log.info(f"  Tier 2 (Aggressive): {len(t2)}")
        log.info(f"  Tier 3 (Watchlist):  {len(t3)}")
        log.info(f"  Macro: {self.macro_state} | Event: {self.event_state}")

        # Auto-log T1 picks
        if t1:
            try:
                from trade_logger import auto_log_t1_picks
                logged = auto_log_t1_picks(t1, regime=self.regime)
                log.info(f"  Auto-logged {logged} T1 picks to trades_v4")
            except Exception as e:
                log.warning(f"  Auto-log failed: {e}")

        # Near-breakout watchlist
        existing      = {r.ticker for r in all_results}
        near_breakout = find_near_breakout_stocks(
            universe_items, stock_data, delivery_data, existing
        )
        log.info(f"  Near-breakout watchlist: {len(near_breakout)} stocks")
        import json
        picks_json = []
        for r in t1 + t2:
            picks_json.append({
                "ticker":  r.ticker,
                "name":    r.name,
                "sector":  r.sector,
                "security_id": "",
                "segment": "NSE_EQ",
                "entry":   round(r.entry, 2),
                "sl":      round(r.stop_loss, 2),
                "pivot":   round(r.breakout_level if r.breakout_level > 0 else r.entry, 2),
                "t1":      round(r.target1, 2),
                "t2":      round(r.target2, 2),
                "rr":      round(r.asymmetry_rr, 1),
                "score":   r.total_score,
                "tier":    r.tier,
                "pattern": r.pattern or "",
            })
        with open("picks_latest.json", "w") as f:
            json.dump(picks_json, f, indent=2)
        log.info(f"  Saved {len(picks_json)} picks to picks_latest.json")
        return {
            "tier1": t1, "tier2": t2, "tier3": t3,
            "near_breakout": near_breakout,
            "all_results": all_results,
            "regime": self.regime, "regime_name": self.regime_name,
            "breadth": self.breadth_score,
            "breadth_detail": self.breadth_detail,
            "macro_state": self.macro_state,
            "event_risk": self.event_state,
            "t1_cap": self.t1_cap,
        }

    def _why_working(self, r: StockResult) -> List[str]:
        reasons = []
        if r.rs_percentile >= 70:
            reasons.append(f"RS Rank {r.rs_percentile:.0f}th - outperforming {r.rs_percentile:.0f}% of market")
        if r.pattern:
            bq = f" [{r.breakout_quality}]" if r.breakout_quality else ""
            reasons.append(f"{r.pattern}{bq} - entry Rs.{r.entry_low:.1f}-{r.entry_high:.1f}")
        if r.rvol >= 1.5:
            reasons.append(f"Volume {r.rvol:.1f}x avg - institutional activity")
        if r.del_pct >= 50:
            reasons.append(f"Delivery {r.del_pct:.0f}% - holders not selling")
        if r.sector_score >= 5:
            top = self.sector_agent.get_top_sectors(3)
            if r.sector in [s for s, _ in top]:
                reasons.append(f"Sector leadership - {r.sector} top-3 in rotation")
        if r.rsi_val > 0:
            reasons.append(f"RSI {r.rsi_val:.0f} - momentum constructive")
        if r.earnings_flag:
            reasons.append("Earnings acceleration within 14 days - catalyst-backed")
        if r.confirmation_state == "BREAKOUT_CONFIRMED":
            reasons.append("Breakout confirmed - held above pivot 1+ sessions")
        if r.rs_persistence >= 8:
            reasons.append(f"RS persistence {r.rs_persistence}/13 weeks - sustained leader")
        return reasons[:4]

    def _what_missing(self, r: StockResult, gate: int) -> List[str]:
        missing = []
        gap = gate - r.total_score
        missing.append(f"Score {r.total_score} - needs {gap} more pts for Tier 1")
        if r.rs_percentile < 60:
            missing.append(f"RS {r.rs_percentile:.0f}th pct - stronger RS needed")
        if r.rvol < 1.3:
            missing.append(f"Volume {r.rvol:.1f}x - needs breakout volume >= 1.5x")
        if r.regime in ["C", "D", "E"]:
            missing.append(f"Regime {r.regime} penalty applied")
        return missing[:3]

    def _triggers(self, r: StockResult) -> List[str]:
        triggers = []
        if r.breakout_level > 0:
            triggers.append(f"Close above Rs.{r.breakout_level:.1f} on 1.5x volume")
        triggers.append("RS rank improves above 65th percentile")
        if r.regime in ["C", "D"]:
            triggers.append("Market regime shifts to B or better")
        triggers.append(f"Stop: Rs.{r.stop_loss:.1f} ({r.stop_pct:.1f}% risk)")
        return triggers[:3]

    def _risk_factors(self, r: StockResult) -> List[str]:
        risks = []
        if r.regime in ["D", "E"]:
            risks.append(f"Market {r.regime_name} - elevated breakout failure rate")
        if r.asymmetry_rr > 0 and r.asymmetry_rr < 2.5:
            risks.append(f"R:R {r.asymmetry_rr:.1f}x - size position accordingly")
        if r.vcp_penalty > 0:
            risks.append(f"VCP W4={r.vcp_w4_pct:.1f}% not fully compressed")
        if r.universe == "SMALL":
            risks.append("Small-cap - wider spreads, lower liquidity")
        if r.breadth_score <= 3:
            risks.append(f"Breadth {r.breadth_score}/10 - sector risk elevated")
        if self.macro_state == "HOSTILE":
            risks.append("Macro HOSTILE - reduce position size")
        if r.confirmation_state == "SETUP_READY":
            risks.append("Not yet confirmed - wait for next session close above pivot")
        return risks[:3]
