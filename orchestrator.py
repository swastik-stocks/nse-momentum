"""
NSE Momentum v5.3 - Orchestrator
All 14 agents wired. P0+P1+P2 implementation.

CHANGES vs v5.2:
  [BUG-1] MarketBreadthAgent now fetches NSE-WIDE A/D ratio (all ~2000 traded
          symbols) instead of computing it from our 401-stock universe subset.
          On 24 Jun 2026 this caused A/D=0.27 when actual NSE was 1.07.
          Fix: fetch_nse_wide_breadth() → fallback to full Bhavcopy (~3200 sym)

  [BUG-2] MacroAgent VIX thresholds recalibrated. VIX 13.33 was landing in
          HOSTILE bucket. Now correctly maps VIX < 14 → +3 pts (Healthy bull).

  [BUG-3] Regime confidence score added. When breadth signals contradict
          (e.g. above_50_ema=64.9% but A/D=0.27), regime is marked LOW_CONFIDENCE
          and penalty is dampened: D penalty -12 → -5 pts, T1 cap 0 → 5 picks.

  [BUG-4] MacroAgent FII fallback fixed. If NSE FII CSV fetch fails, macro
          no longer defaults to HOSTILE — uses neutral (0 pts) with a warning.

  [BUG-5] Calibration logs auto-written to logs/breadth_calibration.log and
          logs/regime_calibration.log for daily audit.

CHANGES vs v5.3 (this edit):
  [NEW] Defensive/RS watchlist wired in — see agents/defensive_agent.py.
        Only produces output when regime is D or E (matches the existing
        t1_cap=0 risk-off behaviour in those regimes). Saved separately to
        defensive_watchlist.json, never merged into picks_latest.json.
"""

import sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from loguru import logger as log
    log.remove()
    log.add(sys.stderr, level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    log.add("logs/orchestrator_{time:YYYY-MM-DD}.log", level="DEBUG",
            rotation="1 day", retention="14 days", compression="zip")
except ImportError:
    import logging
    log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents.pattern_agent              import PatternAgent, is_low_edge_pattern, PATTERN_EXPECTANCY
from agents.rs_agent                   import RSAgent, compute_universe_ranks, compute_sector_relative_ranks
from agents.volume_agent               import VolumeAgent
from agents.market_agent               import MarketAgent
from agents.market_breadth_agent       import (
    fetch_nse_wide_breadth,
    fetch_breadth_from_bhavcopy,
    compute_breadth_score,
    print_breadth_dashboard,
)
from agents.sector_agent               import SectorAgent
from agents.risk_agent               import RiskAgent, STOP_CAP
from agents.liquidity_agent            import LiquidityAgent
from agents.conviction_agent           import ConvictionAgent
from agents.fundamental_proxy_agent    import FundamentalProxyAgent
from agents.institutional_proxy_agent  import InstitutionalProxyAgent
from agents.asymmetry_gate             import AsymmetryGate
from agents.vcp_gate                   import VCPContractionGate
from agents.macro_agent                import MacroAgent, fetch_fii_flow_crore
from agents.event_risk_agent           import EventRiskAgent
from agents.confirmation_agent         import ConfirmationAgent
from agents.near_breakout              import find_near_breakout_stocks
from agents.regime_classifier          import RegimeClassifier
from agents.defensive_agent            import run_defensive_scan
from agents.weekly_trend_agent         import WeeklyTrendAgent
import data_fetcher
from trade_logger                      import get_dynamic_weight
from nse_universe                      import UNIVERSE_CONFIG, UNIVERSE_SEED

# Hard cap on T2 picks sent to picks_latest.json and confirmation email
T2_CAP = 8

# Sector cap — max T1 picks in a single sector
SECTOR_T1_CAP = 3


# ─────────────────────────────────────────────────────────────────────────────
# Circuit limit pre-filter  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _is_at_circuit_limit(ticker: str) -> bool:
    try:
        from nsepython import nse_get_quote_info
        symbol = ticker.replace(".NS", "")
        info   = nse_get_quote_info(symbol)
        band   = str(info.get("priceBand", "")).strip()
        if band in ("5%", "10%", "15%", "20%"):
            log.debug(f"Circuit limit {band} detected for {ticker} — rejecting")
            return True
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"Circuit check failed for {ticker}: {e}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Sector Concentration Gate  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class SectorConcentrationGate:
    def __init__(self, max_per_sector: int = SECTOR_T1_CAP):
        self.max_per_sector = max_per_sector
        self._counts: Dict[str, int] = {}

    def can_add(self, sector: str) -> bool:
        return self._counts.get(sector, 0) < self.max_per_sector

    def add(self, sector: str):
        self._counts[sector] = self._counts.get(sector, 0) + 1

    def summary(self) -> str:
        return "  ".join(f"{s}:{n}" for s, n in self._counts.items())


# ─────────────────────────────────────────────────────────────────────────────
# StockResult dataclass  (unchanged from v5.2)
# ─────────────────────────────────────────────────────────────────────────────

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
    rs_sector_pct: float = 0.0   # NEW — Mansfield sector-relative RS percentile
    low_edge_pattern: bool = False   # NEW — True if pattern's own backtested expectancy < 0.15%
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
    circuit_limit:        str   = ""
    # v5.3 additions
    regime_confidence:    str   = "HIGH"   # HIGH or LOW
    regime_sanity_flags:  List[str] = field(default_factory=list)
    breadth_source:       str   = ""       # NSE_LIVE_API / BHAVCOPY_FULL / DEFAULT
    weekly_trend_bonus:   float = 0.0
    weekly_trend_weeks:   int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# P2-05 support: technical stop-loss for an EXISTING holding
# ─────────────────────────────────────────────────────────────────────────────

def compute_holding_stop(df: pd.DataFrame, universe: str = "LARGE") -> dict:
    """
    Reuses RiskAgent's exact three-candidate stop methodology (EMA21 / 10-day
    low / ATR-based, whichever is tightest) and its per-tier STOP_CAP table --
    same risk philosophy as fresh entries. The one deliberate difference:
    anchored to today's closing price instead of a breakout entry zone, since
    an existing holding has no "entry" decision left to make -- only "where's
    my protective stop from here." Does NOT reuse RiskAgent's target/R:R
    gating -- that logic is about whether a NEW trade is worth taking, not
    about protecting a position you already hold.

    Returns None if there isn't enough price history to compute (mirrors
    RiskAgent's own "No valid breakout level" / "ATR calculation failed"
    guards) -- callers must handle that, same as any other missing-data case
    in this pipeline.
    """
    if df.empty or len(df) < 15:
        return None

    high  = df["High"].squeeze().to_numpy(dtype=float)
    low   = df["Low"].squeeze().to_numpy(dtype=float)
    close = df["Close"].squeeze().to_numpy(dtype=float)
    price = close[-1]

    atr = RiskAgent._atr(high, low, close, 14)
    if atr <= 0:
        return None

    ema21    = float(pd.Series(close).ewm(span=21, adjust=False).mean().iloc[-1])
    stop_ema = ema21 * 0.993

    lookback_stop = min(len(low), 10)
    stop_10d = float(np.min(low[-lookback_stop:])) * 0.997

    atr_mult = {"LARGE": 1.5, "MID": 1.8, "SMALL": 2.0}.get(universe, 1.5)
    stop_atr = price - atr_mult * atr

    candidates = [(s, label) for s, label in
                  [(stop_ema, "EMA21"), (stop_10d, "10D_LOW"), (stop_atr, "ATR")]
                  if 0 < s < price]
    if not candidates:
        return None
    stop, method = max(candidates, key=lambda x: x[0])

    cap       = STOP_CAP.get(universe, 0.03)
    cap_price = price * (1 - cap)
    if stop < cap_price:
        stop, method = cap_price, "CAPPED"
    stop = max(stop, 0.01)

    return {
        "current_price": round(price, 2),
        "stop":          round(stop, 2),
        "stop_pct":      round((price - stop) / price * 100, 1),
        "method":        method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AgentOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    def __init__(self, data_dict: dict):
        self.data = data_dict

        # ── STEP 1: Fetch NSE-WIDE breadth (BUG-1 FIX) ───────────────────────
        # Old code computed A/D from our 401-stock universe → wrong on divergent days.
        # New code: hit NSE live API (all ~2000 symbols), fall back to full Bhavcopy.
        log.info("  Fetching NSE-wide market breadth...")
        breadth_raw = fetch_nse_wide_breadth()

        if breadth_raw is None:
            log.warning("  NSE live breadth API failed — falling back to Bhavcopy.")
            bhavcopy_full = data_dict.get("bhavcopy_full_df")   # full ~3200-symbol df
            if bhavcopy_full is not None and not bhavcopy_full.empty:
                breadth_raw = fetch_breadth_from_bhavcopy(bhavcopy_full)
            else:
                log.warning(
                    "  Bhavcopy full df not available in data_dict. "
                    "Key expected: 'bhavcopy_full_df'. "
                    "Breadth will default to neutral."
                )

        # Compute above_50_ema_pct from our internal universe (still valid —
        # this is a universe-internal metric, not market-wide, so universe-bounded is OK)
        above_50_ema_pct = self._compute_above_50_ema(
            data_dict.get("stock_data", {}),
            data_dict.get("nifty50_data", pd.DataFrame()),
        )

        breadth_result = compute_breadth_score(
            breadth_data=breadth_raw,
            above_50_ema_pct=above_50_ema_pct,
        )

        self.breadth_score    = int(round(breadth_result["breadth_score"]))
        self.breadth_detail   = breadth_result
        self.breadth_source   = breadth_result.get("source", "UNKNOWN")
        self.breadth_conf     = breadth_result.get("confidence", "HIGH")

        data_dict["breadth_score"]    = self.breadth_score
        data_dict["adv_dec_ratio"]    = breadth_result.get("ad_ratio", 1.0)
        data_dict["above_50_ema_pct"] = above_50_ema_pct

        # Print breadth dashboard to log
        print_breadth_dashboard(breadth_result)

        if breadth_result.get("warnings"):
            for w in breadth_result["warnings"]:
                log.warning(f"  BREADTH WARNING: {w}")

        # ── STEP 2: Market regime via existing MarketAgent ────────────────────
        # MarketAgent handles Nifty EMA stack (unchanged — still correct)
        self.market_agent   = MarketAgent(data_dict)
        self.regime         = self.market_agent.get_regime()
        self.regime_name    = self.market_agent.get_regime_name()
        self.market_score   = self.market_agent.score()

        # ── STEP 3: Regime confidence + sanity checks (BUG-3 FIX) ────────────
        # RegimeClassifier cross-checks VIX / A/D / above_50_ema for contradictions.
        # If contradictions found: confidence=LOW, penalty dampened, T1 cap loosened.
        vix = data_dict.get("vix", 15.0)
        classifier = RegimeClassifier(
            nifty50_data      = data_dict.get("nifty50_data",   pd.DataFrame()),
            banknifty_data    = data_dict.get("banknifty_data", pd.DataFrame()),
            vix               = vix,
            ad_ratio          = breadth_result.get("ad_ratio", 1.0),
            breadth_score     = breadth_result["breadth_score"],
            above_50_ema_pct  = above_50_ema_pct if above_50_ema_pct else 50.0,
            breadth_confidence= self.breadth_conf,
            macro_state       = "MIXED",   # placeholder — set properly after MacroAgent
        )
        regime_result = classifier.classify()

        # Regime letter and penalty both come from MarketAgent's EMA-based classification.
        # RegimeClassifier is used ONLY for sanity checks and confidence scoring —
        # not for the penalty, which must match the displayed regime letter.
        REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}
        self.regime_penalty     = REGIME_PENALTIES.get(self.regime, -5)
        self.regime_confidence  = regime_result["confidence"]     # HIGH or LOW
        self.regime_sanity_flags= regime_result["sanity_flags"]   # list of contradictions

        if self.regime_confidence == "LOW":
            log.warning(
                f"  REGIME LOW_CONFIDENCE: penalty dampened "
                f"{regime_result['full_penalty']} → {self.regime_penalty} pts. "
                f"{len(self.regime_sanity_flags)} contradiction(s) detected."
            )
            for flag in self.regime_sanity_flags:
                log.warning(f"    🚩 {flag}")
        else:
            log.info(f"  Regime confidence: HIGH — all signals consistent.")

        # ── STEP 4: Sector ────────────────────────────────────────────────────
        self.sector_agent = SectorAgent(data_dict)
        self.sector_ranks = self.sector_agent.get_ranks()

        # ── STEP 5: RS ranks ──────────────────────────────────────────────────
        self.universe_rs_ranks = compute_universe_ranks(data_dict)
        # NEW — Mansfield sector-relative RS: stock vs its OWN sector peers,
        # not the whole ~401-stock universe. See agents/rs_agent.py for the
        # full rationale (this was documented as a v5 change but never
        # actually implemented until now).
        self.sector_rs_ranks = compute_sector_relative_ranks(
            data_dict, universe_meta=data_dict.get("universe_meta", {})
        )

        # ── STEP 6: MacroAgent (BUG-2 + BUG-4 FIX) ───────────────────────────
        # BUG-2: VIX thresholds recalibrated — VIX 13.33 now correctly = BENIGN (+3 pts)
        # BUG-4: FII fetch failure now uses neutral (0 pts), not HOSTILE
        log.info("  Fetching FII flow from NSE...")
        fii_flow = fetch_fii_flow_crore()   # returns None on failure — handled gracefully
        if fii_flow is None:
            log.warning("  FII flow unavailable — MacroAgent will use neutral (0 pts).")

        self.macro_agent = MacroAgent(
            vix               = vix,
            breadth_score     = breadth_result["breadth_score"],
            fii_flow_crore    = fii_flow,           # None = neutral, NOT hostile
            adv_dec_ratio     = breadth_result.get("ad_ratio", 1.0),
            breadth_confidence= self.breadth_conf,
        )
        self.macro_state = self.macro_agent.get_state()
        self.macro_score = self.macro_agent.get_score()

        # Now re-run RegimeClassifier with correct macro_state for audit log
        # (does not change penalty — macro_state is informational in classifier)
        classifier._macro_state = self.macro_state

        # ── STEP 6b: Optional NIM narration (annotation only, never authoritative) ──
        # Placed HERE (after MacroAgent, not before) so it gets real macro_state
        # and fii_flow rather than a hardcoded placeholder. Fail-safe by design —
        # see agents/regime_narrator.py docstring. If NVIDIA_API_KEY is unset or
        # the call fails, this silently no-ops and the scan proceeds exactly as
        # it always has. Never overrides regime/confidence/penalty above.
        try:
            from agents.regime_narrator import narrate_regime
            narration = narrate_regime({
                "regime": self.regime, "regime_name": self.regime_name,
                "regime_confidence": self.regime_confidence,
                "regime_penalty": self.regime_penalty,
                "above_50_ema_pct": above_50_ema_pct,
                "ad_ratio": breadth_result.get("ad_ratio", 1.0),
                "breadth_score": breadth_result["breadth_score"],
                "vix": vix,
                "macro_state": self.macro_state,
                "fii_flow_crore": fii_flow,
                "sanity_flags": self.regime_sanity_flags,
            })
            self.regime_narrative = narration.get("narrative", "")
            if self.regime_narrative:
                log.info(f"  Regime narrative: {self.regime_narrative[:200]}...")
        except Exception as e:
            # Belt-and-suspenders on top of regime_narrator.py's own internal
            # fail-safe — an optional annotation feature must never be able
            # to take down the evening scan.
            log.warning(f"  Regime narration skipped (non-fatal): {e}")
            self.regime_narrative = ""

        # T1 cap: use macro agent's cap, but floor at regime_result t1_cap
        # (e.g. LOW_CONFIDENCE Regime D gets t1_cap=5, macro may say 15 — take min)
        macro_t1_cap   = self.macro_agent.get_t1_cap()
        regime_t1_cap  = regime_result["t1_cap"]
        self.t1_cap    = min(macro_t1_cap, regime_t1_cap) if regime_t1_cap > 0 else macro_t1_cap
        # Special case: HIGH_CONF Regime D always forces t1_cap=0 (full conviction bear call)
        if self.regime == "D" and self.regime_confidence == "HIGH":
            self.t1_cap = 0
        elif self.regime == "E":
            self.t1_cap = 0   # Bear market — always 0 regardless of confidence

        # ── STEP 7: Event risk ────────────────────────────────────────────────
        self.event_agent   = EventRiskAgent()
        self.event_state   = self.event_agent.get_state()
        self.event_penalty = self.event_agent.get_score_penalty()

        self.macro_agent.print_summary()

        log.info(
            f"  Regime: {self.regime} ({self.regime_name}) "
            f"[conf={self.regime_confidence}] | "
            f"Penalty: {self.regime_penalty} pts | "
            f"Breadth: {self.breadth_score}/10 [{self.breadth_source}] | "
            f"Macro: {self.macro_state} (+{self.macro_score} pts) | "
            f"Event: {self.event_state} | "
            f"T1 cap: {self.t1_cap} | T2 cap: {T2_CAP}"
        )

    # ── helper: above_50_ema from internal universe ───────────────────────────

    @staticmethod
    def _compute_above_50_ema(stock_data: dict, nifty_df: pd.DataFrame) -> Optional[float]:
        """
        Compute % of our universe stocks above their 50-day EMA.
        This is a universe-internal metric — universe-bounded is correct here.
        Returns float (0–100) or None if stock_data is empty.
        """
        if not stock_data:
            return None
        above = 0
        total = 0
        for ticker, df in stock_data.items():
            if df.empty or len(df) < 55:
                continue
            try:
                close = df["Close"].squeeze()
                ema50 = float(close.ewm(span=50).mean().iloc[-1])
                last  = float(close.iloc[-1])
                total += 1
                if last > ema50:
                    above += 1
            except Exception:
                continue
        if total == 0:
            return None
        return round(above / total * 100, 1)

    # ── per-stock evaluation (unchanged from v5.2) ────────────────────────────

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
            regime_confidence=self.regime_confidence,
            regime_sanity_flags=self.regime_sanity_flags,
            breadth_source=self.breadth_source,
        )

        if df.empty or len(df) < 60:
            r.rejected = True; r.reject_reason = "Insufficient price history"
            return r

        cfg     = UNIVERSE_CONFIG[universe]
        del_pct = (delivery_data or {}).get(ticker.replace(".NS", ""), 0.0)
        r.del_pct = del_pct

        # [P2] Circuit limit pre-filter
        if _is_at_circuit_limit(ticker):
            r.rejected     = True
            r.circuit_limit = "CIRCUIT"
            r.reject_reason = f"Circuit limit active — stop orders cannot execute"
            return r

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
        dyn_weight         = get_dynamic_weight(pa.pattern)
        r.pattern          = pa.pattern
        r.breakout_level   = pa.breakout_level
        r.entry_low        = pa.entry_low
        r.entry_high       = pa.entry_high
        r.pattern_score    = min(dyn_weight, 18)
        r.ema_score        = pa.get_ema_score()
        r.macd_score       = pa.get_macd_score()
        r.rsi_score        = pa.get_rsi_score()
        r.breakout_quality = getattr(pa, "breakout_quality", "MINOR")

        # RSI value
        try:
            import ta
            r.rsi_val = float(
                ta.momentum.RSIIndicator(df["Close"].squeeze(), 14).rsi().iloc[-1])
        except Exception:
            try:
                c     = df["Close"].squeeze()
                delta = c.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain / loss.replace(0, float("nan"))
                rsi_s = 100 - 100 / (1 + rs)
                r.rsi_val = float(rsi_s.iloc[-1])
            except Exception:
                r.rsi_val = 0.0

        vol    = df["Volume"].squeeze().to_numpy(dtype=float)
        avg20v = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1
        r.rvol = round(float(vol[-1]) / avg20v, 2)

        # G3: RS (gate at 30th percentile)
        rsa = RSAgent(
            df, self.data.get("nifty50_data", pd.DataFrame()),
            universe_ranks=self.universe_rs_ranks,
            sector_ranks=self.sector_rs_ranks,   # NEW
            ticker=ticker
        )
        r.rs_score       = min(int(rsa.score() * cfg["rs_weight_mult"]), 20)
        r.rs_percentile  = rsa.get_percentile()
        r.rs_sector_pct  = rsa.get_sector_percentile()   # NEW
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
        risk = RiskAgent(df, r.breakout_level, r.entry_low, r.entry_high,
                         universe=universe)
        if not risk.passes():
            r.rejected = True; r.reject_reason = risk.reject_reason()
            return r
        r.entry = risk.entry; r.stop_loss = risk.stop
        r.target1 = risk.target1; r.target2 = risk.target2
        r.rrr = risk.rrr; r.stop_pct = risk.stop_pct; r.gain_pct_t1 = risk.gain_pct

        # [P0-3] G5: AsymmetryGate — dynamic path with ATR recompute
        ag = AsymmetryGate(entry=r.entry_high, stop=r.stop_loss,
                           target1=r.target1, universe=universe)
        ag_result = ag.check_dynamic(df=df, w4_pct=r.vcp_w4_pct)
        if not ag_result["qualified"] and ag_result["fail_stage"] == "INPUT":
            ag_result = ag.check()
        r.asymmetry_risk_pct   = ag_result["risk_pct"]
        r.asymmetry_reward_pct = ag_result["reward_pct"]
        r.asymmetry_rr         = ag_result["rr_ratio"]
        r.asymmetry_fail_stage = ag_result["fail_stage"]
        if not ag_result["qualified"]:
            r.rejected = True; r.reject_reason = ag_result["fail_reason"]
            return r

        # [P0-1] G6: VCP — hard reject now enforced
        vcpg = VCPContractionGate(df=df)
        vcp  = vcpg.check()
        r.vcp_w4_pct    = vcp["w4_pct"]
        r.vcp_contracting = vcp["contracting"]
        r.vcp_penalty   = vcp["penalty"]
        if vcp["hard_reject"]:
            r.rejected = True; r.reject_reason = vcp["fail_reason"]
            return r

        # G7: Headroom
        if r.entry_high > 0 and r.target1 > r.entry_high:
            r.headroom_pct = round(
                (r.target1 - r.entry_high) / r.entry_high * 100, 2)
        if r.headroom_pct < 4.5:
            r.rejected = True
            r.reject_reason = (
                f"Headroom {r.headroom_pct:.1f}% < 4.5% "
                f"(T1 {r.target1:.0f} vs entry {r.entry_high:.0f})"
            )
            return r

        # Fundamental proxy
        wta = WeeklyTrendAgent(df)
        r.weekly_trend_bonus = wta.score_bonus()
        r.weekly_trend_weeks = wta.get_weeks_available()

        fp = FundamentalProxyAgent(ticker, df, del_pct, r.rs_percentile,
                                   self.sector_ranks.get(sector, 7))
        r.fundamental_score = fp.evaluate()["fundamental_proxy_score"]

        # Institutional proxy
        ip = InstitutionalProxyAgent(ticker, df, del_pct)
        r.institutional_score = ip.evaluate()["institutional_proxy_score"]

        r.bonus_score = min(
            r.liq_score // 2
            + r.fundamental_score // 4
            + r.institutional_score // 4, 5
        )

        # Earnings catalyst
        try:
            from agents.earnings_catalyst_agent import EarningsCatalystAgent
            eca = EarningsCatalystAgent(ticker=ticker, df=df).analyze()
            r.earnings_flag  = eca.get("catalyst_found", False)
            r.earnings_score = eca.get("score", 0)
        except Exception as e:
            log.debug(f"EarningsCatalystAgent {ticker}: {e}")

        # ConfirmationAgent (v5)
        conf = ConfirmationAgent(ticker, r.entry, r.stop_loss, r.breakout_level)
        r.confirmation_state = conf.get_state()
        r.confirmation_score = conf.get_score()

        # Master score
        r.raw_score = (
            r.rs_score       +
            r.pattern_score  +
            r.rsi_score      +
            r.volume_score   +
            r.ema_score      +
            r.market_score   +
            r.macd_score     +
            r.sector_score   +
            r.bonus_score    +
            round(r.weekly_trend_bonus)
        )

        penalty = int(self.regime_penalty * cfg["regime_penalty_mult"])
        r.total_score = max(0, (
            r.raw_score
            + penalty
            + r.earnings_score
            + self.macro_score
            + r.confirmation_score
            - r.vcp_penalty
            - self.event_penalty
        ))

        # Conviction
        ca = ConvictionAgent()
        r.confidence_pct = ca.calibrate_confidence(r.pattern, r.total_score, universe)

        # Tier assignment
        gate           = cfg["score_gate"]
        effective_gate = gate + self.event_penalty
        r.low_edge_pattern = is_low_edge_pattern(r.pattern)

        if r.total_score >= effective_gate and not r.low_edge_pattern:
            r.tier = 1
            r.what_is_working = self._why_working(r)
        elif r.total_score >= effective_gate and r.low_edge_pattern:
            # NEW — score clears the Tier 1 gate, but the pattern itself has
            # near-zero backtested edge (High Base +0.09%, Volume Expansion
            # 0.00% — see PATTERN_EXPECTANCY in pattern_agent.py). Capped at
            # Tier 2 so the email doesn't present this with the same
            # confidence as a genuinely validated setup like High Tight Flag.
            r.tier = 2
            r.what_is_working    = self._why_working(r)
            r.what_is_missing    = [f"Pattern '{r.pattern}' has low historical edge "
                                     f"({PATTERN_EXPECTANCY.get(r.pattern, 0):.2f}% backtested "
                                     f"expectancy) — score cleared on other factors, not pattern strength"]
            r.trigger_conditions = self._triggers(r)
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
            r.rejected     = True
            r.reject_reason = f"Score {r.total_score} below watchlist threshold"
            return r

        r.risk_factors = self._risk_factors(r)
        return r

    # ── universe scan (unchanged from v5.2) ──────────────────────────────────

    def run_universe(self, universe_items: list,
                     stock_data: dict, delivery_data: dict, dry_run: bool = False) -> dict:
        all_results    = []
        reject_reasons = {}

        # P2-01: held stocks get full-detail tracking regardless of whether
        # they clear the gate chain — a rejected StockResult was previously
        # discarded entirely (only its reject_reason string was tallied into
        # the summary counts below). For a held stock, that detail is exactly
        # what a future EXIT/TRIM alert needs, so we keep it here. This does
        # NOT decide ADD-ON/EXIT/TRIM/HOLD yet -- that's P2-02/P2-05/P2-06.
        held_symbols     = set(self.data.get("holdings", {}).keys())
        held_status: List[dict] = []
        universe_tickers = {item[0] for item in universe_items}

        for item in universe_items:
            ticker, name, sector, universe = item
            df = stock_data.get(ticker, pd.DataFrame())
            if df.empty:
                if ticker in held_symbols:
                    held_status.append({"ticker": ticker, "status": "NO_PRICE_DATA", "result": None, "technical_stop": None})
                continue
            try:
                result = self.run(ticker, name, sector, df, delivery_data, universe)
                if ticker in held_symbols:
                    # P2-05 support: compute a real technical stop for this
                    # holding regardless of today's gate outcome -- most
                    # held stocks get REJECTED at "No pattern detected"
                    # before RiskAgent ever runs, so without this an EXIT
                    # alert would have nothing to check a breach against.
                    held_status.append({
                        "ticker": ticker,
                        "status": "REJECTED" if result.rejected else "CLEARED_GATE",
                        "result": result,
                        "technical_stop": compute_holding_stop(df, universe),
                    })
                if not result.rejected:
                    all_results.append(result)
                else:
                    reason = result.reject_reason
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            except Exception as e:
                log.warning(f"{ticker} CRASH: {type(e).__name__}: {e}")
                key = f"Exception: {type(e).__name__}"
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                if ticker in held_symbols:
                    held_status.append({"ticker": ticker, "status": "CRASH", "result": None, "technical_stop": None})

        # P2-01: held stocks with no universe entry at all (e.g. recent IPOs
        # or ETFs not in nse_universe.py) -- flag explicitly rather than let
        # them silently have no status.
        for symbol in held_symbols - universe_tickers:
            held_status.append({"ticker": symbol, "status": "NOT_IN_UNIVERSE", "result": None, "technical_stop": None})

        # P2-05: EXIT check. Rule: effective_stop = max(hard_stop_from_your_
        # actual_avg_price, today's technical_stop) -- same ratchet-only
        # philosophy day5_stop_ratchet.py already applies to trades_v4
        # positions (stop only ever moves up, never loosens), extended here
        # to ALL held stocks using your real avg_price from Portfolio
        # Dashboard, not just scanner-originated trades. hard_stop uses the
        # same per-tier STOP_CAP as everything else in this file, so a
        # holding can never be told to accept a worse loss than the tier's
        # normal risk cap from what you actually paid.
        holdings_data = self.data.get("holdings", {})
        for h in held_status:
            h["exit_check"] = None
            ts = h.get("technical_stop")
            if not ts:
                continue
            hold_info = holdings_data.get(h["ticker"])
            if not hold_info:
                continue
            avg_price   = hold_info.get("avg_price", 0)
            uni         = h["result"].universe if h.get("result") else "LARGE"
            cap         = STOP_CAP.get(uni, 0.03)
            hard_stop   = avg_price * (1 - cap) if avg_price > 0 else 0
            eff_stop    = max(hard_stop, ts["stop"])
            eff_source  = "ENTRY" if hard_stop >= ts["stop"] else "TECHNICAL"
            breached    = ts["current_price"] < eff_stop
            h["exit_check"] = {
                "avg_price":      round(avg_price, 2),
                "hard_stop":      round(hard_stop, 2),
                "effective_stop": round(eff_stop, 2),
                "source":         eff_source,
                "breached":       breached,
            }

        exit_alerts = [h for h in held_status if h.get("exit_check") and h["exit_check"]["breached"]]
        if exit_alerts:
            log.warning(f"  [P2-05] {len(exit_alerts)} EXIT ALERT(S) — price below effective stop:")
            for h in exit_alerts:
                ec = h["exit_check"]
                log.warning(f"    {h['ticker']:<15} price {h['technical_stop']['current_price']:.2f} < "
                            f"stop {ec['effective_stop']:.2f} ({ec['source']}) | "
                            f"avg cost {ec['avg_price']:.2f}")

        # P2-06: TRIM signal -- RS percentile computed INDEPENDENTLY of
        # pattern detection via RSAgent directly (needs only price history +
        # the pre-computed universe rank tables), NOT read from the
        # StockResult's rs_percentile field. That field defaults to 0.0 for
        # any stock rejected before Gate 2 (pattern) -- which was 8 of 11
        # held stocks today -- so reading it directly would misclassify
        # "never computed" as "0th percentile, terrible," which is wrong.
        # EXIT takes priority: a stock already breached is not also flagged
        # TRIM (mutually exclusive branches per the backlog's own decision
        # tree, not stacked). Threshold: RS < 40th percentile -- well below
        # the 70th-percentile "outperforming" bar used elsewhere (P2-02's
        # ADD-ON, _why_working), so TRIM and ADD-ON can never overlap.
        exit_tickers = {h["ticker"] for h in exit_alerts}
        trim_signals = []
        for h in held_status:
            h["trim_check"] = None
            ticker = h["ticker"]
            if ticker in exit_tickers or h["status"] not in ("REJECTED", "CLEARED_GATE"):
                continue
            df = stock_data.get(ticker, pd.DataFrame())
            if df.empty:
                continue
            try:
                rsa = RSAgent(
                    df, self.data.get("nifty50_data", pd.DataFrame()),
                    universe_ranks=self.universe_rs_ranks,
                    sector_ranks=self.sector_rs_ranks,
                    ticker=ticker,
                )
                rs_pct = rsa.get_percentile()
            except Exception as e:
                log.debug(f"  [P2-06] RS computation failed for {ticker}: {e}")
                continue
            h["trim_check"] = {"rs_percentile": rs_pct}
            if rs_pct < 40:
                trim_signals.append({"ticker": ticker, "rs_percentile": rs_pct})

        if trim_signals:
            log.info(f"  [P2-06] {len(trim_signals)} TRIM signal(s) — RS deteriorating (< 40th pct), "
                     f"not yet stopped out:")
            for t in sorted(trim_signals, key=lambda x: x["rs_percentile"]):
                log.info(f"    {t['ticker']:<15} RS {t['rs_percentile']:.0f}th percentile")

        if held_status:
            n_cleared        = sum(1 for h in held_status if h["status"] == "CLEARED_GATE")
            n_rejected        = sum(1 for h in held_status if h["status"] == "REJECTED")
            n_no_data         = sum(1 for h in held_status if h["status"] == "NO_PRICE_DATA")
            n_not_in_universe = sum(1 for h in held_status if h["status"] == "NOT_IN_UNIVERSE")
            n_crash           = sum(1 for h in held_status if h["status"] == "CRASH")
            log.info(f"  [P2-01] Held stocks evaluated: {len(held_status)} total | "
                     f"{n_cleared} cleared gate | {n_rejected} rejected | "
                     f"{n_no_data} no price data | {n_not_in_universe} not in universe"
                     + (f" | {n_crash} crashed" if n_crash else ""))
            for h in sorted(held_status, key=lambda x: x["ticker"]):
                if h["status"] == "NOT_IN_UNIVERSE":
                    log.warning(f"    {h['ticker']:<15} NOT IN UNIVERSE — not in nse_universe.py, cannot evaluate")
                elif h["status"] == "NO_PRICE_DATA":
                    log.warning(f"    {h['ticker']:<15} NO PRICE DATA")
                elif h["status"] == "CRASH":
                    log.warning(f"    {h['ticker']:<15} CRASHED during evaluation — see warning above")
                elif h["status"] == "REJECTED":
                    ts = h.get("technical_stop")
                    stop_note = f" | stop {ts['stop']:.2f} ({ts['stop_pct']:.1f}% via {ts['method']})" if ts else " | stop N/A"
                    log.info(f"    {h['ticker']:<15} rejected — {h['result'].reject_reason}{stop_note}")
                else:
                    ts = h.get("technical_stop")
                    stop_note = f" | stop {ts['stop']:.2f} ({ts['stop_pct']:.1f}% via {ts['method']})" if ts else " | stop N/A"
                    log.info(f"    {h['ticker']:<15} cleared gate — score {h['result'].total_score}, "
                             f"tier {h['result'].tier}, pattern {h['result'].pattern or '(none)'}{stop_note}")

        # P2-02: ADD-ON candidates — held stocks that cleared the gate chain
        # (Tier 1 or 2) AND show a genuine fresh-breakout signature: RVOL and
        # RS thresholds reused from _why_working()'s existing "institutional
        # activity" (RVOL >= 1.5x) and "outperforming" (RS >= 70th) cutoffs,
        # rather than inventing new numbers. Without this explicit flag, a
        # held stock clearing the gate is indistinguishable from any other
        # day's Tier 1/2 pick in the output -- this makes "you already own
        # this, and it just broke out again" visible as its own category.
        # NOTE: sizing (P2-03), blended-cost stop recompute (P2-04 -- do not
        # skip per backlog), and actually alerting on this (P2-07) are all
        # separate, later steps. This only identifies candidates.
        add_on_candidates = [
            h for h in held_status
            if h["status"] == "CLEARED_GATE"
            and h["result"].tier in (1, 2)
            and h["result"].rvol >= 1.5
            and h["result"].rs_percentile >= 70
        ]
        if add_on_candidates:
            log.info(f"  [P2-02] {len(add_on_candidates)} ADD-ON candidate(s) — held stock(s) with a "
                     f"fresh breakout (RVOL>=1.5x, RS>=70th):")
            for h in add_on_candidates:
                r = h["result"]
                log.info(f"    {h['ticker']:<15} Tier {r.tier} | score {r.total_score} | "
                         f"RVOL {r.rvol:.1f}x | RS {r.rs_percentile:.0f}th | pattern {r.pattern or '(none)'} | "
                         f"entry {r.entry:.1f} SL {r.stop_loss:.1f}")

        # Rejection summary
        if reject_reasons:
            buckets: Dict[str, int] = {}
            for reason, count in reject_reasons.items():
                if "AsymmetryGate" in reason and "stop" in reason.lower():
                    key = "AsymmetryGate: stop too wide"
                elif "AsymmetryGate" in reason and "reward" in reason.lower():
                    key = "AsymmetryGate: insufficient headroom"
                elif "AsymmetryGate" in reason:
                    key = "AsymmetryGate: R:R below minimum"
                elif "VCPGate" in reason:
                    key = "VCPGate: W4 too wide (> 8%)"
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
                elif "Circuit" in reason:
                    key = "Circuit limit — stop cannot execute"
                else:
                    key = reason
                buckets[key] = buckets.get(key, 0) + count
            log.info("  Rejection breakdown:")
            for reason, count in sorted(buckets.items(), key=lambda x: -x[1])[:12]:
                log.info(f"    {count:>3} × {reason}")

        all_results.sort(key=lambda x: x.total_score, reverse=True)

        t1_raw = [r for r in all_results if r.tier == 1]
        t2_raw = [r for r in all_results if r.tier == 2]
        t3     = [r for r in all_results if r.tier == 3]

        # [P0-2] Sector concentration gate applied to T1
        t1_sorted = sorted(t1_raw, key=lambda r: (
            0 if r.confirmation_state == "BREAKOUT_CONFIRMED" else 1,
            -r.rs_persistence,
            -r.total_score
        ))
        sector_gate = SectorConcentrationGate(max_per_sector=SECTOR_T1_CAP)
        t1_accepted: List[StockResult] = []
        sector_overflow: List[StockResult] = []

        for r in t1_sorted:
            if sector_gate.can_add(r.sector):
                sector_gate.add(r.sector)
                t1_accepted.append(r)
            else:
                r.tier = 2
                r.reject_reason = f"SECTOR_CAP_{r.sector} (max {SECTOR_T1_CAP})"
                sector_overflow.append(r)
                log.debug(f"  {r.ticker} demoted T1→T2: sector cap ({r.sector})")

        # [P1] T1 cap
        if len(t1_accepted) > self.t1_cap:
            overflow = t1_accepted[self.t1_cap:]
            t1_accepted = t1_accepted[:self.t1_cap]
            for r in overflow:
                r.tier = 2
            sector_overflow = overflow + sector_overflow

        # Merge T2
        t2 = sector_overflow + t2_raw
        t2.sort(key=lambda r: -r.total_score)
        t2 = t2[:T2_CAP]

        log.info(
            f"  Tier 1 (Top Picks):  {len(t1_accepted)}"
            f" (t1_cap={self.t1_cap}, sector_cap={SECTOR_T1_CAP})"
            f" | sector dist: {sector_gate.summary()}"
        )
        log.info(f"  Tier 2 (Aggressive): {len(t2)} (cap={T2_CAP})")
        log.info(f"  Tier 3 (Watchlist):  {len(t3)}")
        log.info(
            f"  Regime: {self.regime} [conf={self.regime_confidence}] "
            f"| Macro: {self.macro_state} | Event: {self.event_state}"
        )

        # v5.3: Log confidence warning in summary if LOW
        if self.regime_confidence == "LOW":
            log.warning(
                f"  ⚠️  REGIME LOW_CONFIDENCE — {len(self.regime_sanity_flags)} "
                f"contradiction(s). Penalty dampened. Check breadth_calibration.log."
            )

        # Auto-log T1 picks
        if t1_accepted:
            try:
                from trade_logger import auto_log_t1_picks
                logged = auto_log_t1_picks(t1_accepted, regime=self.regime)
                log.info(f"  Auto-logged {logged} T1 picks to trades_v4")
            except Exception as e:
                log.warning(f"  Auto-log failed: {e}")

        # Publish today's T1+T2 picks to Turso (P1-03) — Portfolio Dashboard bridge.
        # Same failure-isolation pattern as auto_log_t1_picks above: never
        # allowed to break the scan or block the evening email (P1-06).
        # Skipped entirely on --dry-run, same as the email send below in
        # scanner.py — a test run must never write test data into the shared
        # production Turso table Portfolio Dashboard reads from.
        if dry_run:
            log.info("  [DRY RUN] Skipping Turso publish (would publish "
                      f"{len(t1_accepted) + len(t2)} signals)")
        else:
            try:
                from turso_sync import publish_signals
                published = publish_signals(t1_accepted + t2, regime=self.regime)
                log.info(f"  Published {published} signals to Turso (scanner_signals)")
            except Exception as e:
                log.warning(f"  Turso publish failed (non-fatal): {e}")

        # Near-breakout watchlist
        existing      = {r.ticker for r in all_results}
        bhavcopy_cmp_map = self.data.get("bhavcopy_cmp_map", {})
        near_breakout = find_near_breakout_stocks(
            universe_items, stock_data, delivery_data, existing,
            bhavcopy_cmp_map=bhavcopy_cmp_map,
        )
        log.info(f"  Near-breakout watchlist: {len(near_breakout)} stocks")

        # Defensive / relative-strength watchlist — no-op unless regime is D or E.
        # Reuses self.universe_rs_ranks / self.sector_rs_ranks computed in
        # __init__ above; does not recompute RS. See agents/defensive_agent.py.
        defensive_watchlist = run_defensive_scan(
            self, universe_items, stock_data,
            self.data.get("nifty50_data", pd.DataFrame()),
            delivery_data,
        )

        # Save picks_latest.json
        from datetime import date as _date
        picks_json = []
        for r in t1_accepted + t2:
            picks_json.append({
                "ticker":           r.ticker.replace(".NS", ""),
                "ticker_raw":       r.ticker,
                "name":             r.name,
                "sector":           r.sector,
                "security_id":      "",
                "segment":          "NSE_EQ",
                "entry":            round(r.entry,           2),
                "sl":               round(r.stop_loss,       2),
                "pivot":            round(r.breakout_level if r.breakout_level > 0
                                          else r.entry,      2),
                "t1":               round(r.target1,         2),
                "t2":               round(r.target2,         2),
                # [FIX] Was r.asymmetry_rr — AsymmetryGate's own R:R, computed
                # against entry=r.entry_high (top of the entry zone). Using
                # r.rrr (Entry-anchored) makes this match emailer.py's evening
                # card and the 10am confirmation email consistently.
                "rr":               round(r.rrr,             1),
                "score":            r.total_score,
                "tier":             r.tier,
                "pattern":          r.pattern or "",
                "vcp_w4_pct":       round(r.vcp_w4_pct,     2),
                "scan_date":        _date.today().isoformat(),
                "regime":           self.regime,               # v5.4: was missing entirely --
                                                                  # confirm_picks.py's regime-aware
                                                                  # chase tolerance had nothing to read
                "regime_confidence": r.regime_confidence,      # v5.3: in JSON output
                "breadth_source":   r.breadth_source,          # v5.3: audit trail
            })
        with open("picks_latest.json", "w") as f:
            json.dump(picks_json, f, indent=2)
        log.info(
            f"  Saved {len(picks_json)} picks to picks_latest.json "
            f"(T1={len(t1_accepted)}, T2={len(t2)})"
        )

        # Defensive watchlist saved SEPARATELY — never merged into
        # picks_latest.json, which confirm_picks.py parses expecting
        # entry/sl/pivot/t1/t2/rr fields a defensive pick doesn't have.
        with open("defensive_watchlist.json", "w") as f:
            json.dump(defensive_watchlist, f, indent=2)
        log.info(
            f"  Saved {len(defensive_watchlist)} defensive candidates to "
            f"defensive_watchlist.json"
        )

        return {
            "tier1":              t1_accepted,
            "tier2":              t2,
            "tier3":              t3,
            "near_breakout":      near_breakout,
            "defensive_watchlist": defensive_watchlist,
            "all_results":        all_results,
            "held_status":        held_status,
            "add_on_candidates":  add_on_candidates,
            "exit_alerts":        exit_alerts,
            "trim_signals":       trim_signals,
            "regime":             self.regime,
            "regime_name":        self.regime_name,
            "regime_confidence":  self.regime_confidence,
            "regime_sanity_flags": self.regime_sanity_flags,
            "breadth":            self.breadth_score,
            "breadth_detail":     self.breadth_detail,
            "breadth_source":     self.breadth_source,
            "macro_state":        self.macro_state,
            "event_risk":         self.event_state,
            "t1_cap":             self.t1_cap,
            "t2_cap":             T2_CAP,
            "sector_distribution": sector_gate.summary(),
            "dhan_status":        data_fetcher.get_dhan_status(),
        }

    # ── narrative helpers (unchanged from v5.2) ──────────────────────────────

    def _why_working(self, r: StockResult) -> List[str]:
        reasons = []
        if r.rs_percentile >= 70:
            reasons.append(f"RS Rank {r.rs_percentile:.0f}th - outperforming {r.rs_percentile:.0f}% of market")
        if r.pattern:
            bq = f" [{r.breakout_quality}]" if r.breakout_quality else ""
            reasons.append(f"{r.pattern}{bq} - entry Rs.{r.entry_low:.1f}-{r.entry_high:.1f}")
        if r.rvol >= 1.5:
            reasons.append(f"Volume {r.rvol:.1f}x avg (EOD) - institutional activity")
        if r.del_pct >= 50:
            reasons.append(f"Delivery {r.del_pct:.0f}% - holders not selling")
        if r.sector_score >= 5:
            top = self.sector_agent.get_top_sectors(3)
            if r.sector in [s for s, _ in top]:
                reasons.append(f"Sector leadership - {r.sector} top-3 in rotation")
        if r.rsi_val > 0:
            reasons.append(f"RSI(D) {r.rsi_val:.0f} - momentum constructive")
        if r.earnings_flag:
            reasons.append("Earnings acceleration within 14 days - catalyst-backed")
        if r.confirmation_state == "BREAKOUT_CONFIRMED":
            reasons.append("Breakout confirmed - held above pivot 1+ sessions")
        if r.rs_persistence >= 8:
            reasons.append(f"RS persistence {r.rs_persistence}/13 weeks - sustained leader")
        if r.weekly_trend_weeks >= 30 and r.weekly_trend_bonus >= 1.5:
            reasons.append(f"Weekly trend strongly aligned (+{r.weekly_trend_bonus:.1f}) - "
                            f"multi-timeframe confirmation")
        return reasons[:4]

    def _what_missing(self, r: StockResult, gate: int) -> List[str]:
        missing = []
        gap = gate - r.total_score
        missing.append(f"Score {r.total_score} - needs {gap} more pts for Tier 1")
        if r.rs_percentile < 60:
            missing.append(f"RS {r.rs_percentile:.0f}th pct - stronger RS needed")
        if r.rvol < 1.3:
            missing.append(f"Volume {r.rvol:.1f}x (EOD) - needs breakout volume >= 1.5x")
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
        if r.circuit_limit:
            risks.append(f"Circuit limit {r.circuit_limit} detected at scan time")
        if r.regime_confidence == "LOW":
            risks.append("Regime LOW_CONFIDENCE — verify breadth data before acting")
        if r.weekly_trend_weeks >= 30 and r.weekly_trend_bonus <= -1.0:
            risks.append(f"Weekly trend not aligned ({r.weekly_trend_bonus:.1f}) - "
                         f"daily breakout against the larger trend")
        return risks[:3]
