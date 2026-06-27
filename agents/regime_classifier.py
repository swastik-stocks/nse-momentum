"""
NSE Momentum v5.1 — RegimeClassifier
=======================================
BUG FIXES (Jun 2026):

  1. CONFIDENCE SCORE: Regime now carries a confidence field.
     When individual signals contradict each other, confidence = LOW.
     LOW_CONFIDENCE regimes DO NOT apply the full penalty.
     
     Penalty dampening on LOW_CONFIDENCE:
       Normal Regime D penalty : -12 pts
       LOW_CONFIDENCE D penalty:  -5 pts (41% of normal)
       → System stays cautious but does NOT zero out all setups.

  2. SANITY CHECKS (4 rules):
     R1: above_50_ema > 60% AND ad_ratio < 0.15   → CRASH-LEVEL → LOW_CONF
     R2: above_50_ema > 60% AND regime in (D, E)  → FLAG → may recheck
     R3: VIX < 14 AND regime == E                 → CONTRADICT → LOW_CONF
     R4: breadth_score > 7 AND regime in (D, E)   → CONTRADICT → LOW_CONF

  3. T1 CAP LOGIC: With LOW_CONFIDENCE, T1 cap is not 0 (which forces
     "stay in cash" even when market is actually constructive). Instead:
       HIGH_CONFIDENCE Regime D: T1 cap = 0  (as before)
       LOW_CONFIDENCE Regime D:  T1 cap = 5  (reduced but not zero)

  4. Raw input dump: every classify() call logs all inputs + result
     to logs/regime_calibration.log for audit.
"""

import logging
import json
import os
from datetime import date, datetime

log = logging.getLogger(__name__)

_CAL_LOG = os.path.join(os.path.dirname(__file__), '..', 'logs', 'regime_calibration.log')

# ── Regime definitions ────────────────────────────────────────────────────────
# (label, score, full_penalty, description, t1_cap_high_conf, t1_cap_low_conf)
REGIMES = {
    "A": ("Strong Bull",   5,   0, "Full conviction. All systems go.",          25, 20),
    "B": ("Bull",          4,   0, "Good conditions. Standard scoring.",         20, 15),
    "C": ("Range Bound",   2,  -5, "Choppy. Tighten filters. Reduce size.",     12,  8),
    "D": ("Correction",    1, -12, "Pullback. Watchlist only. Heavy penalty.",    0,  5),
    "E": ("Bear Market",   0, -20, "Stay in cash. All setups suppressed.",        0,  0),
}

# LOW_CONFIDENCE dampened penalties (< 50% of full penalty)
_LOW_CONF_PENALTY = {
    "A":  0,
    "B":  0,
    "C": -2,
    "D": -5,
    "E": -8,   # Even bear market penalty is dampened on low-confidence data
}


class RegimeClassifier:
    """
    5-regime classifier with confidence scoring and sanity checks.

    Inputs:
        nifty50_data     : pd.DataFrame with 'Close' column, min 200 rows
        banknifty_data   : pd.DataFrame with 'Close' column, min 50 rows
        vix              : float — India VIX current value
        ad_ratio         : float — NSE-wide A/D ratio (from MarketBreadthAgent)
        breadth_score    : float 0–10 (from MarketBreadthAgent)
        above_50_ema_pct : float — % of our universe above 50 EMA
        breadth_confidence: str  — "HIGH" or "LOW" (from MarketBreadthAgent)
        macro_state      : str   — SUPPORTIVE/MIXED/CAUTION/HOSTILE
    """

    def __init__(
        self,
        nifty50_data     = None,
        banknifty_data   = None,
        vix: float       = 15.0,
        ad_ratio: float  = 1.0,
        breadth_score: float = 5.0,
        above_50_ema_pct: float = 50.0,
        breadth_confidence: str = "HIGH",
        macro_state: str = "MIXED",
    ):
        import pandas as pd
        self.nifty50     = nifty50_data   if nifty50_data   is not None else pd.DataFrame()
        self.banknifty   = banknifty_data if banknifty_data is not None else pd.DataFrame()
        self.vix         = vix
        self.ad_ratio    = ad_ratio
        self.breadth     = breadth_score
        self.above_50    = above_50_ema_pct
        self.breadth_conf = breadth_confidence
        self.macro_state = macro_state

        self._regime     = None
        self._pts        = None
        self._confidence = "HIGH"
        self._sanity_flags = []
        self._detail     = {}

    def classify(self) -> dict:
        """
        Run classification. Returns full result dict.
        Idempotent — calling twice returns same result.
        """
        if self._regime is not None:
            return self._build_result()

        pts = 0
        detail = {}

        # ── 1. Nifty 50 EMA alignment (0–4 pts) ──────────────────────────────
        nifty_pts, nifty_detail = self._score_nifty50()
        pts += nifty_pts
        detail["nifty50"] = nifty_detail

        # ── 2. Bank Nifty alignment (0–2 pts) ────────────────────────────────
        bn_pts, bn_detail = self._score_banknifty()
        pts += bn_pts
        detail["banknifty"] = bn_detail

        # ── 3. VIX (0–2 pts) ─────────────────────────────────────────────────
        vix_pts = 0
        if self.vix < 13:
            vix_pts = 2
        elif self.vix < 16:
            vix_pts = 1
        elif self.vix < 20:
            vix_pts = 0
        else:
            vix_pts = -1
        pts += vix_pts
        detail["vix"] = {"value": self.vix, "pts": vix_pts}

        # ── 4. A/D ratio (0–2 pts) ───────────────────────────────────────────
        ad_pts = 0
        if self.ad_ratio >= 1.5:
            ad_pts = 2
        elif self.ad_ratio >= 1.0:
            ad_pts = 1
        elif self.ad_ratio >= 0.7:
            ad_pts = 0
        else:
            ad_pts = -1
        pts += ad_pts
        detail["ad_ratio"] = {"value": self.ad_ratio, "pts": ad_pts}

        # ── 5. Breadth score injection (−2 to +2) ────────────────────────────
        b_adj = 0
        if self.breadth >= 8:
            b_adj = +2
        elif self.breadth >= 6:
            b_adj = +1
        elif self.breadth <= 2:
            b_adj = -2
        elif self.breadth <= 3:
            b_adj = -1
        pts += b_adj
        detail["breadth"] = {"score": self.breadth, "adj": b_adj}

        # ── 6. Raw regime from points ─────────────────────────────────────────
        if pts >= 9:
            raw_regime = "A"
        elif pts >= 7:
            raw_regime = "B"
        elif pts >= 4:
            raw_regime = "C"
        elif pts >= 2:
            raw_regime = "D"
        else:
            raw_regime = "E"

        self._pts = pts
        self._regime = raw_regime
        self._detail = detail

        # ── 7. Sanity checks ──────────────────────────────────────────────────
        self._run_sanity_checks()

        # ── 8. Log calibration ────────────────────────────────────────────────
        self._log_calibration()

        return self._build_result()

    def _score_nifty50(self) -> tuple:
        """Score Nifty 50 EMA position. Returns (pts, detail_dict)."""
        if self.nifty50.empty or len(self.nifty50) < 200:
            log.warning("Nifty 50 data insufficient for EMA scoring.")
            return 1, {"pts": 1, "note": "insufficient data — using neutral"}

        close = self.nifty50["Close"].squeeze()
        c     = float(close.iloc[-1])
        e20   = float(close.ewm(span=20).mean().iloc[-1])
        e50   = float(close.ewm(span=50).mean().iloc[-1])
        e200  = float(close.ewm(span=200).mean().iloc[-1])

        # Higher-high / higher-low check over last 20 sessions
        recent  = close.tail(20)
        hh      = float(recent.max())
        ll      = float(recent.min())
        hl_ok   = c > (hh + ll) / 2

        if c > e20 > e50 > e200 and hl_ok:
            pts, label = 4, "Full bull stack (price > EMA20 > 50 > 200 + HH/HL)"
        elif c > e50 > e200:
            pts, label = 3, "Above EMA50 & EMA200"
        elif c > e50:
            pts, label = 2, "Above EMA50 only"
        elif c > e200:
            pts, label = 1, "Above EMA200 only"
        else:
            pts, label = 0, "Below EMA200 — bearish structure"

        return pts, {
            "close": round(c, 1), "e20": round(e20, 1),
            "e50": round(e50, 1), "e200": round(e200, 1),
            "hl_ok": hl_ok, "pts": pts, "label": label,
        }

    def _score_banknifty(self) -> tuple:
        """Score Bank Nifty EMA position. Returns (pts, detail_dict)."""
        if self.banknifty.empty or len(self.banknifty) < 50:
            return 1, {"pts": 1, "note": "insufficient data — using neutral"}

        close = self.banknifty["Close"].squeeze()
        c     = float(close.iloc[-1])
        e50   = float(close.ewm(span=50).mean().iloc[-1])
        e200  = float(close.ewm(span=200).mean().iloc[-1])

        if c > e50 > e200:
            pts, label = 2, "Above EMA50 > EMA200"
        elif c > e50:
            pts, label = 1, "Above EMA50"
        else:
            pts, label = 0, "Below EMA50"

        return pts, {
            "close": round(c, 1), "e50": round(e50, 1),
            "e200": round(e200, 1), "pts": pts, "label": label,
        }

    def _run_sanity_checks(self):
        """
        Run 4 contradiction rules. Sets _confidence = "LOW" if any fire.
        """
        flags = []

        # ── IMPORTANT: above_50_ema and daily A/D measure DIFFERENT timeframes ──
        # above_50_ema = structural (weeks/months). A/D = daily (one session).
        # A market with 63% of stocks above 50EMA CAN have A/D=0.34 on a down day.
        # Only flag EXTREME divergences suggesting actual data corruption.
        # ──────────────────────────────────────────────────────────────────────

        # R1: Only flag crash-level daily decline vs bullish structure
        # (A/D < 0.15 = >85% stocks down = crash level, not a normal pullback)
        if self.above_50 > 60 and self.ad_ratio < 0.15:
            flags.append(
                f"R1: above_50_ema={self.above_50:.1f}% but ad_ratio={self.ad_ratio:.3f} "
                f"— crash-level decline in bull structure. Likely Bhavcopy date mismatch."
            )

        # R2: Uptrend majority but regime is bearish
        # FIXED: Only flag when BOTH structural (above_50_ema) AND daily (ad_ratio)
        # signal bull but the index says bear. If today's A/D is weak (<= 1.0),
        # regime D is consistent — stocks can be in uptrends while the index corrects
        # on a down day. That is normal market behaviour, not a data error.
        if self.above_50 > 60 and self._regime in ("D", "E") and self.ad_ratio > 1.0:
            flags.append(
                f"R2: above_50_ema={self.above_50:.1f}% AND ad_ratio={self.ad_ratio:.3f} "
                f"(both bullish) but regime={self._regime}. "
                f"Both structural and daily signals contradict the index regime."
            )
        elif self.above_50 > 60 and self._regime in ("D", "E") and self.ad_ratio <= 1.0:
            log.info(
                f"R2 note: above_50_ema={self.above_50:.1f}% but regime={self._regime} "
                f"with ad_ratio={self.ad_ratio:.3f} (down day). "
                f"Structural uptrend + index correction + weak breadth — consistent, not a contradiction."
            )

        # R3: VIX benign but regime is full bear
        if self.vix < 14 and self._regime == "E":
            flags.append(
                f"R3: VIX={self.vix:.1f} (benign, <14) but regime=E (Bear Market). "
                f"Bear market regimes historically don't occur with VIX < 14."
            )

        # R4: Strong breadth but regime is D/E
        if self.breadth > 7 and self._regime in ("D", "E"):
            flags.append(
                f"R4: breadth_score={self.breadth:.1f} (>7, bullish) "
                f"but regime={self._regime}. Breadth and regime contradict."
            )

        self._sanity_flags = flags

        if flags:
            self._confidence = "LOW"
            for f in flags:
                log.warning(f"REGIME SANITY: {f}")
        else:
            self._confidence = "HIGH"

    def _build_result(self) -> dict:
        """Build and return the full result dict."""
        r = self._regime
        label, score, full_penalty, desc, t1_high, t1_low = REGIMES[r]

        if self._confidence == "LOW":
            penalty = _LOW_CONF_PENALTY[r]
            t1_cap  = t1_low
            conf_note = (
                "LOW CONFIDENCE — penalty dampened. "
                "Individual signals contradict. "
                f"Full penalty would be {full_penalty} pts. "
                f"Applied: {penalty} pts."
            )
        else:
            penalty = full_penalty
            t1_cap  = t1_high
            conf_note = "HIGH CONFIDENCE — all signals consistent."

        return {
            "date":          date.today().isoformat(),
            "regime":        r,
            "regime_label":  label,
            "regime_desc":   desc,
            "score":         score,
            "penalty":       penalty,
            "full_penalty":  full_penalty,
            "t1_cap":        t1_cap,
            "confidence":    self._confidence,
            "confidence_note": conf_note,
            "raw_pts":       self._pts,
            "sanity_flags":  self._sanity_flags,
            "macro_state":   self.macro_state,
            "inputs": {
                "vix":            self.vix,
                "ad_ratio":       self.ad_ratio,
                "breadth_score":  self.breadth,
                "above_50_ema":   self.above_50,
                "macro_state":    self.macro_state,
                "breadth_conf":   self.breadth_conf,
            },
            "component_detail": self._detail,
        }

    def _log_calibration(self):
        """Append raw inputs + result to audit log."""
        try:
            os.makedirs(os.path.dirname(_CAL_LOG), exist_ok=True)
            entry = {
                "ts":           datetime.now().isoformat(),
                "date":         date.today().isoformat(),
                "regime":       self._regime,
                "raw_pts":      self._pts,
                "confidence":   self._confidence,
                "vix":          self.vix,
                "ad_ratio":     self.ad_ratio,
                "breadth":      self.breadth,
                "above_50_ema": self.above_50,
                "macro_state":  self.macro_state,
                "sanity_flags": self._sanity_flags,
            }
            with open(_CAL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.warning(f"Could not write regime calibration log: {e}")

    def print_summary(self):
        """Pretty-print regime result to stdout."""
        result = self._build_result()
        r = result["regime"]
        icon = {"A": "🟢", "B": "🟩", "C": "🟡", "D": "🟠", "E": "🔴"}.get(r, "⚪")
        conf_icon = "✅" if result["confidence"] == "HIGH" else "⚠️"
        flags_str = ""
        if result["sanity_flags"]:
            flags_str = "\n  🚩 " + "\n  🚩 ".join(result["sanity_flags"])

        print(f"""
╔══════════════════════════════════════════════════╗
║  REGIME: {icon} {r} — {result['regime_label']:<25}     ║
╠══════════════════════════════════════════════════╣
║  {result['regime_desc']}
║  Raw score pts : {result['raw_pts']:+d}
║  Penalty       : {result['penalty']} pts  (full={result['full_penalty']})
║  T1 cap        : {result['t1_cap']} picks
║  Confidence    : {conf_icon} {result['confidence']}
║  Macro state   : {result['macro_state']}{flags_str}
╚══════════════════════════════════════════════════╝""")
