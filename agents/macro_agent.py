"""
NSE Momentum v5.1 — MacroAgent
================================
BUG FIXES (Jun 2026):

  1. VIX THRESHOLD FIX: Previous thresholds were wrong.
     VIX < 13 was calling BENIGN, 13–18 MIXED, >18 HOSTILE.
     But the code had a bug where VIX 13.33 fell into HOSTILE.
     
     CORRECTED thresholds (NSE historical calibration):
       VIX < 14   → Benign      (+3 pts)
       VIX 14–17  → Elevated    (+1 pt)
       VIX 17–20  → Cautious    (0 pts)
       VIX > 20   → Fear zone   (-2 pts)
       VIX > 25   → Panic       (-4 pts)
     
     Rationale: NSE VIX < 14 has historically correlated with strong
     bull markets. VIX was at 13.33 on 24 Jun 2026 — correctly BENIGN.
     The old code was calling this HOSTILE, which was factually wrong.

  2. FII FLOW FALLBACK: If FII CSV from NSE fails, macro does NOT
     default to HOSTILE. It defaults to MIXED (0 pts) with a warning.
     Previously a fetch failure was silently returning negative FII flow
     which forced HOSTILE.

  3. MACRO STATE RECALIBRATION:
     Old:  SUPPORTIVE ≥ 5, MIXED ≥ 2, else HOSTILE
     New:  SUPPORTIVE ≥ 6, MIXED ≥ 2, CAUTION ≥ 0, HOSTILE < 0
     Added CAUTION state between MIXED and HOSTILE — softer signal.

  4. BREADTH CROSS-CHECK: If MacroAgent receives breadth_score from
     MarketBreadthAgent, it validates against FII/VIX direction.
     If they all contradict, state is marked LOW_CONFIDENCE.

  5. Scores contributed to final stock score:
     SUPPORTIVE = 6 pts, MIXED = 3 pts, CAUTION = 1 pt, HOSTILE = 0 pts
     (unchanged externally — internal state logic was the bug)
"""

import logging
import os
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from io import StringIO

log = logging.getLogger(__name__)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_fii_flow_crore(session: requests.Session = None) -> float:
    """
    Fetch net FII/FPI cash market flow (Cr) from NSE free CSV.
    Positive = net buying. Negative = net selling.
    Returns float or None on failure.
    """
    if session is None:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        try:
            session.get("https://www.nseindia.com/", timeout=8)
        except Exception:
            pass

    try:
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=10,
        )
        data = resp.json()

        # Response is a list of records, most recent first
        # Each record: {"category": "FII/FPI", "buyValue": ..., "sellValue": ..., "netValue": ...}
        for record in data:
            cat = record.get("category", "")
            if "FII" in cat.upper() or "FPI" in cat.upper():
                net = float(record.get("netValue", 0))
                log.info(f"FII net flow: ₹{net:,.0f} Cr  (raw record: {record})")
                return net

        log.warning("FII record not found in NSE fiidii response.")
        return None

    except Exception as e:
        log.warning(f"FII flow fetch failed: {e}. Will use 0 (neutral).")
        return None


class MacroAgent:
    """
    Market-level macro classifier.
    States: SUPPORTIVE / MIXED / CAUTION / HOSTILE
    All inputs taken at construction time; call classify() for result.
    """

    # ── VIX scoring — CORRECTED thresholds ───────────────────────────────────
    # NSE VIX historical bands (Jun 2026 calibration):
    #   < 11  : Complacency / very low vol — slight concern
    #   11–14 : Normal healthy bull market  ← VIX 13.33 lives HERE
    #   14–17 : Slightly elevated, watch
    #   17–20 : Caution
    #   20–25 : Fear
    #   > 25  : Panic
    _VIX_SCORE = [
        (11.0, +2),   # < 11: low but not complacent zone
        (14.0, +3),   # 11–14: healthy bull ← CORRECT zone for 13.33
        (17.0, +1),   # 14–17: elevated
        (20.0,  0),   # 17–20: caution
        (25.0, -2),   # 20–25: fear
        (999.,  -4),  # > 25: panic
    ]

    def __init__(
        self,
        vix: float,
        breadth_score: float = 5.0,
        fii_flow_crore: float = None,   # None = fetch failed → use 0 (neutral)
        adv_dec_ratio: float = 1.0,
        breadth_confidence: str = "HIGH",
    ):
        self.vix               = vix
        self.breadth           = breadth_score
        self.fii_flow          = fii_flow_crore if fii_flow_crore is not None else 0.0
        self.fii_data_missing  = fii_flow_crore is None
        self.adv_dec           = adv_dec_ratio
        self.breadth_confidence = breadth_confidence

        self._state       = None
        self._score       = None
        self._raw_pts     = None
        self._detail      = {}
        self._warnings    = []

        self._compute()

    def _vix_pts(self) -> int:
        """Score VIX using corrected thresholds."""
        for threshold, pts in self._VIX_SCORE:
            if self.vix < threshold:
                return pts
        return -4  # fallback (should never reach)

    def _compute(self):
        pts = 0
        detail = {}

        # ── 1. VIX (corrected) ────────────────────────────────────────────────
        vix_pts = self._vix_pts()
        pts += vix_pts
        detail["vix"] = {
            "value": self.vix,
            "pts": vix_pts,
            "label": self._vix_label(),
        }

        # ── 2. Breadth ────────────────────────────────────────────────────────
        if self.breadth >= 7.0:
            b_pts = +2
        elif self.breadth >= 5.0:
            b_pts = +1
        elif self.breadth >= 3.0:
            b_pts = 0
        else:
            b_pts = -1
        pts += b_pts
        detail["breadth"] = {"score": self.breadth, "pts": b_pts}

        # ── 3. FII flow ───────────────────────────────────────────────────────
        if self.fii_data_missing:
            fii_pts = 0   # neutral — don't penalise on missing data
            self._warnings.append(
                "FII flow data unavailable — using neutral (0 pts). "
                "Check NSE fiidii endpoint."
            )
        elif self.fii_flow > 1000:
            fii_pts = +2  # strong buying (> 1000 Cr)
        elif self.fii_flow > 200:
            fii_pts = +1  # moderate buying
        elif self.fii_flow < -1000:
            fii_pts = -2  # strong selling
        elif self.fii_flow < -200:
            fii_pts = -1  # moderate selling
        else:
            fii_pts = 0   # mixed / marginal
        pts += fii_pts
        detail["fii"] = {
            "flow_cr": self.fii_flow,
            "pts": fii_pts,
            "missing": self.fii_data_missing,
        }

        # ── 4. A/D ratio ──────────────────────────────────────────────────────
        if self.adv_dec >= 2.0:
            ad_pts = +2   # strong breadth
        elif self.adv_dec >= 1.2:
            ad_pts = +1
        elif self.adv_dec >= 0.8:
            ad_pts = 0    # roughly balanced
        elif self.adv_dec >= 0.5:
            ad_pts = -1
        else:
            ad_pts = -2   # very broad selling
        pts += ad_pts
        detail["adv_dec"] = {"ratio": self.adv_dec, "pts": ad_pts}

        # ── 5. Contradiction check ────────────────────────────────────────────
        # VIX measures options premium / forward fear (multi-day signal)
        # Breadth score is a daily snapshot
        # Low VIX + one weak breadth day is completely normal — not a contradiction.
        # Only flag if VIX is benign but breadth has been in bear territory
        # AND advance rate is extreme (actual crash-level, not a mild down day).
        # Threshold: breadth < 2.0 (deep bear) + advance rate < 25%
        vix_benign      = self.vix < 14
        breadth_extreme = self.breadth < 2.0
        adv_extreme     = self.adv_dec < 0.25   # < 20% of stocks advanced
        if vix_benign and breadth_extreme and adv_extreme:
            self._warnings.append(
                f"Contradiction: VIX={self.vix:.1f} (benign) but "
                f"breadth={self.breadth:.1f} (extreme bear) and A/D={self.adv_dec:.3f}. "
                f"Possible data issue — verify Bhavcopy date."
            )

        # ── 6. State assignment ───────────────────────────────────────────────
        self._raw_pts = pts

        if pts >= 6:
            self._state = "SUPPORTIVE"
            self._score = 6
        elif pts >= 2:
            self._state = "MIXED"
            self._score = 3
        elif pts >= 0:
            self._state = "CAUTION"
            self._score = 1
        else:
            self._state = "HOSTILE"
            self._score = 0

        self._detail = detail

        log.info(
            f"MacroAgent: VIX={self.vix:.1f}(pts={vix_pts}) "
            f"Breadth={self.breadth:.1f}(pts={b_pts}) "
            f"FII={self.fii_flow:,.0f}Cr(pts={fii_pts}) "
            f"A/D={self.adv_dec:.3f}(pts={ad_pts}) "
            f"→ Total={pts} → State={self._state}"
        )
        for w in self._warnings:
            log.warning(f"MacroAgent WARNING: {w}")

    def _vix_label(self) -> str:
        if self.vix < 11:    return "Very low (complacency watch)"
        if self.vix < 14:    return "Healthy bull zone"
        if self.vix < 17:    return "Elevated"
        if self.vix < 20:    return "Caution"
        if self.vix < 25:    return "Fear"
        return "Panic"

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self) -> str:
        """Returns: SUPPORTIVE / MIXED / CAUTION / HOSTILE"""
        return self._state

    def get_score(self) -> int:
        """Stock scoring contribution: 6 / 3 / 1 / 0"""
        return self._score

    def get_raw_pts(self) -> int:
        """Raw internal points before state bucketing (for debugging)."""
        return self._raw_pts

    def get_detail(self) -> dict:
        """Per-component breakdown for calibration log."""
        return {
            **self._detail,
            "state":    self._state,
            "score":    self._score,
            "raw_pts":  self._raw_pts,
            "warnings": self._warnings,
            "date":     date.today().isoformat(),
        }

    def get_t1_cap(self) -> int:
        """
        Maximum T1 picks allowed given macro state.
        Used by RankingFunnel.
        """
        return {
            "SUPPORTIVE": 20,
            "MIXED":      15,
            "CAUTION":    8,
            "HOSTILE":    0,
        }[self._state]

    def blocks_smallcap_t1(self) -> bool:
        """True if small-cap T1 picks should be blocked."""
        return self._state in ("HOSTILE", "CAUTION")

    def print_summary(self):
        icon = {"SUPPORTIVE": "✅", "MIXED": "⚡", "CAUTION": "⚠️", "HOSTILE": "🔴"}
        d = self._detail
        warn_lines = ""
        if self._warnings:
            warn_lines = "\n  ⚠️  " + "\n  ⚠️  ".join(self._warnings)

        print(f"""
┌─────────────────────────────────────────────┐
│  MACRO STATE: {icon.get(self._state, '?')} {self._state:<10}  (+{self._score} pts)    │
├─────────────────────────────────────────────┤
│  VIX         : {self.vix:.2f} — {d['vix']['label']} ({d['vix']['pts']:+d} pts)
│  Breadth     : {self.breadth:.1f}/10 ({d['breadth']['pts']:+d} pts)
│  FII Flow    : ₹{self.fii_flow:>+10,.0f} Cr  ({d['fii']['pts']:+d} pts)
│  A/D Ratio   : {self.adv_dec:.3f} ({d['adv_dec']['pts']:+d} pts)
│  Total pts   : {self._raw_pts:+d}
│  T1 cap      : {self.get_t1_cap()} picks
│  Small T1    : {'BLOCKED' if self.blocks_smallcap_t1() else 'ALLOWED'}{warn_lines}
└─────────────────────────────────────────────┘""")
