"""
NSE Momentum v5.2 — AsymmetryGate (complete replacement)

WHAT CHANGED vs v5.1:
  - Hard fixed-% thresholds replaced with ATR-based dynamic stop sizing
  - Stop is now computed FROM the dataframe (EMA21 floor + ATR), not just
    accepted as-is from RiskAgent. This is the root cause of 4-7% stops
    sneaking through: RiskAgent was placing the stop, AsymmetryGate was only
    checking it, not recomputing it.
  - Profile system: TIGHT_VCP / STANDARD / WIDE_SWING keyed on VCP W4 width
  - Universe multipliers preserved (LARGE / MID / SMALL still have different
    absolute caps) but the floor logic is ATR-driven
  - Required reward scales with actual stop so R:R guarantee always holds
  - Absolute stop cap per universe — nothing passes that exceeds this
  - All existing callers in orchestrator.py remain unchanged:
      ag = AsymmetryGate(entry=r.entry_high, stop=r.stop_loss,
                         target1=r.target1, universe=universe)
      ag_result = ag.check()
    The .check() signature is identical; new .check_dynamic() is available
    when df + w4_pct are passed for the full recompute path.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE CAPS  — absolute maximum stop we will ever accept per universe tier
# These are wider than the old hard rule because mid/small ATR justifies it,
# but they are still HARD — nothing above these passes under any profile.
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_CAPS = {
    "LARGE": {"abs_stop_cap": 4.0, "abs_reward_floor": 8.0,  "min_rr": 2.0},
    "MID":   {"abs_stop_cap": 5.0, "abs_reward_floor": 10.0, "min_rr": 2.0},
    "SMALL": {"abs_stop_cap": 6.0, "abs_reward_floor": 12.0, "min_rr": 2.0},
}

# ─────────────────────────────────────────────────────────────────────────────
# VCP W4 PROFILES — determines how much ATR room we give the stop
# The wider the base, the higher the required reward to compensate.
# ─────────────────────────────────────────────────────────────────────────────
VCP_PROFILES = {
    #  w4 <= threshold → profile name, ATR multiplier, min R:R required
    3.0:  ("TIGHT_VCP",    1.2, 2.7),
    5.0:  ("STANDARD",     1.5, 2.5),
    8.0:  ("WIDE_SWING",   1.8, 2.3),
}
# w4 > 8% is handled by VCPContractionGate upstream — should never reach here,
# but we add a safety fallback.
FALLBACK_PROFILE = ("WIDE_SWING", 1.8, 2.3)


def _get_profile(w4_pct: float) -> tuple:
    """Return (profile_name, atr_multiplier, min_rr) for a given W4 value."""
    for threshold in sorted(VCP_PROFILES.keys()):
        if w4_pct <= threshold:
            return VCP_PROFILES[threshold]
    return FALLBACK_PROFILE


def _compute_atr14(df: pd.DataFrame) -> float:
    """Pure-pandas ATR-14 — no external dependency."""
    if len(df) < 15:
        return 0.0
    high  = df["High"].squeeze().to_numpy(dtype=float)
    low   = df["Low"].squeeze().to_numpy(dtype=float)
    close = df["Close"].squeeze().to_numpy(dtype=float)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1])
        )
    )
    return float(np.mean(tr[-14:]))


class AsymmetryGate:
    """
    Drop-in replacement for the old AsymmetryGate.

    Legacy path (orchestrator unchanged):
        ag = AsymmetryGate(entry=r.entry_high, stop=r.stop_loss,
                           target1=r.target1, universe=universe)
        result = ag.check()        ← uses UNIVERSE_CAPS only, no ATR recompute

    Enhanced path (call from orchestrator after passing df + w4_pct):
        result = ag.check_dynamic(df=df, w4_pct=r.vcp_w4_pct)
                                   ← recomputes stop from ATR + EMA21 floor
                                      and scales required target accordingly

    Both paths return the same dict schema so downstream code is unchanged.
    """

    def __init__(self, entry: float, stop: float,
                 target1: float, universe: str = "LARGE"):
        self.entry    = entry
        self.stop     = stop
        self.target1  = target1
        self.universe = universe.upper() if universe else "LARGE"
        if self.universe not in UNIVERSE_CAPS:
            self.universe = "LARGE"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _base_result(self) -> dict:
        return {
            "qualified":    False,
            "risk_pct":     0.0,
            "reward_pct":   0.0,
            "rr_ratio":     0.0,
            "stop_price":   self.stop,
            "target_price": self.target1,
            "profile":      "LEGACY",
            "atr_pct":      0.0,
            "fail_reason":  "",
            "fail_stage":   "",
        }

    def _validate_inputs(self, result: dict) -> bool:
        """Returns True if inputs are sane, False + populates result if not."""
        if self.entry <= 0 or self.stop <= 0 or self.target1 <= 0:
            result["fail_reason"] = "AsymmetryGate: invalid inputs (zero/negative)"
            result["fail_stage"]  = "INPUT"
            return False
        if self.stop >= self.entry:
            result["fail_reason"] = "AsymmetryGate: stop >= entry"
            result["fail_stage"]  = "RISK"
            return False
        if self.target1 <= self.entry:
            result["fail_reason"] = "AsymmetryGate: target <= entry"
            result["fail_stage"]  = "REWARD"
            return False
        return True

    # ── legacy path (check) ──────────────────────────────────────────────────

    def check(self) -> dict:
        """
        Legacy-compatible check. Uses the stop/target already computed by
        RiskAgent. Applies UNIVERSE_CAPS as hard limits.
        Orchestrator calls this path — no change needed there.
        """
        result = self._base_result()
        if not self._validate_inputs(result):
            return result

        caps = UNIVERSE_CAPS[self.universe]

        risk_pct   = (self.entry  - self.stop)   / self.entry  * 100.0
        reward_pct = (self.target1 - self.entry) / self.entry  * 100.0
        rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0.0

        result["risk_pct"]   = round(risk_pct,   2)
        result["reward_pct"] = round(reward_pct, 2)
        result["rr_ratio"]   = round(rr_ratio,   2)

        if risk_pct > caps["abs_stop_cap"]:
            result["fail_reason"] = (
                f"AsymmetryGate: stop {risk_pct:.1f}% > cap "
                f"{caps['abs_stop_cap']}% for {self.universe}"
            )
            result["fail_stage"] = "RISK"
            return result

        if reward_pct < caps["abs_reward_floor"]:
            result["fail_reason"] = (
                f"AsymmetryGate: reward {reward_pct:.1f}% < floor "
                f"{caps['abs_reward_floor']}% for {self.universe}"
            )
            result["fail_stage"] = "REWARD"
            return result

        if rr_ratio < caps["min_rr"]:
            result["fail_reason"] = (
                f"AsymmetryGate: R:R {rr_ratio:.2f}x < min "
                f"{caps['min_rr']}x for {self.universe}"
            )
            result["fail_stage"] = "RR"
            return result

        result["qualified"] = True
        return result

    # ── enhanced path (check_dynamic) ────────────────────────────────────────

    def check_dynamic(self, df: pd.DataFrame, w4_pct: float = 4.0) -> dict:
        """
        Full dynamic check. Recomputes stop from ATR + EMA21 floor,
        then scales required target to maintain the profile's minimum R:R.

        Call this from orchestrator's G5 block as a drop-in upgrade:
            ag_result = ag.check_dynamic(df=df, w4_pct=r.vcp_w4_pct)
        """
        result = self._base_result()
        caps   = UNIVERSE_CAPS[self.universe]

        if self.entry <= 0:
            result["fail_reason"] = "AsymmetryGate: entry price is zero"
            result["fail_stage"]  = "INPUT"
            return result

        # ── ATR ──────────────────────────────────────────────────────────────
        atr = _compute_atr14(df)
        if atr <= 0:
            # Fall back to legacy path if ATR can't be computed
            log.debug("ATR=0 for %s, falling back to legacy check", self.entry)
            return self.check()

        atr_pct  = atr / self.entry * 100.0
        result["atr_pct"] = round(atr_pct, 2)

        # ── Profile selection ─────────────────────────────────────────────────
        profile_name, atr_mult, profile_min_rr = _get_profile(w4_pct)
        result["profile"] = profile_name

        # ── Dynamic stop computation ──────────────────────────────────────────
        dynamic_stop_pct = atr_pct * atr_mult

        # EMA21 floor: stop cannot be placed ABOVE EMA21
        close = df["Close"].squeeze().to_numpy(dtype=float)
        if len(close) >= 21:
            alpha = 2 / (21 + 1)
            ema21 = close[0]
            for c in close[1:]:
                ema21 = alpha * c + (1 - alpha) * ema21
            ema21_dist_pct = (self.entry - ema21) / self.entry * 100.0
            if ema21_dist_pct > 0:
                dynamic_stop_pct = max(dynamic_stop_pct, ema21_dist_pct)

        # Recent swing low floor (last 5 bars)
        if len(df) >= 5:
            swing_low = float(df["Low"].squeeze().iloc[-5:].min())
            swing_low_pct = (self.entry - swing_low) / self.entry * 100.0
            if swing_low_pct > 0:
                dynamic_stop_pct = max(dynamic_stop_pct, swing_low_pct)

        # ── Absolute cap check ────────────────────────────────────────────────
        if dynamic_stop_pct > caps["abs_stop_cap"]:
            result["fail_reason"] = (
                f"AsymmetryGate[{profile_name}]: ATR stop "
                f"{dynamic_stop_pct:.1f}% > cap {caps['abs_stop_cap']}% "
                f"({self.universe}) — volatility too extreme for current setup"
            )
            result["fail_stage"] = "RISK"
            result["risk_pct"]   = round(dynamic_stop_pct, 2)
            return result

        # ── Required reward ───────────────────────────────────────────────────
        effective_min_rr    = max(profile_min_rr, caps["min_rr"])
        required_reward_pct = dynamic_stop_pct * effective_min_rr
        required_reward_pct = max(required_reward_pct, caps["abs_reward_floor"])

        actual_reward_pct = (self.target1 - self.entry) / self.entry * 100.0

        result["risk_pct"]   = round(dynamic_stop_pct,   2)
        result["reward_pct"] = round(actual_reward_pct,  2)
        result["rr_ratio"]   = round(actual_reward_pct / dynamic_stop_pct, 2) if dynamic_stop_pct > 0 else 0.0

        if actual_reward_pct < required_reward_pct:
            result["fail_reason"] = (
                f"AsymmetryGate[{profile_name}]: headroom "
                f"{actual_reward_pct:.1f}% < required "
                f"{required_reward_pct:.1f}% "
                f"(stop {dynamic_stop_pct:.1f}% × {effective_min_rr:.1f}x R:R)"
            )
            result["fail_stage"] = "REWARD"
            return result

        if result["rr_ratio"] < caps["min_rr"]:
            result["fail_reason"] = (
                f"AsymmetryGate[{profile_name}]: R:R "
                f"{result['rr_ratio']:.2f}x < min {caps['min_rr']}x"
            )
            result["fail_stage"] = "RR"
            return result

        # ── PASS ──────────────────────────────────────────────────────────────
        stop_price   = self.entry * (1 - dynamic_stop_pct  / 100)
        target_price = self.entry * (1 + actual_reward_pct / 100)

        result["qualified"]    = True
        result["stop_price"]   = round(stop_price,   2)
        result["target_price"] = round(target_price, 2)
        return result

    # ── display ──────────────────────────────────────────────────────────────

    def summary(self, result: dict) -> str:
        if result["qualified"]:
            return (
                f"PASS [{result['profile']}] "
                f"risk={result['risk_pct']}% "
                f"reward={result['reward_pct']}% "
                f"RR={result['rr_ratio']}x"
            )
        return f"FAIL [{result['fail_stage']}] {result['fail_reason']}"
