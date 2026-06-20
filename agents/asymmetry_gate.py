"""NSE Momentum v5.1 - AsymmetryGate - Relaxed thresholds"""
import logging
log = logging.getLogger(__name__)

THRESHOLDS = {
    "LARGE": {"max_risk": 5.0, "min_reward": 8.0,  "min_rr": 1.5},
    "MID":   {"max_risk": 6.0, "min_reward": 10.0, "min_rr": 1.5},
    "SMALL": {"max_risk": 7.0, "min_reward": 12.0, "min_rr": 1.5},
}

class AsymmetryGate:
    def __init__(self, entry: float, stop: float,
                 target1: float, universe: str = "LARGE"):
        self.entry    = entry
        self.stop     = stop
        self.target1  = target1
        self.universe = universe.upper()

    def check(self) -> dict:
        result = {"qualified": False, "risk_pct": 0.0, "reward_pct": 0.0,
                  "rr_ratio": 0.0, "fail_reason": "", "fail_stage": ""}
        entry, stop, target1 = self.entry, self.stop, self.target1
        t = THRESHOLDS.get(self.universe, THRESHOLDS["LARGE"])
        if entry <= 0 or stop <= 0 or target1 <= 0:
            result["fail_reason"] = "AsymmetryGate: invalid inputs"
            result["fail_stage"]  = "RISK"
            return result
        if stop >= entry:
            result["fail_reason"] = f"AsymmetryGate: stop >= entry"
            result["fail_stage"]  = "RISK"
            return result
        if target1 <= entry:
            result["fail_reason"] = f"AsymmetryGate: target <= entry"
            result["fail_stage"]  = "REWARD"
            return result
        risk_pct   = (entry - stop)    / entry * 100.0
        reward_pct = (target1 - entry) / entry * 100.0
        rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0.0
        result["risk_pct"]   = round(risk_pct,   2)
        result["reward_pct"] = round(reward_pct, 2)
        result["rr_ratio"]   = round(rr_ratio,   2)
        if risk_pct > t["max_risk"]:
            result["fail_reason"] = f"AsymmetryGate: risk {risk_pct:.1f}% > max {t['max_risk']}% for {self.universe}"
            result["fail_stage"]  = "RISK"
            return result
        if reward_pct < t["min_reward"]:
            result["fail_reason"] = f"AsymmetryGate: reward {reward_pct:.1f}% < min {t['min_reward']}% for {self.universe}"
            result["fail_stage"]  = "REWARD"
            return result
        if rr_ratio < t["min_rr"]:
            result["fail_reason"] = f"AsymmetryGate: R:R {rr_ratio:.2f}x < min {t['min_rr']}x"
            result["fail_stage"]  = "RR"
            return result
        result["qualified"] = True
        return result

    def summary(self, result: dict) -> str:
        if result["qualified"]:
            return f"PASS risk={result['risk_pct']}% reward={result['reward_pct']}% RR={result['rr_ratio']}x"
        return f"FAIL [{result['fail_stage']}] {result['fail_reason']}"
