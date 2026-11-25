"""
[2026-11-25] Fixes 4-7 from the pick-quality investigation:
  4. RegimeClassifier trend_scoring="graded" removes the EMA50>EMA200 cliff.
  6. RiskAgent stop_mode / respect_resistance switches (defaults unchanged).
  7. Card wording: RS trigger only when RS < 65; volume is a morning-check
     note, not a "missing" Tier 1 condition; RSI >= 80 is a risk, not "constructive".
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.regime_classifier import RegimeClassifier
from agents.risk_agent import RiskAgent
from orchestrator import AgentOrchestrator, StockResult

RegimeClassifier._log_calibration = lambda self: None


# ── 4. Regime trend scoring ──────────────────────────────────────────────────

def _v_shaped_nifty():
    """Long decline then a sharp rally: price above EMA20/50/200, EMA50 still
    just under EMA200 -- the 23 Nov 2026 shape."""
    down = np.linspace(26000, 22400, 220)
    up = np.linspace(22400, 25350, 22)
    return pd.DataFrame({"Close": np.concatenate([down, up])})


def test_stack_scoring_hits_the_cliff_and_graded_does_not():
    n = _v_shaped_nifty()
    c = n["Close"]
    e50, e200 = c.ewm(span=50).mean().iloc[-1], c.ewm(span=200).mean().iloc[-1]
    assert c.iloc[-1] > e50 and e50 < e200          # the shape we mean to test
    stack = RegimeClassifier(nifty50_data=n).classify()["component_detail"]["nifty50"]["pts"]
    graded = RegimeClassifier(nifty50_data=n, trend_scoring="graded").classify()["component_detail"]["nifty50"]["pts"]
    assert stack == 2
    assert graded >= 3


def test_graded_keeps_the_same_0_to_4_range():
    rising = pd.DataFrame({"Close": np.linspace(20000, 26000, 300)})
    falling = pd.DataFrame({"Close": np.linspace(26000, 20000, 300)})
    assert RegimeClassifier(nifty50_data=rising, trend_scoring="graded").classify()["component_detail"]["nifty50"]["pts"] == 4
    assert RegimeClassifier(nifty50_data=falling, trend_scoring="graded").classify()["component_detail"]["nifty50"]["pts"] == 0


def test_default_trend_scoring_is_unchanged():
    n = _v_shaped_nifty()
    assert RegimeClassifier(nifty50_data=n).trend_scoring == "stack"
    with pytest.raises(ValueError):
        RegimeClassifier(trend_scoring="nope")


# ── 6. RiskAgent switches ────────────────────────────────────────────────────

def _ohlc(seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, 300))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})


def test_risk_agent_defaults_match_explicit_original_rules():
    df = _ohlc(); bo = float(df["High"].iloc[-20:].max())
    a = RiskAgent(df, bo, bo * 0.995, bo * 1.005, universe="MID")
    b = RiskAgent(df, bo, bo * 0.995, bo * 1.005, universe="MID", stop_mode="tightest", respect_resistance=False)
    assert (a.passes(), a.stop, a.target1) == (b.passes(), b.stop, b.target1)


def test_loosest_stop_is_never_tighter_than_tightest():
    for seed in range(10):
        df = _ohlc(seed); bo = float(df["High"].iloc[-20:].max())
        t = RiskAgent(df, bo, bo * 0.995, bo * 1.005, universe="MID", stop_mode="tightest")
        l = RiskAgent(df, bo, bo * 0.995, bo * 1.005, universe="MID", stop_mode="loosest")
        if t.passes() and l.passes():
            assert l.stop <= t.stop


def test_stop_cap_is_a_minimum_distance():
    for seed in range(10):
        df = _ohlc(seed); bo = float(df["High"].iloc[-20:].max())
        r = RiskAgent(df, bo, bo * 0.995, bo * 1.005, universe="LARGE")
        if r.passes():
            assert r.stop_pct >= 3.0 - 0.05


def test_bad_stop_mode_rejected():
    with pytest.raises(ValueError):
        RiskAgent(_ohlc(), 100, 99, 101, stop_mode="widest")


# ── 7. Card wording ──────────────────────────────────────────────────────────

def _r(**kw):
    base = dict(ticker="X.NS", name="X", sector="IT", price=100.0, tier=2, total_score=60,
                pattern="Swing High Breakout", entry=100.0, stop_loss=95.0, target1=110.0,
                target2=120.0, rrr=2.0, rs_percentile=99.0, rvol=0.8, del_pct=40.0,
                universe="MID", breakout_level=100.0, regime="C", rsi_val=84.0)
    base.update(kw)
    return StockResult(**base)


SELF = SimpleNamespace(macro_state="MIXED")


def test_rs_trigger_only_when_rs_is_actually_low():
    assert not any("RS rank" in t for t in AgentOrchestrator._triggers(SELF, _r(rs_percentile=99)))
    assert any("RS rank" in t and "now 50th" in t for t in AgentOrchestrator._triggers(SELF, _r(rs_percentile=50)))


def test_volume_is_not_a_missing_tier1_condition_but_a_morning_check_note():
    missing = AgentOrchestrator._what_missing(SELF, _r(rvol=0.5), 64)
    assert not any("Volume" in m for m in missing)
    risks = AgentOrchestrator._risk_factors(SELF, _r(rvol=0.5))
    assert any("morning check" in x for x in risks)


def test_overbought_rsi_is_a_risk_not_constructive():
    risks = AgentOrchestrator._risk_factors(SELF, _r(rsi_val=84, rvol=2.0))
    assert any("RSI(D) 84 - extended" in x for x in risks)


# ── 4. Regime explanation in the email ───────────────────────────────────────

def test_regime_explainer_names_the_inputs_and_the_stakes():
    from emailer import _regime_explainer
    r = RegimeClassifier(nifty50_data=_v_shaped_nifty(), vix=17.7, ad_ratio=0.42,
                         breadth_score=6.0).classify()
    html = _regime_explainer(r["regime"], {"raw_pts": r["raw_pts"], "components": r["component_detail"]})
    assert f"Why {r['regime']}:" in html
    assert "still below EMA200" in html          # the cliff is spelled out
    assert "advance/decline 0.42 (-1)" in html
    assert "C -0.4%" in html                      # historical stakes shown
    assert _regime_explainer("C", None) == ""


# ── Deep replay re-score (2026-11-25): pattern is not presented as the edge ──

def test_email_says_pattern_is_not_the_edge():
    from emailer import _pattern_edge_note
    note = _pattern_edge_note()
    assert "-0.01R" in note and "+0.06R" in note and "regime" in note


def test_why_working_frames_pattern_as_entry_zone():
    r = _r(entry_low=99.0, entry_high=101.0, rs_percentile=50, rvol=0.5, del_pct=10)
    r.breakout_quality = "MAJOR"
    reasons = AgentOrchestrator._why_working(SimpleNamespace(macro_state="MIXED"), r)
    assert any(x.startswith("Entry zone Rs.99.0-101.0 (Swing High Breakout setup [MAJOR])") for x in reasons)


def test_live_patterns_still_not_low_edge():
    """Guard: the stale PATTERN_EXPECTANCY values are what keep Tier 1 alive."""
    from agents.pattern_agent import is_low_edge_pattern
    assert not is_low_edge_pattern("Cup & Handle")
    assert not is_low_edge_pattern("Swing High Breakout")
