"""Structural breadth + neutral-breadth handling (2026-11-17 regime double-count fix)."""
import numpy as np
import pandas as pd
import pytest

from agents.macro_agent import MacroAgent
from agents.regime_classifier import RegimeClassifier
from agents.structural_breadth import (AD_WINDOW, HL_WINDOW, MIN_STOCKS,
                                       compute_structural_breadth, structural_breadth_series)

RegimeClassifier._log_calibration = lambda self: None   # keep tests from writing the audit log


def _market(n_stocks=150, n_days=320, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    rets = rng.normal(drift, 0.01, size=(n_days, n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i}.NS" for i in range(n_stocks)])
    return close


def test_uptrend_scores_above_downtrend():
    up = structural_breadth_series(_market(drift=0.003)).score.iloc[-1]
    down = structural_breadth_series(_market(drift=-0.003)).score.iloc[-1]
    assert up > 6 > 4 > down


def test_single_down_day_barely_moves_score():
    """The bug being fixed: one session must not swing breadth by a regime level."""
    close = _market(drift=0.002)
    before = structural_breadth_series(close).score.iloc[-1]
    # A 16-Nov-style session: ~70% of stocks down 1%, the rest up 0.5% (A/D ~0.43).
    move = np.where(np.arange(close.shape[1]) % 10 < 7, 0.99, 1.005)
    close.iloc[-1] = close.iloc[-2] * move
    after = structural_breadth_series(close).score.iloc[-1]
    assert before - after < 1.0
    # The legacy breadth score for the same session: A/D 0.43 -> 3/10, from
    # wherever it was the day before -- a multi-point swing by construction.


def test_new_highs_need_full_prior_window():
    close = _market(n_days=HL_WINDOW - 10, drift=0.01)   # strong rally, short history
    out = structural_breadth_series(close)
    assert (out.new_highs == 0).all(), "no 52w high can exist without 52w of history"


def test_too_few_stocks_gives_no_score():
    out = structural_breadth_series(_market(n_stocks=MIN_STOCKS - 1))
    assert out.score.isna().all()


def test_member_mask_excludes_non_members():
    close = _market()
    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    mask.iloc[:, :60] = False
    assert (structural_breadth_series(close, member_mask=mask).n_stocks == 90).all()


def test_live_entry_point_uses_last_row_and_ohlc():
    close = _market(drift=0.002)
    stock_data = {t: pd.DataFrame({"Close": close[t], "High": close[t] * 1.01, "Low": close[t] * 0.99})
                  for t in close.columns}
    r = compute_structural_breadth(stock_data)
    assert r and r["source"] == "UNIVERSE_STRUCTURAL" and r["n_stocks"] == 150
    assert r["as_of"] == close.index[-1].date().isoformat()
    assert r["score"] == pytest.approx(structural_breadth_series(
        close, close * 1.01, close * 0.99).score.iloc[-1], abs=0.01)


def test_live_entry_point_never_raises():
    assert compute_structural_breadth({}) is None
    assert compute_structural_breadth({"X.NS": pd.DataFrame()}) is None


def _nifty(n=260):
    return pd.DataFrame({"Close": np.linspace(20000, 25000, n)})


def test_regime_classifier_none_breadth_is_neutral():
    kw = dict(nifty50_data=_nifty(), vix=17.0, ad_ratio=0.42, above_50_ema_pct=56.5)
    neutral = RegimeClassifier(breadth_score=None, **kw).classify()
    mid = RegimeClassifier(breadth_score=5.0, **kw).classify()     # 5.0 -> adj 0
    assert neutral["raw_pts"] == mid["raw_pts"]
    assert neutral["component_detail"]["breadth"]["adj"] == 0


def test_regime_classifier_double_count_costs_a_point():
    """A/D 0.42 used to cost -1 (ad_pts) AND -1 (breadth 3/10 from the same A/D)."""
    kw = dict(nifty50_data=_nifty(), vix=17.0, ad_ratio=0.42, above_50_ema_pct=56.5)
    assert (RegimeClassifier(breadth_score=None, **kw).classify()["raw_pts"]
            - RegimeClassifier(breadth_score=3.0, **kw).classify()["raw_pts"]) == 1


def test_macro_agent_none_breadth_is_neutral():
    m = MacroAgent(vix=17.0, breadth_score=None, fii_flow_crore=0.0, adv_dec_ratio=1.0)
    assert m.get_detail()["breadth"]["pts"] == 0
