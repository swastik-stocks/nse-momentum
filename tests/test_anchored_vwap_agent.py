"""
[2026-08-22] Regression tests for agents/anchored_vwap_agent.py -- the
support-anchored VWAP factor synthesized from prorealcode.com's "Auto
Midas Anchored VWAP" indicator (72-bar / "value zone" level only, first
test scoped cheap -- see the agent's own docstring).
"""
import numpy as np
import pandas as pd
import pytest

from agents.anchored_vwap_agent import (
    compute_stock_support_avwap_components,
    compute_universe_avwap_percentiles,
)


def _flat_bar_series(closes: list, volumes: list) -> pd.DataFrame:
    """High=Low=Close for every bar -- typical price collapses to Close
    exactly, making the expected AVWAP value hand-computable."""
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes,
    }, index=idx)


def test_support_avwap_anchors_to_the_lowest_low_in_the_window():
    # Hand-computed: closes=[10,8,12,15,20], vols=[100,200,100,100,100],
    # lookback=5 -> anchor is bar 1 (Close=8, the lowest).
    # AVWAP from bar 1 to bar 4 = (8*200+12*100+15*100+20*100)/(200+100+100+100)
    #                           = 6300/500 = 12.6
    # pct_above = (20-12.6)/12.6 = 0.587301...
    df = _flat_bar_series([10, 8, 12, 15, 20], [100, 200, 100, 100, 100])
    result = compute_stock_support_avwap_components(df, lookback=5)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[3])   # not enough bars yet (need 5)
    assert result.iloc[4] == pytest.approx(0.587301, rel=1e-4)


def test_support_avwap_rolls_the_anchor_forward_as_new_lows_appear():
    # A new, lower low pushes the anchor forward for subsequent bars.
    closes = [10, 8, 12, 15, 20, 5, 9]
    vols = [100] * 7
    df = _flat_bar_series(closes, vols)
    result = compute_stock_support_avwap_components(df, lookback=5)
    # At index 6 (bars 2-6), the lowest Low is index 5 (Close=5) -> anchor=5.
    # AVWAP(5,6) = (5*100 + 9*100) / 200 = 7.0. pct_above = (9-7)/7 = 0.2857
    assert result.iloc[6] == pytest.approx(0.285714, rel=1e-4)


def test_flat_price_series_yields_zero_pct_above_support():
    df = _flat_bar_series([50.0] * 10, [100] * 10)
    result = compute_stock_support_avwap_components(df, lookback=5)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_universe_percentiles_rank_correctly():
    # 12 tickers (clears the internal N<10 floor), pct_above_support
    # values deliberately ordered -- percentile ranking must preserve
    # their relative order.
    def _mk(jump_close):
        closes = [100.0] * 6 + [jump_close]
        return _flat_bar_series(closes, [100] * 7)

    jumps = [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123]
    data_dict = {"stock_data": {f"T{i}.NS": _mk(j) for i, j in enumerate(jumps)}}
    percentiles = compute_universe_avwap_percentiles(data_dict, lookback=5)

    assert len(percentiles) == 12
    ordered = sorted(percentiles.items(), key=lambda kv: jumps[int(kv[0][1:-3])])
    pct_values = [p for _, p in ordered]
    assert pct_values == sorted(pct_values)  # monotonically non-decreasing with the jump size
