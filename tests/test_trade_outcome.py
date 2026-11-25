"""Tests for validation/trade_outcome.py — the shared tradeable-exit simulator."""
import pandas as pd
import pytest

from validation.trade_outcome import simulate_trade


def _df(bars):
    """bars: list of (open, high, low, close). Bar 0 is the signal bar."""
    return pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"])


SIGNAL = (100, 100, 98, 99)   # signal bar; entry 100 is a buy-stop above it


def test_stop_hit_before_later_high_is_a_loss():
    # The old excursion scoring called this a big win (high of 120 within
    # 20 bars). A real trade with a 95 stop was stopped out on bar 2.
    df = _df([SIGNAL, (99, 101, 99, 100), (99, 99, 94, 95), (96, 120, 96, 119)])
    t = simulate_trade(df, 0, entry=100, stop=95, target=110)
    assert t["exit_reason"] == "STOP"
    assert t["net_r"] == pytest.approx(-1.0)
    assert not t["is_win"]


def test_target_hit_books_target_not_the_high():
    df = _df([SIGNAL, (99, 101, 99, 100), (101, 125, 100, 124)])
    t = simulate_trade(df, 0, entry=100, stop=95, target=110)
    assert t["exit_reason"] == "TARGET"
    assert t["exit_price"] == 110
    assert t["net_r"] == pytest.approx(2.0)


def test_entry_never_reached_is_no_trade():
    df = _df([SIGNAL] + [(97, 99, 96, 98)] * 25)
    assert simulate_trade(df, 0, entry=100, stop=95, target=110) is None


def test_gap_above_entry_fills_at_open_and_measures_r_from_fill():
    df = _df([SIGNAL, (104, 106, 103, 105), (105, 106, 104, 105)])
    t = simulate_trade(df, 0, entry=100, stop=95, target=120, forward_bars=2)
    assert t["fill_price"] == 104
    assert t["risk_pct"] == pytest.approx(9 / 104, rel=1e-3)
    assert t["exit_reason"] == "TIME"


def test_gap_down_through_stop_exits_at_open():
    df = _df([SIGNAL, (99, 101, 99, 100), (90, 91, 89, 90)])
    t = simulate_trade(df, 0, entry=100, stop=95, target=110)
    assert t["exit_reason"] == "STOP_GAP"
    assert t["exit_price"] == 90
    assert t["net_r"] == pytest.approx(-2.0)


def test_same_bar_stop_and_target_counts_as_stop():
    df = _df([SIGNAL, (99, 101, 99, 100), (100, 112, 94, 105)])
    assert simulate_trade(df, 0, entry=100, stop=95, target=110)["exit_reason"] == "STOP"


def test_time_exit_uses_close_and_costs_are_subtracted():
    df = _df([SIGNAL] + [(100, 101, 99, 102)] * 25)
    t = simulate_trade(df, 0, entry=100, stop=95, target=110, forward_bars=20, cost_pct=0.01)
    assert t["exit_reason"] == "TIME"
    assert t["bars_held"] == 20
    assert t["net_return"] == pytest.approx(0.02 - 0.01)


def test_defaults_when_no_stop_or_target():
    df = _df([SIGNAL, (99, 101, 99, 100), (101, 111, 100, 110)])
    t = simulate_trade(df, 0, entry=100)
    assert t["risk_pct"] == pytest.approx(0.05)
    assert t["exit_reason"] == "TARGET" and t["exit_price"] == pytest.approx(110)


def test_require_full_window_returns_none_on_short_history():
    df = _df([SIGNAL, (99, 101, 99, 100)])
    assert simulate_trade(df, 0, entry=100, require_full_window=True) is None
