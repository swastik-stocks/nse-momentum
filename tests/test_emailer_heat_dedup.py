"""
_portfolio_heat_alert must not repeat holdings already in EXIT ALERTS
(2026-11-07: the 5 Nov email listed 10 below-stop holdings in both tables).
"""
from emailer import _portfolio_heat_alert


def _heat(dist):
    return {"current_price": 100.0, "stop": 100.0 - dist, "distance_pct": dist, "method": "CAPPED"}


HEAT = {"ANANTRAJ.NS": _heat(-34.2), "JIOFIN.NS": _heat(-1.7),
        "RELIANCE.NS": _heat(2.9), "YATHARTH.NS": _heat(5.0), "TCS.NS": _heat(12.0)}


def test_exit_alert_tickers_excluded():
    html = _portfolio_heat_alert(HEAT, exclude_tickers={"ANANTRAJ.NS", "JIOFIN.NS"})
    assert "ANANTRAJ" not in html and "JIOFIN" not in html
    assert "RELIANCE" in html and "YATHARTH" in html and "TCS" not in html
    assert "2 position(s) within 5% of their stop" in html


def test_below_stop_but_not_exit_alert_still_shown():
    html = _portfolio_heat_alert(HEAT, exclude_tickers={"ANANTRAJ.NS"})
    assert "JIOFIN" in html


def test_default_behaviour_unchanged():
    assert "ANANTRAJ" in _portfolio_heat_alert(HEAT)


def test_all_excluded_section_disappears():
    only_exits = {k: v for k, v in HEAT.items() if v["distance_pct"] < 0}
    assert _portfolio_heat_alert(only_exits, exclude_tickers=set(only_exits)) == ""
