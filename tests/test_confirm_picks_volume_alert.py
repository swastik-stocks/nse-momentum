"""
[2026-08-23] Regression tests for confirm_picks.py's real-time volume-cross
alert -- fires when a previous day's recommendation crosses RVOL_CONFIRM_MIN
(1.5x), independent of price/pivot classification, via both Telegram and
email. Only the pure decision function and the message-building/dispatch
function are unit-tested here; the live fetch/classify loop this plugs into
has no existing test harness (real-market-dependent, verified live per the
module's own established convention -- see MEESHO/GLAXO incident notes).
"""
from unittest.mock import patch

from confirm_picks import (
    RVOL_CONFIRM_MIN,
    _crosses_volume_alert_threshold,
    _send_volume_cross_alert,
    _volume_alert_state_path,
)


def test_fires_when_rvol_crosses_threshold_and_not_already_alerted():
    assert _crosses_volume_alert_threshold(
        rvol=1.6, ticker_key="TCS", status="PENDING", already_alerted={}
    ) is True


def test_does_not_fire_below_threshold():
    assert _crosses_volume_alert_threshold(
        rvol=1.2, ticker_key="TCS", status="PENDING", already_alerted={}
    ) is False


def test_does_not_fire_twice_for_same_ticker():
    assert _crosses_volume_alert_threshold(
        rvol=1.8, ticker_key="TCS", status="PENDING",
        already_alerted={"TCS": "09:35"},
    ) is False


def test_does_not_double_fire_alongside_confirmed_alert():
    for status in ("CONFIRMED", "CONFIRMED_LOW_VOL"):
        assert _crosses_volume_alert_threshold(
            rvol=2.0, ticker_key="TCS", status=status, already_alerted={}
        ) is False


def test_still_fires_for_other_non_terminal_statuses_above_threshold():
    assert _crosses_volume_alert_threshold(
        rvol=1.5, ticker_key="TCS", status="BREAKOUT_NO_VOLUME", already_alerted={}
    ) is True


def test_volume_alert_state_path_is_separate_from_confirm_state():
    path = _volume_alert_state_path("2026-08-23")
    assert "volume_alert_state_20260823" in path
    assert "confirm_state" not in path


def test_send_volume_cross_alert_pushes_telegram_and_email():
    pick = {"entry": 100.0, "pivot": 98.5}
    with patch("confirm_picks.send_telegram_alert") as mock_tg, \
         patch("confirm_picks.send_email") as mock_email:
        _send_volume_cross_alert("TCS", pick, cmp=101.2, rvol=1.7, rvol_src="Dhan")

    mock_tg.assert_called_once()
    tg_msg = mock_tg.call_args[0][0]
    assert "TCS" in tg_msg
    assert "1.7x" in tg_msg
    assert "Dhan" in tg_msg
    # [2026-08-23] Bold + HTML per user feedback -- volume alerts are the
    # most time-sensitive of the three alert types and must stand out.
    assert "<b>VOLUME ALERT: TCS</b>" in tg_msg
    assert mock_tg.call_args.kwargs.get("parse_mode") == "HTML"

    mock_email.assert_called_once()
    subject, html = mock_email.call_args[0]
    assert "TCS" in subject
    assert str(RVOL_CONFIRM_MIN) in subject
    assert "TCS" in html
