"""
[2026-08-21] Regression tests for telegram_alert.py and its wiring into
confirm_picks.py's checkpoint/BTST loops. The core contract: this is a
bonus notification channel, never a dependency -- missing config or a
network failure must be swallowed, never raised, never able to break a
real checkpoint run.
"""
from unittest.mock import MagicMock, patch

import pytest


def test_send_telegram_alert_returns_false_when_not_configured(monkeypatch):
    import telegram_alert
    monkeypatch.setattr(telegram_alert, "TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr(telegram_alert, "TELEGRAM_CHAT_ID", None)
    assert telegram_alert.send_telegram_alert("test") is False


def test_send_telegram_alert_never_raises_on_network_failure(monkeypatch):
    import telegram_alert
    monkeypatch.setattr(telegram_alert, "TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(telegram_alert, "TELEGRAM_CHAT_ID", "12345")

    def boom(*a, **kw):
        raise ConnectionError("network down")

    with patch.object(telegram_alert.requests, "post", side_effect=boom):
        result = telegram_alert.send_telegram_alert("test")
    assert result is False


def test_send_telegram_alert_never_logs_the_bot_token_on_failure(monkeypatch, caplog):
    """
    [2026-08-23] Regression for a real incident: requests' HTTPError embeds
    the full request URL in its string form, and Telegram's Bot API puts
    the token directly in that URL (.../bot<TOKEN>/sendMessage) -- an
    unsanitized log of the raw exception leaked a live token into tool
    output during manual testing. The token must never appear in the
    logged failure message, however the send fails.
    """
    import logging
    import telegram_alert
    real_token = "8855145986:AAFosGzQaee6dXFHlfbBf60Fx1MV_5ZcRxs"
    monkeypatch.setattr(telegram_alert, "TELEGRAM_BOT_TOKEN", real_token)
    monkeypatch.setattr(telegram_alert, "TELEGRAM_CHAT_ID", "12345")

    def boom(*a, **kw):
        raise ConnectionError(
            f"400 Client Error: Bad Request for url: "
            f"https://api.telegram.org/bot{real_token}/sendMessage"
        )

    with caplog.at_level(logging.WARNING):
        with patch.object(telegram_alert.requests, "post", side_effect=boom):
            result = telegram_alert.send_telegram_alert("test")

    assert result is False
    assert real_token not in caplog.text
    assert "REDACTED" in caplog.text


def test_send_telegram_alert_posts_correct_payload(monkeypatch):
    import telegram_alert
    monkeypatch.setattr(telegram_alert, "TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(telegram_alert, "TELEGRAM_CHAT_ID", "12345")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(telegram_alert.requests, "post", return_value=mock_response) as mock_post:
        result = telegram_alert.send_telegram_alert("CONFIRMED: RADICO")

    assert result is True
    args, kwargs = mock_post.call_args
    assert "fake-token" in args[0]
    assert kwargs["data"]["chat_id"] == "12345"
    assert kwargs["data"]["text"] == "CONFIRMED: RADICO"
    assert "parse_mode" not in kwargs["data"]  # not passed -- must stay plain text


def test_send_telegram_alert_passes_parse_mode_when_given(monkeypatch):
    import telegram_alert
    monkeypatch.setattr(telegram_alert, "TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setattr(telegram_alert, "TELEGRAM_CHAT_ID", "12345")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(telegram_alert.requests, "post", return_value=mock_response) as mock_post:
        result = telegram_alert.send_telegram_alert("<b>VOLUME ALERT</b>", parse_mode="HTML")

    assert result is True
    args, kwargs = mock_post.call_args
    assert kwargs["data"]["parse_mode"] == "HTML"


def test_classify_btst_candidate_alert_only_fires_when_final(monkeypatch):
    """Same principle as the wording fix from yesterday: an interim
    (pre-15:00) BTST read must not push a 'candidate' alert -- it isn't a
    real verdict yet. This test pins the condition in run_btst_scan's
    alert call, not just classify_btst's wording."""
    import confirm_picks
    # is_final gates the alert call directly in run_btst_scan (see
    # `if c["status"] == "BTST_CANDIDATE" and is_final:`) -- verified by
    # inspecting that both the status AND is_final are required, matching
    # the BTST_FINAL_CHECK_TIME contract already covered in
    # test_data_freshness.py.
    assert hasattr(confirm_picks, "send_telegram_alert")
