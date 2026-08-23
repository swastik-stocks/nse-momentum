"""
[2026-08-21] Regression tests for renew_dhan_token.py -- proactive Dhan
token renewal via /v2/RenewToken, meant to replace the daily manual
regenerate-at-web.dhan.co step. All Dhan API calls are mocked: Dhan has no
sandbox, and hitting the real endpoint rotates the actual live token, so
these tests only exercise the script's own logic (request construction,
response parsing, .env rewriting, the already-expired guard).
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from renew_dhan_token import renew_dhan_token, update_env_token, main
from tests.test_dhan_token_check import _fake_jwt


def test_renew_dhan_token_success_returns_new_token():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"accessToken": "brand-new-token"}
    with patch("renew_dhan_token.requests.get", return_value=mock_resp) as mock_get:
        result = renew_dhan_token("12345", "old-token")
    assert result == "brand-new-token"
    args, kwargs = mock_get.call_args
    assert kwargs["headers"]["access-token"] == "old-token"
    assert kwargs["headers"]["dhanClientId"] == "12345"


def test_renew_dhan_token_returns_none_on_missing_field():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"someOtherField": "x"}
    with patch("renew_dhan_token.requests.get", return_value=mock_resp):
        result = renew_dhan_token("12345", "old-token")
    assert result is None


def test_renew_dhan_token_returns_none_on_network_error():
    with patch("renew_dhan_token.requests.get", side_effect=ConnectionError("down")):
        result = renew_dhan_token("12345", "old-token")
    assert result is None


def test_update_env_token_replaces_existing_line(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GMAIL_ADDRESS=x@x.com\nDHAN_ACCESS_TOKEN=old-value\nDHAN_CLIENT_ID=123\n")
    update_env_token("new-value", env_file)
    content = env_file.read_text()
    assert "DHAN_ACCESS_TOKEN=new-value" in content
    assert "old-value" not in content
    assert "GMAIL_ADDRESS=x@x.com" in content  # untouched


def test_update_env_token_appends_if_missing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GMAIL_ADDRESS=x@x.com\n")
    update_env_token("new-value", env_file)
    assert "DHAN_ACCESS_TOKEN=new-value" in env_file.read_text()


def test_main_refuses_to_renew_an_already_expired_token(tmp_path, monkeypatch):
    """RenewToken errors on an expired token per Dhan's docs -- the script
    must check this itself up front and fail fast with a clear message
    rather than burning an API call to find out."""
    expired_token = _fake_jwt(int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()))
    env_file = tmp_path / ".env"
    env_file.write_text(f"DHAN_CLIENT_ID=123\nDHAN_ACCESS_TOKEN={expired_token}\n")

    monkeypatch.setattr(sys := __import__("sys"), "argv",
                         ["renew_dhan_token.py", "--env-path", str(env_file)])

    with patch("renew_dhan_token.renew_dhan_token") as mock_renew:
        rc = main()
    assert rc == 1
    mock_renew.assert_not_called()


def test_main_happy_path_updates_env_without_github_push(tmp_path, monkeypatch):
    valid_token = _fake_jwt(int((datetime.now(timezone.utc) + timedelta(hours=5)).timestamp()))
    env_file = tmp_path / ".env"
    env_file.write_text(f"DHAN_CLIENT_ID=123\nDHAN_ACCESS_TOKEN={valid_token}\n")

    new_token = _fake_jwt(int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()))
    monkeypatch.setattr(__import__("sys"), "argv",
                         ["renew_dhan_token.py", "--env-path", str(env_file)])

    with patch("renew_dhan_token.renew_dhan_token", return_value=new_token) as mock_renew:
        rc = main()
    assert rc == 0
    mock_renew.assert_called_once_with("123", valid_token)
    assert new_token in env_file.read_text()
