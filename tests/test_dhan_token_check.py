"""
[2026-08-21] Regression tests for dhan_token_check.py -- the JWT exp-claim
decoder used to give scanner.py an early, specific warning when the Dhan
access token has lapsed, instead of a 30-minute misdiagnosed retry loop.
"""
import base64
import json
from datetime import datetime, timedelta, timezone

from dhan_token_check import decode_dhan_token_expiry, dhan_token_status


def _fake_jwt(exp_epoch: int) -> str:
    """Builds a syntactically valid (unsigned) JWT with the given exp claim
    -- decode_dhan_token_expiry never checks the signature, only the
    payload segment, so a dummy header/signature is fine."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS512"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp_epoch}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesignature"


def test_decode_dhan_token_expiry_reads_exp_claim():
    exp = int((datetime.now(timezone.utc) + timedelta(hours=5)).timestamp())
    token = _fake_jwt(exp)
    decoded = decode_dhan_token_expiry(token)
    assert decoded is not None
    assert abs(decoded.timestamp() - exp) < 2


def test_decode_dhan_token_expiry_handles_malformed_token():
    assert decode_dhan_token_expiry("not-a-jwt") is None
    assert decode_dhan_token_expiry("") is None
    assert decode_dhan_token_expiry("a.b") is None


def test_dhan_token_status_flags_expired_token():
    past_exp = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp())
    token = _fake_jwt(past_exp)
    status = dhan_token_status(token)
    assert status["expired"] is True
    assert status["hours_remaining"] < 0


def test_dhan_token_status_flags_valid_token():
    future_exp = int((datetime.now(timezone.utc) + timedelta(hours=10)).timestamp())
    token = _fake_jwt(future_exp)
    status = dhan_token_status(token)
    assert status["expired"] is False
    assert 9 < status["hours_remaining"] < 11


def test_dhan_token_status_handles_missing_token():
    status = dhan_token_status("")
    assert status["expiry"] is None
    assert status["expired"] is None
    assert status["hours_remaining"] is None
