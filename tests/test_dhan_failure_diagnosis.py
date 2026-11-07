"""
data_fetcher._diagnose_dhan_failure -- maps Dhan's 401 error codes to the
real cause. Added 2026-11-07 after weeks of "refresh the token" warnings
whose actual cause was a lapsed Data APIs plan (DH-902), not the token.
"""
import base64
import json
import time

from data_fetcher import _diagnose_dhan_failure


def _jwt(exp_epoch: int) -> str:
    """Unsigned JWT with just an exp claim -- dhan_token_check never verifies signatures."""
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp_epoch}).encode()).decode().rstrip("=")
    return f"hdr.{payload}.sig"


DH902 = json.dumps({"errorType": "Invalid_Access", "errorCode": "DH-902",
                    "errorMessage": "This API requires a Data APIs Plan. Current plan: Free Plan"})
DH901 = json.dumps({"errorType": "Invalid_Authentication", "errorCode": "DH-901",
                    "errorMessage": "Client ID or user generated access token is invalid or expired."})


def test_dh902_is_plan_not_token():
    d = _diagnose_dhan_failure(401, DH902, _jwt(int(time.time()) + 3600))
    assert d["reason"] == "plan_inactive"
    assert "will NOT fix" in d["message"]
    assert "Data APIs" in d["action"]


def test_dh901_expired_token_reports_expiry_and_paste():
    d = _diagnose_dhan_failure(401, DH901, _jwt(int(time.time()) - 3600))
    assert d["reason"] == "token_invalid"
    assert "EXPIRED" in d["message"]
    assert "--paste" in d["action"]


def test_dh901_unexpired_token_flags_revocation():
    # Valid exp but rejected anyway -> token was replaced by a newer one
    # (each web.dhan.co generation invalidates the previous token).
    d = _diagnose_dhan_failure(401, DH901, _jwt(int(time.time()) + 3600))
    assert d["reason"] == "token_invalid"
    assert "revoked/replaced" in d["message"]


def test_dh901_undecodable_token():
    d = _diagnose_dhan_failure(401, DH901, "not-a-jwt")
    assert "could not be decoded" in d["message"]


def test_non_json_body_falls_back_to_generic():
    d = _diagnose_dhan_failure(502, "<html>Bad Gateway</html>", "x")
    assert d["reason"] == "other"
    assert "HTTP 502" in d["message"]


# ── fetch_dhan date handling (2026-11-07) ────────────────────────────────────
def test_dhan_daily_index_maps_ist_midnight_to_session_date():
    from data_fetcher import _dhan_daily_index
    # Real values from Dhan: 00:00 IST on 5 Nov and 6 Nov 2026 (= 18:30 UTC day before)
    idx = _dhan_daily_index([1793817000, 1793903400])
    assert [str(d.date()) for d in idx] == ["2026-11-05", "2026-11-06"]
    assert idx.tz is None and all(d.hour == 0 for d in idx)
