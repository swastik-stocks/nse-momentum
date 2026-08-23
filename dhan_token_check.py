"""
NSE Momentum -- Dhan access token expiry check.

[2026-08-21] Discovered while wiring up the E: dev environment: Dhan access
tokens carry a hard 24h lifetime from issue time (the JWT's iat/exp claims
are exactly 86400 seconds apart) and must be regenerated manually every day
at web.dhan.co. Nothing in the pipeline previously distinguished "Dhan auth
expired" from any other fetch failure -- it just surfaced as generic 0%/low
coverage after the full 3-attempt retry loop in scanner.py, the same failure
signature as a genuine Dhan outage. This module decodes the token's own
stated expiry so callers can raise a specific, immediately actionable alert
instead of a 30-minute guessing game.

No signature verification -- we don't have (and don't need) Dhan's signing
key, we only need the token's own claim about when it expires.
"""
import base64
import json
from datetime import datetime, timezone
from typing import Optional, TypedDict


class DhanTokenStatus(TypedDict):
    expiry: Optional[datetime]       # tz-aware UTC, or None if undecodable
    expired: Optional[bool]          # None if undecodable
    hours_remaining: Optional[float] # None if undecodable


def decode_dhan_token_expiry(token: str) -> Optional[datetime]:
    """Decode the `exp` claim from a Dhan JWT access token. Returns a
    tz-aware UTC datetime, or None if the token is empty/malformed."""
    if not token:
        return None
    try:
        payload_b64 = token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    except Exception:
        return None


def dhan_token_status(token: str) -> DhanTokenStatus:
    expiry = decode_dhan_token_expiry(token)
    if expiry is None:
        return {"expiry": None, "expired": None, "hours_remaining": None}
    remaining_hours = (expiry - datetime.now(timezone.utc)).total_seconds() / 3600
    return {
        "expiry": expiry,
        "expired": remaining_hours <= 0,
        "hours_remaining": remaining_hours,
    }
