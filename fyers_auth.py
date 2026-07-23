"""
Fyers client getter, updated for the semi-automated login flow.

Priority order for getting a working token:
  1. FYERS_ACCESS_TOKEN environment variable (set as a GitHub Actions
     secret after running fyers_manual_login.py locally each morning).
  2. Local same-day cached token file (.fyers_token_cache.json) --
     useful when running scanner.py/confirm_picks.py locally right
     after fyers_manual_login.py.
  3. Last resort: attempt the headless TOTP+PIN login. As of writing,
     Fyers' send_login_otp_v2 endpoint is rejecting all requests with
     a -1025 error regardless of correct credentials -- this path is
     kept in case Fyers fixes it, but don't rely on it for now.

If none of these produce a token, this raises a clear error telling
you to run fyers_manual_login.py.

Requires: pip install fyers-apiv3 pyotp requests --break-system-packages
"""

import base64
import json
import os
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from fyers_apiv3 import fyersModel

TOKEN_CACHE_PATH = Path(os.environ.get("FYERS_TOKEN_CACHE", ".fyers_token_cache.json"))

BROWSER_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    ),
}


def _b64(value) -> str:
    return base64.b64encode(str(value).encode("ascii")).decode("ascii")


def _load_cached_token() -> str | None:
    if not TOKEN_CACHE_PATH.exists():
        return None
    cache = json.loads(TOKEN_CACHE_PATH.read_text())
    if cache.get("date") == date.today().isoformat():
        return cache.get("access_token")
    return None  # cache is from a previous day -- needs refresh


def _save_token_cache(access_token: str) -> None:
    TOKEN_CACHE_PATH.write_text(json.dumps({
        "date": date.today().isoformat(),
        "access_token": access_token,
    }))


def _generate_access_token_headless() -> str:
    """
    Headless TOTP+PIN login. Currently broken on Fyers' side (send_login_otp_v2
    returns -1025 invalid request regardless of credential correctness) --
    kept here in case Fyers fixes it, but get_fyers_client() no longer
    calls this automatically. Use fyers_manual_login.py instead.
    """
    client_id = os.environ["FYERS_CLIENT_ID"]
    secret_key = os.environ["FYERS_SECRET_KEY"]
    redirect_uri = os.environ["FYERS_REDIRECT_URI"]
    totp_key = os.environ["FYERS_TOTP_KEY"]
    pin = os.environ["FYERS_PIN"]
    fy_id = os.environ["FYERS_ID"]

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    otp = pyotp.TOTP(totp_key).now()
    r1 = session.post(
        "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2",
        json={"fy_id": _b64(fy_id), "app_id": "2"},
    )
    request_key = r1.json()["request_key"]

    r2 = session.post(
        "https://api-t2.fyers.in/vagator/v2/verify_otp",
        json={"request_key": request_key, "otp": otp},
    )
    assert r2.status_code == 200, f"Error in r2: {r2.text}"
    request_key2 = r2.json()["request_key"]

    r3 = session.post(
        "https://api-t2.fyers.in/vagator/v2/verify_pin_v2",
        json={"request_key": request_key2, "identity_type": "pin", "identifier": _b64(pin)},
    )
    assert r3.status_code == 200, f"Error in r3: {r3.json()}"
    access_token_step2 = r3.json()["data"]["access_token"]

    auth_headers = {
        "authorization": f"Bearer {access_token_step2}",
        "content-type": "application/json; charset=UTF-8",
    }
    r4 = session.post(
        "https://api.fyers.in/api/v2/token",
        headers=auth_headers,
        json={
            "fyers_id": fy_id,
            "app_id": client_id.split("-")[0],
            "redirect_uri": redirect_uri,
            "appType": client_id.split("-")[1],
            "code_challenge": "",
            "state": "sample_state",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        },
    )
    assert r4.status_code == 308, f"Error in r4: {r4.json()}"

    parsed = urlparse(r4.json()["Url"])
    auth_code = parse_qs(parsed.query)["auth_code"][0]

    session_model = fyersModel.SessionModel(
        client_id=client_id, secret_key=secret_key,
        redirect_uri=redirect_uri, response_type="code",
        grant_type="authorization_code",
    )
    session_model.set_token(auth_code)
    response = session_model.generate_token()
    return response["access_token"]


def get_fyers_client() -> fyersModel.FyersModel:
    """Main entry point -- call this from scanner.py / confirm_picks.py."""
    # 1. GitHub Actions secret (pushed manually each morning via fyers_manual_login.py)
    token = os.environ.get("FYERS_ACCESS_TOKEN")

    # 2. Local same-day cache
    if not token:
        token = _load_cached_token()

    # 3. Nothing available -- do NOT silently attempt the broken headless flow
    if not token:
        raise RuntimeError(
            "No valid Fyers access token found.\n"
            "Run `python fyers_manual_login.py` locally to log in and generate one,\n"
            "then set it as the FYERS_ACCESS_TOKEN secret in GitHub Actions.\n"
            "(The fully headless TOTP login is currently blocked by Fyers -- see\n"
            "_generate_access_token_headless() docstring for details.)"
        )

    return fyersModel.FyersModel(
        client_id=os.environ["FYERS_CLIENT_ID"],
        is_async=False,
        token=token,
        log_path="",
    )


if __name__ == "__main__":
    # Quick smoke test: confirms auth works and prints your profile
    fy = get_fyers_client()
    profile = fy.get_profile()
    print(json.dumps(profile, indent=2))
