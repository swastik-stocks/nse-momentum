"""
Semi-automated Fyers login -- run this once each morning (or whenever
your cached token expires) before the scanner/confirmation jobs need
fresh data.

Why this exists: Fyers' headless TOTP+PIN login endpoint
(send_login_otp_v2) is currently rejecting all requests with a
-1025 "invalid request" error, even with correct credentials and a
byte-identical payload to the known-working community reference.
This looks like a new anti-automation check on Fyers' side. Rather
than fight that, this script uses Fyers' own OFFICIAL auth-code flow,
which just needs you to log in once in a real browser.

Usage:
    python fyers_manual_login.py

It will:
  1. Print a login URL.
  2. You open it, log into Fyers normally (TOTP/PIN as usual), and
     get redirected to your Redirect URI -- the resulting page may
     look blank/broken, that's expected. What matters is the URL
     in your browser's address bar.
  3. Paste that FULL redirect URL back into this script when prompted.
  4. It extracts the auth_code, exchanges it for an access token, and
     saves it to the same local cache file fyers_auth.py reads from.
  5. It prints the token so you can copy it into a GitHub Actions
     secret (FYERS_ACCESS_TOKEN) for the automated pipeline to use.

Requires the same env vars as fyers_auth.py:
  FYERS_CLIENT_ID, FYERS_SECRET_KEY, FYERS_REDIRECT_URI
"""

import json
import os
from datetime import date
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from fyers_apiv3 import fyersModel

TOKEN_CACHE_PATH = Path(os.environ.get("FYERS_TOKEN_CACHE", ".fyers_token_cache.json"))


def _save_token_cache(access_token: str) -> None:
    TOKEN_CACHE_PATH.write_text(json.dumps({
        "date": date.today().isoformat(),
        "access_token": access_token,
    }))


def main():
    client_id = os.environ["FYERS_CLIENT_ID"]
    secret_key = os.environ["FYERS_SECRET_KEY"]
    redirect_uri = os.environ["FYERS_REDIRECT_URI"]

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        state="sample_state",
        grant_type="authorization_code",
    )

    login_url = session.generate_authcode()

    print("\n" + "=" * 70)
    print("STEP 1: Open this URL in your browser and log in to Fyers:")
    print(login_url)
    print("=" * 70)
    print("\nAfter logging in, you'll be redirected to your Redirect URI.")
    print("The page itself may look blank or show an error -- that's fine.")
    print("What matters is copying the FULL URL from your browser's address bar.")
    print("It will look something like:")
    print(f"  {redirect_uri}?s=ok&code=200&auth_code=XXXXXXXX&state=sample_state\n")

    redirected_url = input("STEP 2: Paste the full redirected URL OR just the authorization code:\n> ").strip()

    if redirected_url.startswith("http"):
        parsed = urlparse(redirected_url)
        query = parse_qs(parsed.query)
        if "auth_code" not in query:
            print("\nCouldn't find 'auth_code' in that URL. Make sure you pasted the")
            print("complete address-bar URL after being redirected, not the login page URL.")
            return
        auth_code = query["auth_code"][0]
    else:
        # Assume they pasted the raw auth_code value directly
        # (e.g. copied from the "authorization code" field / Copy button)
        auth_code = redirected_url

    session.set_token(auth_code)
    response = session.generate_token()

    if "access_token" not in response:
        print("\nToken exchange failed. Full response from Fyers:")
        print(json.dumps(response, indent=2))
        return

    access_token = response["access_token"]
    _save_token_cache(access_token)

    print("\n" + "=" * 70)
    print("SUCCESS -- token saved locally to:", TOKEN_CACHE_PATH.resolve())
    print("=" * 70)
    print("\nSTEP 3: Push this token to GitHub so your Actions workflows can use it.")
    print("Copy the value below, then either:")
    print("  a) Run: gh secret set FYERS_ACCESS_TOKEN --body \"<paste_token>\"")
    print("  b) Or go to GitHub repo -> Settings -> Secrets and variables -> Actions")
    print("     -> FYERS_ACCESS_TOKEN -> Update, and paste it there.\n")
    print("ACCESS TOKEN:")
    print(access_token)
    print()

    # Quick smoke test using the token right now
    fy = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
    profile = fy.get_profile()
    print("Profile check:", json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
