"""
Updates DHAN_ACCESS_TOKEN in .env and validates the connection.
Run via refresh_dhan_token.bat, or directly: python update_dhan_token.py
"""

import re
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

ENV_PATH = Path(".env")


def update_env_token(new_token: str) -> None:
    if ENV_PATH.exists():
        content = ENV_PATH.read_text(encoding="utf-8")
    else:
        content = ""

    if "DHAN_ACCESS_TOKEN" in content:
        content = re.sub(r"DHAN_ACCESS_TOKEN=.*", f"DHAN_ACCESS_TOKEN={new_token}", content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"DHAN_ACCESS_TOKEN={new_token}\n"

    ENV_PATH.write_text(content, encoding="utf-8")
    print("Token updated in .env successfully.")


def check_profile_status() -> None:
    """Diagnostic: hits Dhan's /v2/profile endpoint, which directly reports
    token validity and Data API subscription status -- much more informative
    than the generic quote-endpoint error."""
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    access_token = os.getenv("DHAN_ACCESS_TOKEN", "")

    req = urllib.request.Request(
        "https://api.dhan.co/v2/profile",
        headers={"access-token": access_token},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        print("DEBUG /v2/profile response:")
        print(json.dumps(data, indent=2))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"DEBUG /v2/profile FAILED: HTTP {e.code}: {body}")
    except Exception as e:
        print(f"DEBUG /v2/profile FAILED: {e}")


def test_connection() -> bool:
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)

    client_id = os.getenv("DHAN_CLIENT_ID", "")
    access_token = os.getenv("DHAN_ACCESS_TOKEN", "")

    if not client_id or not access_token:
        print("ERROR: Credentials missing from .env")
        return False

    payload = json.dumps({"NSE_EQ": [1333]}).encode()
    print(f"DEBUG sending client_id=[{client_id}] token_prefix=[{access_token[:20]}...]")
    req = urllib.request.Request(
        "https://api.dhan.co/v2/marketfeed/quote",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        nse = data.get("data", {}).get("NSE_EQ", {})
        if nse:
            ltp = list(nse.values())[0].get("last_price", "?")
            print(f"Dhan connection: OK  |  HDFC Bank LTP: Rs.{ltp}")
            return True
        else:
            print(f"Connected but no data returned: {data}")
            return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"Connection FAILED: HTTP {e.code}: {body}")
        return False
    except Exception as e:
        print(f"Connection FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print(" DHAN TOKEN REFRESH -- NSE Momentum v4.3")
    print("=" * 50)
    print()
    print("Step 1: Go to https://web.dhan.co")
    print("        Profile (top-right) > API > Generate Access Token")
    print("        Copy the NEW Access Token")
    print()

    new_token = input("Paste your new Dhan Access Token here and press Enter: ").strip()
    if not new_token:
        print("ERROR: No token entered. Exiting.")
        sys.exit(1)

    update_env_token(new_token)

    print()
    print("Checking profile / token / data-plan status...")
    print()
    check_profile_status()

    print()
    print("Validating Dhan connection...")
    print()
    ok = test_connection()

    print()
    if ok:
        print("Token refresh complete.")
    else:
        print("Token refresh saved to .env, but connection test failed -- check the error above.")
