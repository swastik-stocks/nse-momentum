"""
NSE Momentum -- automated Dhan access token renewal.

[2026-08-21] Dhan access tokens have a hard 24h lifetime (SEBI/exchange
requirement, not a Dhan-specific limit -- see
https://dhanhq.co/docs/v2/authentication/). Previously this required a
human to manually regenerate the token at web.dhan.co every single day,
an easy thing to forget that silently breaks the evening scan and
checkpoints. Dhan's /v2/RenewToken endpoint extends an ACTIVE (not yet
expired) token by another 24h with a single scripted call and no browser
login -- called proactively once a day, well before the current token's
actual expiry, this can replace the manual step entirely.

Run standalone: python renew_dhan_token.py [--push-github-secret]
  - Always updates DHAN_ACCESS_TOKEN in the local .env on success.
  - --push-github-secret also pushes the renewed token to the
    DHAN_ACCESS_TOKEN GitHub Actions secret (needs GITHUB_TOKEN env var
    and GITHUB_REPO env var, e.g. "owner/name").

Intended to be called from a scheduled workflow BEFORE the current
token's expiry (RenewToken errors on an already-expired token -- this is
a refresh, not a resurrection). If a day is missed and the token lapses,
this falls back to needing one manual regeneration to restart the chain,
same as today's process -- this makes that the rare case, not the daily
default.
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from dhan_token_check import dhan_token_status

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

RENEW_URL = "https://api.dhan.co/v2/RenewToken"
ENV_PATH = Path(__file__).parent / ".env"


def renew_dhan_token(client_id: str, access_token: str) -> str | None:
    """Calls /v2/RenewToken. Returns the new token string, or None on
    failure (expired token, network error, bad credentials, etc.)."""
    try:
        resp = requests.get(
            RENEW_URL,
            headers={"access-token": access_token, "dhanClientId": client_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        log.error(f"RenewToken FAILED: HTTP {e.response.status_code if e.response is not None else '?'}: {body}")
        return None
    except Exception as e:
        log.error(f"RenewToken FAILED: {e}")
        return None

    new_token = data.get("accessToken") or data.get("access_token") or data.get("token")
    if not new_token:
        log.error(f"RenewToken response had no recognizable token field: {data}")
        return None
    return new_token


def update_env_token(new_token: str, env_path: Path = ENV_PATH) -> None:
    content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if "DHAN_ACCESS_TOKEN" in content:
        content = re.sub(r"DHAN_ACCESS_TOKEN=.*", f"DHAN_ACCESS_TOKEN={new_token}", content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"DHAN_ACCESS_TOKEN={new_token}\n"
    env_path.write_text(content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Proactively renew the Dhan access token before it expires.")
    ap.add_argument("--push-github-secret", action="store_true",
                     help="Also push the renewed token to the DHAN_ACCESS_TOKEN GitHub secret "
                          "(needs GITHUB_TOKEN and GITHUB_REPO env vars).")
    ap.add_argument("--env-path", default=str(ENV_PATH))
    args = ap.parse_args()

    env_path = Path(args.env_path)
    # override=True: an earlier load_dotenv() call elsewhere in the process
    # (e.g. telegram_alert's module-level load) may have already populated
    # os.environ from the default .env -- --env-path must win over that,
    # not silently no-op against whatever was loaded first.
    load_dotenv(env_path, override=True)

    client_id = os.environ.get("DHAN_CLIENT_ID", "")
    access_token = os.environ.get("DHAN_ACCESS_TOKEN", "")
    if not client_id or not access_token:
        log.error("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not found -- nothing to renew.")
        return 1

    status = dhan_token_status(access_token)
    if status["expired"]:
        log.error("Current token is already EXPIRED -- RenewToken cannot resurrect it. "
                   "Manual regeneration required at web.dhan.co > Profile > API > Generate Access Token.")
        return 1
    log.info(f"Current token has {status['hours_remaining']:.1f}h remaining -- renewing now.")

    new_token = renew_dhan_token(client_id, access_token)
    if not new_token:
        return 1

    update_env_token(new_token, env_path)
    new_status = dhan_token_status(new_token)
    log.info(f"Renewed. New token valid for {new_status['hours_remaining']:.1f}h "
             f"(until {new_status['expiry']}).")

    if args.push_github_secret:
        from github_secrets import push_github_secret
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        gh_repo = os.environ.get("GITHUB_REPO", "")
        if not gh_token or not gh_repo:
            log.error("--push-github-secret needs GITHUB_TOKEN and GITHUB_REPO env vars -- skipped.")
            return 1
        ok = push_github_secret(gh_repo, "DHAN_ACCESS_TOKEN", new_token, gh_token)
        log.info("GitHub secret DHAN_ACCESS_TOKEN updated." if ok else "GitHub secret push FAILED.")
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
