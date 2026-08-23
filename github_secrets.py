"""
NSE Momentum -- push a value into a GitHub Actions repo secret via the REST
API. Extracted 2026-08-21 from the one-off approach used to wire up the
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID secrets, so the Dhan auto-renewal
workflow can reuse it instead of re-deriving the encryption dance.

GitHub requires secret values to be encrypted client-side with the repo's
current public key before the PUT -- see
https://docs.github.com/en/rest/actions/secrets.
"""
import base64
import logging

import requests
from nacl import encoding, public

log = logging.getLogger(__name__)


def push_github_secret(repo: str, secret_name: str, secret_value: str, github_token: str) -> bool:
    """
    repo: "owner/name"
    Returns True on success, False on any failure (never raises) -- same
    fail-soft contract as telegram_alert.send_telegram_alert, since this is
    a side effect of token renewal, not something that should crash the
    renewal run if GitHub's API hiccups.
    """
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        key_resp = requests.get(
            f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
            headers=headers, timeout=15,
        )
        key_resp.raise_for_status()
        key_data = key_resp.json()

        public_key = public.PublicKey(key_data["key"].encode("utf-8"), encoding.Base64Encoder())
        sealed_box = public.SealedBox(public_key)
        encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
        encrypted_value = base64.b64encode(encrypted).decode("utf-8")

        put_resp = requests.put(
            f"https://api.github.com/repos/{repo}/actions/secrets/{secret_name}",
            headers=headers,
            json={"encrypted_value": encrypted_value, "key_id": key_data["key_id"]},
            timeout=15,
        )
        put_resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"Failed to push GitHub secret {secret_name}: {e}")
        return False
