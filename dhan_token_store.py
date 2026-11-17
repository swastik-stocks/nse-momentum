r"""
Encrypted, shared "current Dhan token" record in Turso.

WHY THIS EXISTS (2026-11-17)
    Dhan's /v2/RenewToken EXPIRES the token it is called with and returns a
    new one (https://dhanhq.co/docs/v2/authentication/). Once renewal runs in
    the cloud (renew_dhan_token.yml), every other copy of the token -- the
    laptop's F:\ and E:\ .env files -- dies at that moment. GitHub Actions
    secrets are write-only, so the laptop cannot read the new value back from
    there. Turso is the one store both GitHub Actions and the laptop can read
    and write (already used for scanner_signals / sector_pick_history), so it
    becomes the canonical "latest token" record.

WHY ENCRYPTED
    The repo is public and the Turso database is also reachable by the
    Portfolio Dashboard. The token is sealed with a NaCl SecretBox (PyNaCl is
    already a dependency via github_secrets.py) under DHAN_TOKEN_STORE_KEY,
    which lives only in the GitHub secret of the same name and the laptop's
    user environment -- a Turso read alone does not yield a usable token.

NEWER-WINS
    publish_token() never replaces the stored token with one that expires
    EARLIER than what is already stored, so a stale laptop --sync cannot
    clobber a fresher cloud-renewed token (and vice versa).
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dhan_token_check import dhan_token_status

log = logging.getLogger(__name__)

KEY_ENV = "DHAN_TOKEN_STORE_KEY"
TABLE = "dhan_token_store"
ROW_ID = "current"


class TokenStoreError(RuntimeError):
    pass


# ── crypto ───────────────────────────────────────────────────────────────────

def generate_key() -> str:
    """New random 32-byte key, base64 -- for one-time setup only."""
    from nacl.utils import random
    from nacl.secret import SecretBox
    return base64.b64encode(random(SecretBox.KEY_SIZE)).decode()


def _box(key_b64: Optional[str] = None):
    from nacl.secret import SecretBox
    key_b64 = key_b64 or os.environ.get(KEY_ENV, "")
    if not key_b64:
        raise TokenStoreError(f"{KEY_ENV} is not set -- cannot encrypt/decrypt the shared token")
    try:
        key = base64.b64decode(key_b64)
    except Exception as e:
        raise TokenStoreError(f"{KEY_ENV} is not valid base64: {e}") from None
    if len(key) != SecretBox.KEY_SIZE:
        raise TokenStoreError(f"{KEY_ENV} must decode to {SecretBox.KEY_SIZE} bytes, got {len(key)}")
    return SecretBox(key)


def encrypt(token: str, key_b64: Optional[str] = None) -> str:
    return base64.b64encode(_box(key_b64).encrypt(token.encode())).decode()


def decrypt(ciphertext_b64: str, key_b64: Optional[str] = None) -> str:
    from nacl.exceptions import CryptoError
    try:
        return _box(key_b64).decrypt(base64.b64decode(ciphertext_b64)).decode()
    except CryptoError:
        raise TokenStoreError("stored token could not be decrypted -- wrong "
                              f"{KEY_ENV} on this machine?") from None


# ── Turso ────────────────────────────────────────────────────────────────────

def _client():
    import turso_sync   # lazy: needs libsql_client + TURSO_* creds
    try:
        return turso_sync.get_client()
    except SystemExit as e:   # turso_sync.get_client() sys.exit()s when creds are missing
        raise TokenStoreError(str(e)) from None


def _ensure_table(client) -> None:
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id          TEXT PRIMARY KEY,
            ciphertext  TEXT NOT NULL,
            expiry_utc  TEXT NOT NULL,
            source      TEXT,
            updated_at  TEXT NOT NULL
        )
    """)


def fetch_token(client=None) -> Optional[dict]:
    """{"token", "expiry" (aware UTC datetime), "source", "updated_at"} or
    None if nothing is stored yet. Raises TokenStoreError on config/crypto
    problems so callers can tell "empty" from "broken"."""
    own = client is None
    client = client or _client()
    try:
        _ensure_table(client)
        rs = client.execute(f"SELECT ciphertext, expiry_utc, source, updated_at "
                            f"FROM {TABLE} WHERE id = ?", [ROW_ID])
        if not rs.rows:
            return None
        ct, expiry, source, updated_at = rs.rows[0]
        return {"token": decrypt(ct), "expiry": datetime.fromisoformat(expiry),
                "source": source, "updated_at": updated_at}
    finally:
        if own:
            client.close()


def publish_token(token: str, source: str, client=None) -> bool:
    """Store `token` as the shared current token unless the stored one
    expires LATER (newer-wins). Returns True if the store now holds `token`
    (written, or already identical). Raises TokenStoreError on failure."""
    expiry = dhan_token_status(token)["expiry"]
    if expiry is None:
        raise TokenStoreError("token has no decodable expiry -- refusing to store it")
    own = client is None
    client = client or _client()
    try:
        _ensure_table(client)
        current = fetch_token(client)
        if current and current["token"] == token:
            return True
        if current and current["expiry"] > expiry:
            log.warning(f"  Token store already holds a NEWER token (expires {current['expiry']}, "
                        f"from {current['source']}) -- not overwriting it with one expiring {expiry}.")
            return False
        client.execute(
            f"INSERT INTO {TABLE} (id, ciphertext, expiry_utc, source, updated_at) "
            f"VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET ciphertext=excluded.ciphertext, "
            f"expiry_utc=excluded.expiry_utc, source=excluded.source, updated_at=excluded.updated_at",
            [ROW_ID, encrypt(token), expiry.isoformat(), source,
             datetime.now(timezone.utc).isoformat()],
        )
        return True
    finally:
        if own:
            client.close()
