"""
[2026-08-21] Regression tests for github_secrets.py -- extracted from the
one-off Telegram-secret wiring so renew_dhan_token.py can reuse it. All
GitHub API calls mocked; the encryption itself uses a real ephemeral NaCl
keypair so the test also proves the payload actually decrypts back to the
original secret value, not just that requests.put got called.
"""
import base64
from unittest.mock import MagicMock, patch

from nacl import encoding, public

from github_secrets import push_github_secret


def test_push_github_secret_encrypts_and_puts_correct_value():
    keypair = public.PrivateKey.generate()
    pub_b64 = keypair.public_key.encode(encoding.Base64Encoder()).decode("utf-8")

    key_resp = MagicMock()
    key_resp.raise_for_status = MagicMock()
    key_resp.json.return_value = {"key": pub_b64, "key_id": "test-key-id"}

    put_resp = MagicMock()
    put_resp.raise_for_status = MagicMock()

    with patch("github_secrets.requests.get", return_value=key_resp), \
         patch("github_secrets.requests.put", return_value=put_resp) as mock_put:
        ok = push_github_secret("owner/repo", "DHAN_ACCESS_TOKEN", "secret-value-123", "gh-token")

    assert ok is True
    put_args, put_kwargs = mock_put.call_args
    assert put_args[0] == "https://api.github.com/repos/owner/repo/actions/secrets/DHAN_ACCESS_TOKEN"
    assert put_kwargs["json"]["key_id"] == "test-key-id"

    sealed_box = public.SealedBox(keypair)
    decrypted = sealed_box.decrypt(base64.b64decode(put_kwargs["json"]["encrypted_value"]))
    assert decrypted.decode("utf-8") == "secret-value-123"


def test_push_github_secret_returns_false_on_key_fetch_failure():
    with patch("github_secrets.requests.get", side_effect=ConnectionError("down")):
        ok = push_github_secret("owner/repo", "DHAN_ACCESS_TOKEN", "value", "gh-token")
    assert ok is False


def test_push_github_secret_returns_false_on_put_failure():
    keypair = public.PrivateKey.generate()
    pub_b64 = keypair.public_key.encode(encoding.Base64Encoder()).decode("utf-8")
    key_resp = MagicMock()
    key_resp.raise_for_status = MagicMock()
    key_resp.json.return_value = {"key": pub_b64, "key_id": "test-key-id"}

    with patch("github_secrets.requests.get", return_value=key_resp), \
         patch("github_secrets.requests.put", side_effect=ConnectionError("down")):
        ok = push_github_secret("owner/repo", "DHAN_ACCESS_TOKEN", "value", "gh-token")
    assert ok is False
