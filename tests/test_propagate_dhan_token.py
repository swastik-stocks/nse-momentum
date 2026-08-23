"""
[2026-08-23] Regression test for propagate_dhan_token.py's _write_token --
this touches F:'s live .env, so a regex mistake here is a real production
risk, not just a cosmetic bug. Verified against a temp file, never a real
.env.
"""
from pathlib import Path

from propagate_dhan_token import _write_token


def test_write_token_replaces_only_the_dhan_line(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DHAN_CLIENT_ID=1234567890\n"
        "DHAN_ACCESS_TOKEN=old-stale-token-value\n"
        "TELEGRAM_BOT_TOKEN=some-other-secret\n",
        encoding="utf-8",
    )

    ok = _write_token(env_file, "brand-new-token-value")

    assert ok is True
    content = env_file.read_text(encoding="utf-8")
    assert "DHAN_ACCESS_TOKEN=brand-new-token-value" in content
    assert "old-stale-token-value" not in content
    assert "DHAN_CLIENT_ID=1234567890" in content          # untouched
    assert "TELEGRAM_BOT_TOKEN=some-other-secret" in content  # untouched


def test_write_token_dry_run_does_not_modify_file(tmp_path):
    env_file = tmp_path / ".env"
    original = "DHAN_ACCESS_TOKEN=old-token\n"
    env_file.write_text(original, encoding="utf-8")

    ok = _write_token(env_file, "would-be-new-token", dry_run=True)

    assert ok is True
    assert env_file.read_text(encoding="utf-8") == original


def test_write_token_missing_file_returns_false(tmp_path):
    missing = tmp_path / "does_not_exist.env"
    assert _write_token(missing, "token") is False


def test_write_token_missing_dhan_line_returns_false(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_OTHER_VAR=value\n", encoding="utf-8")
    assert _write_token(env_file, "token") is False
