"""Cloud Dhan token auto-renewal (2026-11-17): store, decision flow, failure paths."""
import base64
import json
import sqlite3
import time as _time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import dhan_token_autorenew as ar
import dhan_token_store as store

KEY = store.generate_key()


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(store.KEY_ENV, KEY)


def jwt(hours_left: float, tag: str = "x") -> str:
    """Fake Dhan JWT with an exp claim `hours_left` from now (tag keeps tokens distinct)."""
    exp = int((datetime.now(timezone.utc) + timedelta(hours=hours_left)).timestamp())
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp, "t": tag}).encode()).decode().rstrip("=")
    return f"hdr.{body}.sig"


class FakeTurso:
    """libsql_client-shaped wrapper over in-memory sqlite."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
    def execute(self, sql, params=()):
        cur = self.db.execute(sql, params)
        self.db.commit()
        return SimpleNamespace(rows=cur.fetchall())
    def close(self):
        pass


# ── token store ──────────────────────────────────────────────────────────────

def test_store_roundtrip_is_encrypted():
    c = FakeTurso()
    tok = jwt(20)
    assert store.publish_token(tok, "test", client=c)
    raw = c.execute("SELECT ciphertext FROM dhan_token_store").rows[0][0]
    assert tok not in raw and "hdr." not in raw          # never stored in clear
    assert store.fetch_token(client=c)["token"] == tok


def test_store_newer_wins():
    c = FakeTurso()
    fresh, stale = jwt(23, "fresh"), jwt(5, "stale")
    store.publish_token(fresh, "cloud", client=c)
    assert store.publish_token(stale, "laptop", client=c) is False
    assert store.fetch_token(client=c)["token"] == fresh


def test_wrong_key_is_a_clear_error(monkeypatch):
    c = FakeTurso()
    store.publish_token(jwt(10), "t", client=c)
    monkeypatch.setenv(store.KEY_ENV, store.generate_key())
    with pytest.raises(store.TokenStoreError, match="wrong"):
        store.fetch_token(client=c)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(store.KEY_ENV)
    with pytest.raises(store.TokenStoreError, match=store.KEY_ENV):
        store.encrypt("x")


# ── renewal decision flow ────────────────────────────────────────────────────

class FakeDeps:
    def __init__(self, stored=None, renew_result="NEW", busy=None,
                 store_ok=True, secret_ok=True, store_raises=False):
        self.stored, self.renew_result = stored, renew_result
        self.busy = list(busy or [])
        self.store_ok, self.secret_ok, self.store_raises = store_ok, secret_ok, store_raises
        self.renew_calls, self.published, self.pushed, self.rescued = [], [], [], []
        self.sleep = lambda s: None
        self.store = store
    def fetch_store(self):
        if self.store_raises:
            raise store.TokenStoreError("turso down")
        return {"token": self.stored} if self.stored else None
    def publish_store(self, t):
        self.published.append(t); return self.store_ok
    def push_secret(self, t):
        self.pushed.append(t); return self.secret_ok
    def renew(self, cid, t):
        self.renew_calls.append(t)
        return self.renew_result if self.renew_result != "NEW" else jwt(24, "renewed")
    def busy_runs(self):
        return self.busy.pop(0) if self.busy else []
    def write_rescue(self, t):
        self.rescued.append(t); return True


@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(ar, "_alert", lambda m: sent.append(m))
    return sent


def test_fresh_token_skips_without_calling_dhan(alerts):
    tok = jwt(23)
    d = FakeDeps(stored=tok)
    assert ar.run("cid", tok, d) == 0
    assert d.renew_calls == [] and not alerts


def test_renews_when_due_and_saves_store_before_secret(alerts):
    tok = jwt(14)
    d = FakeDeps(stored=tok)
    assert ar.run("cid", tok, d) == 0
    assert d.renew_calls == [tok]
    assert d.published and d.pushed and d.published[-1] == d.pushed[-1] != tok
    assert not alerts


def test_uses_newest_copy_and_self_heals_stale_secret(alerts):
    """A previous run renewed but the secret push failed: the store holds the live token."""
    stale_secret, live = jwt(2, "old"), jwt(23, "live")
    d = FakeDeps(stored=live)
    assert ar.run("cid", stale_secret, d) == 0
    assert d.pushed == [live]            # secret healed
    assert d.renew_calls == []           # still fresh -> no renewal


def test_turso_down_falls_back_to_secret(alerts):
    tok = jwt(10)
    d = FakeDeps(store_raises=True)
    assert ar.run("cid", tok, d) == 0
    assert d.renew_calls == [tok]


def test_expired_everywhere_alerts_with_paste_instructions(alerts):
    d = FakeDeps(stored=jwt(-1, "a"))
    assert ar.run("cid", jwt(-2, "b"), d) == 1
    assert "--paste" in alerts[0] and d.renew_calls == []


def test_waits_for_running_scan_then_renews(alerts):
    tok = jwt(10)
    d = FakeDeps(stored=tok, busy=[["daily_scan.yml #1"], ["daily_scan.yml #1"]])
    assert ar.run("cid", tok, d) == 0
    assert len(d.renew_calls) == 1


def test_still_busy_skips_if_token_has_time(alerts, monkeypatch):
    monkeypatch.setattr(ar, "BUSY_WAIT_MAX_MIN", 2)
    tok = jwt(10)
    d = FakeDeps(stored=tok, busy=[["scan"]] * 10)
    assert ar.run("cid", tok, d) == 0
    assert d.renew_calls == []


def test_still_busy_renews_anyway_when_nearly_expired(alerts, monkeypatch):
    monkeypatch.setattr(ar, "BUSY_WAIT_MAX_MIN", 2)
    tok = jwt(1)
    d = FakeDeps(stored=tok, busy=[["scan"]] * 10)
    assert ar.run("cid", tok, d) == 0
    assert d.renew_calls == [tok]


def test_renew_failure_alerts_and_keeps_old_token(alerts):
    tok = jwt(10)
    d = FakeDeps(stored=tok, renew_result=None)
    assert ar.run("cid", tok, d) == 1
    assert len(d.renew_calls) == len(ar.RENEW_RETRY_DELAYS)
    assert "RenewToken FAILED" in alerts[0]


def test_both_saves_failing_writes_encrypted_rescue(alerts):
    tok = jwt(10)
    d = FakeDeps(stored=tok, store_ok=False, secret_ok=False)
    assert ar.run("cid", tok, d) == 1
    assert len(d.rescued) == 1 and "CRITICAL" in alerts[-1]


def test_one_save_failing_alerts_but_no_rescue(alerts):
    tok = jwt(10)
    d = FakeDeps(stored=tok, secret_ok=False)
    assert ar.run("cid", tok, d) == 1
    assert d.rescued == [] and "GitHub secret save failed" in alerts[-1]


def test_dry_run_never_calls_dhan_or_writes(alerts):
    tok = jwt(10)
    d = FakeDeps(stored=jwt(5, "other"))
    assert ar.run("cid", tok, d, dry_run=True) == 0
    assert d.renew_calls == d.published == d.pushed == []


def test_force_renews_fresh_token(alerts):
    tok = jwt(23)
    d = FakeDeps(stored=tok)
    assert ar.run("cid", tok, d, force=True) == 0
    assert d.renew_calls == [tok]


def test_mask_emitted_only_in_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    ar._mask("SECRETVALUE")
    assert "::add-mask::SECRETVALUE" in capsys.readouterr().out
    monkeypatch.delenv("GITHUB_ACTIONS")
    ar._mask("SECRETVALUE")
    assert capsys.readouterr().out == ""


# ── workflow wiring (plain text: CI has no PyYAML) ───────────────────────────

def test_workflow_runs_every_day_with_backups_and_concurrency():
    import re
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/renew_dhan_token.yml").read_text(encoding="utf-8")
    crons = re.findall(r"- cron: '([^']+)'", wf)
    assert crons == ["30 2 * * *", "30 12 * * *"]          # every day, incl. weekends
    assert "group: dhan-token-renewal" in wf and "cancel-in-progress: false" in wf
    for secret in ("DHAN_TOKEN_STORE_KEY", "REPO_ADMIN_PAT", "TURSO_AUTH_TOKEN"):
        assert f"secrets.{secret}" in wf
