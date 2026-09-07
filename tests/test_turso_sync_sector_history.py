"""
[2026-09-07] Regression tests for turso_sync.py's sector_pick_history
support (publish_sector_history / get_recent_sectors) -- the durable
store behind sector_novelty.py's "NEW SECTOR" email badge.

These tests NEVER touch the real Turso database. get_client() is always
monkeypatched to return an in-memory fake that implements just enough of
libsql_client's execute()/close() interface to exercise the real SQL this
module sends. See tests/test_sector_novelty.py's module docstring for why
that matters (a real .env with live credentials sits on this dev machine,
and load_dotenv() will refill env vars that a test merely popped).
"""
import pytest

import turso_sync


class _FakeResultSet:
    def __init__(self, rows):
        self.rows = rows


class _FakeTursoClient:
    """In-memory stand-in for libsql_client covering exactly the
    statements turso_sync's sector-history functions send."""

    def __init__(self, raise_on_execute=None):
        self.executed = []
        self.closed = False
        self._rows = {}   # (scan_date, sector) -> published_at
        self._raise_on_execute = raise_on_execute

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if self._raise_on_execute and len(self.executed) >= self._raise_on_execute:
            raise RuntimeError("simulated Turso failure")

        norm = " ".join(sql.split())
        if norm.startswith("CREATE TABLE"):
            return _FakeResultSet([])
        if norm.startswith("INSERT INTO sector_pick_history"):
            scan_date, sector, published_at = params
            self._rows[(scan_date, sector)] = published_at
            return _FakeResultSet([])
        if norm.startswith("SELECT 1 FROM sector_pick_history WHERE scan_date <"):
            before_date = params[0]
            hits = [1 for (d, _s) in self._rows if d < before_date]
            return _FakeResultSet([(1,) for _ in hits[:1]])
        if norm.startswith("SELECT DISTINCT sector FROM sector_pick_history"):
            cutoff, before_date = params
            sectors = sorted({s for (d, s) in self._rows if cutoff <= d < before_date})
            return _FakeResultSet([(s,) for s in sectors])
        raise AssertionError(f"Unexpected SQL sent to fake Turso client: {sql!r}")

    def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeTursoClient()
    monkeypatch.setattr(turso_sync, "get_client", lambda: client)
    return client


def test_publish_sector_history_upserts_rows(fake_client):
    n = turso_sync.publish_sector_history({"Metals", "Pharma"}, scan_date="2026-09-04")
    assert n == 2
    assert ("2026-09-04", "Metals") in fake_client._rows
    assert ("2026-09-04", "Pharma") in fake_client._rows
    assert fake_client.closed is True


def test_publish_sector_history_dry_run_never_touches_client(monkeypatch):
    calls = []
    monkeypatch.setattr(turso_sync, "get_client", lambda: calls.append("called") or _FakeTursoClient())
    n = turso_sync.publish_sector_history({"Metals"}, scan_date="2026-09-04", dry_run=True)
    assert n == 0
    assert calls == []   # get_client() must never even be invoked on a dry run


def test_publish_sector_history_empty_set_is_a_noop(fake_client):
    n = turso_sync.publish_sector_history(set(), scan_date="2026-09-04")
    assert n == 0
    assert fake_client.executed == []


def test_get_recent_sectors_returns_prior_window_and_history_flag(fake_client):
    turso_sync.publish_sector_history({"Pharma"}, scan_date="2026-08-20")
    turso_sync.publish_sector_history({"Metals"}, scan_date="2026-08-25")

    prior, has_history = turso_sync.get_recent_sectors("2026-09-04", lookback_days=15)
    assert prior == {"Pharma", "Metals"}
    assert has_history is True


def test_get_recent_sectors_excludes_same_day_and_outside_window(fake_client):
    turso_sync.publish_sector_history({"OldSector"}, scan_date="2026-01-01")   # far outside window
    turso_sync.publish_sector_history({"TodaySector"}, scan_date="2026-09-04")  # same day as query

    prior, has_history = turso_sync.get_recent_sectors("2026-09-04", lookback_days=15)
    assert "OldSector" not in prior
    assert "TodaySector" not in prior     # same-day rows must not count as "prior"
    assert has_history is True            # OldSector row still proves history exists


def test_get_recent_sectors_no_history_at_all(fake_client):
    prior, has_history = turso_sync.get_recent_sectors("2026-09-04", lookback_days=15)
    assert prior == set()
    assert has_history is False


def test_get_client_unavailable_degrades_gracefully_not_raise(monkeypatch):
    """Simulates missing TURSO_DATABASE_URL/TURSO_AUTH_TOKEN -- get_client()
    calls sys.exit() in that case; both public functions must catch it and
    return benign defaults, never propagate."""
    def _raise_system_exit():
        raise SystemExit("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not found")
    monkeypatch.setattr(turso_sync, "get_client", _raise_system_exit)

    prior, has_history = turso_sync.get_recent_sectors("2026-09-04", lookback_days=15)
    assert (prior, has_history) == (set(), False)

    n = turso_sync.publish_sector_history({"Metals"}, scan_date="2026-09-04")
    assert n == 0


def test_get_recent_sectors_query_failure_closes_client_and_degrades(monkeypatch):
    """A mid-query exception (network blip, etc.) must still close the
    client and return the safe default, not leak a connection or raise
    past the caller (P1-06: never break the scan/email)."""
    client = _FakeTursoClient(raise_on_execute=2)   # fail on the 2nd execute() call
    monkeypatch.setattr(turso_sync, "get_client", lambda: client)

    prior, has_history = turso_sync.get_recent_sectors("2026-09-04", lookback_days=15)
    assert (prior, has_history) == (set(), False)
    assert client.closed is True


def test_publish_sector_history_partial_row_failure_continues(monkeypatch):
    """One bad sector name/row must not abort the whole publish -- matches
    publish_signals' per-row try/except pattern elsewhere in this file."""
    client = _FakeTursoClient()
    real_execute = client.execute
    calls = {"n": 0}

    def flaky_execute(sql, params=None):
        norm = " ".join(sql.split())
        if norm.startswith("INSERT"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated single-row failure")
        return real_execute(sql, params)

    monkeypatch.setattr(client, "execute", flaky_execute)
    monkeypatch.setattr(turso_sync, "get_client", lambda: client)

    n = turso_sync.publish_sector_history({"Metals", "Pharma"}, scan_date="2026-09-04")
    assert n == 1   # one of the two rows failed, the other still published
    assert client.closed is True
