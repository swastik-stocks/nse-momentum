"""
[2026-09-07] Regression tests for sector_novelty.py -- the "NEW SECTOR"
email badge added per user request (JSWSTEEL/JINDALSTEL, Metals theme,
04-Sep-2026 email).

Every test here forces turso_sync.TURSO_URL/TOKEN to None so mark_new_sectors()
takes the local-JSON fallback path and can NEVER reach the real production
Turso database, no matter what a real .env on the dev machine contains.
This isn't paranoia -- it happened once already while building this
feature (see [[project_new_sector_badge]] memory / commit history): naive
os.environ.pop() before import did NOT stop turso_sync's own
load_dotenv() from refilling the creds from .env, and a manual test run
wrote 6 rows into the live sector_pick_history table before being caught
and cleaned up. Monkeypatching the already-imported module's attributes
is the only reliable way to guarantee test isolation here.
"""
import os

import pytest

import turso_sync
import sector_novelty as sn


class _FakeResult:
    """Duck-typed stand-in for orchestrator.StockResult -- only .sector matters here."""
    def __init__(self, sector):
        self.sector = sector


@pytest.fixture(autouse=True)
def _no_real_turso(monkeypatch, tmp_path):
    """Force the local-JSON fallback path and redirect it into tmp_path so
    these tests can never touch the real Turso DB or the real repo's
    sector_pick_history.json."""
    monkeypatch.setattr(turso_sync, "TURSO_URL", None)
    monkeypatch.setattr(turso_sync, "TURSO_TOKEN", None)
    monkeypatch.setattr(sn, "SECTOR_HISTORY_LOCAL_FALLBACK",
                         str(tmp_path / "sector_pick_history.json"))
    yield


def test_first_ever_run_flags_nothing_new():
    """Day 1 with zero history: every sector would trivially be "new" --
    that's not a useful signal, so the bootstrap guard must suppress it."""
    results = [_FakeResult("Pharma"), _FakeResult("Financials")]
    new = sn.mark_new_sectors(results, [{"sector": "Auto"}], scan_date="2026-08-20")
    assert new == set()
    assert all(r.sector_is_new is False for r in results)


def test_sector_absent_within_lookback_window_is_flagged_new():
    """Reproduces the real scenario: Metals hasn't shown up in picks for
    weeks, then JSWSTEEL (T3) and JINDALSTEL (near-breakout) both appear
    on the same day -- both must be flagged, matching what the real email
    needed."""
    seed = [_FakeResult(s) for s in ["Pharma", "Financials", "Auto"]]
    sn.mark_new_sectors(seed, [], scan_date="2026-08-20")   # seeds history

    jswsteel = _FakeResult("Metals")
    other    = _FakeResult("Pharma")
    near_bo  = [{"sector": "Metals"}]   # JINDALSTEL-equivalent
    new = sn.mark_new_sectors([jswsteel, other], near_bo, scan_date="2026-09-04")

    assert new == {"Metals"}
    assert jswsteel.sector_is_new is True
    assert other.sector_is_new is False
    assert near_bo[0]["is_new_sector"] is True


def test_sector_seen_yesterday_is_not_flagged():
    sn.mark_new_sectors([_FakeResult("IT")], [], scan_date="2026-08-20")  # seed
    r = _FakeResult("IT")
    new = sn.mark_new_sectors([r], [], scan_date="2026-08-21")
    assert new == set()
    assert r.sector_is_new is False


def test_sector_outside_lookback_window_is_flagged_new_again():
    """A sector that rolled off the LOOKBACK_DAYS window counts as new
    again -- this is deliberate (the whole point is "hasn't shown up
    recently"), not a bug."""
    sn.mark_new_sectors([_FakeResult("Realty")], [], scan_date="2026-01-01")
    far_later = "2026-01-01"
    from datetime import date, timedelta
    beyond = (date.fromisoformat(far_later) +
              timedelta(days=sn.LOOKBACK_DAYS + 5)).isoformat()
    r = _FakeResult("Realty")
    new = sn.mark_new_sectors([r], [], scan_date=beyond)
    assert new == {"Realty"}
    assert r.sector_is_new is True


def test_near_breakout_dict_without_sector_key_does_not_crash():
    """near_breakout.py always includes 'sector', but mark_new_sectors
    must not assume that -- a malformed/legacy dict must degrade to
    is_new_sector=False, never KeyError."""
    nb = [{"ticker": "FOO.NS"}]
    new = sn.mark_new_sectors([], nb, scan_date="2026-08-20")
    assert new == set()
    assert nb[0]["is_new_sector"] is False


def test_result_without_sector_attribute_does_not_crash():
    class NoSector:
        pass
    r = NoSector()
    new = sn.mark_new_sectors([r], [], scan_date="2026-08-20")
    assert new == set()
    assert r.sector_is_new is False


def test_corrupt_local_history_file_is_treated_as_empty_not_fatal(tmp_path, monkeypatch):
    bad_path = tmp_path / "corrupt.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(sn, "SECTOR_HISTORY_LOCAL_FALLBACK", str(bad_path))
    r = _FakeResult("Pharma")
    new = sn.mark_new_sectors([r], [], scan_date="2026-08-20")   # must not raise
    assert new == set()   # corrupt file -> no history -> bootstrap guard
    assert r.sector_is_new is False


def test_same_day_rerun_is_idempotent_and_not_flagged_against_itself():
    """A same-day re-run (e.g. a retried scan) must never see its own
    earlier pass as "prior history" and flag everything as new."""
    r1 = _FakeResult("Metals")
    sn.mark_new_sectors([r1], [], scan_date="2026-09-04")
    r2 = _FakeResult("Metals")
    new = sn.mark_new_sectors([r2], [], scan_date="2026-09-04")
    assert new == set()
    assert r2.sector_is_new is False


def test_dry_run_flag_is_accepted_and_does_not_raise():
    """dry_run is forwarded from orchestrator.py's run_universe(); the
    local fallback path doesn't do anything different with it, but the
    signature must accept it without error (Turso path is covered
    separately in test_turso_sync_sector_history.py)."""
    r = _FakeResult("Pharma")
    new = sn.mark_new_sectors([r], [], scan_date="2026-08-20", dry_run=True)
    assert isinstance(new, set)
