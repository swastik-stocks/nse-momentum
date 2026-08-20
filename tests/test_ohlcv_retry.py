"""
[2026-08-20] Regression tests for the OHLCV-fetch retry-with-backoff added
to scanner.py's run_scan() after the 2026-08-19 evening incident: the cloud
runner saw 0/503 (0%) fresh Dhan coverage at BOTH the original 19:30 IST
trigger and a retimed 20:15 IST trigger, while a local fetch in between the
two succeeded (502/503). Moving the trigger later didn't fix it, so the
fetch itself now retries (up to 3 attempts, 10 min apart) before the scan
gives up and sends a HELD alert -- these tests pin that behaviour so a
future refactor can't silently turn it back into a single-shot check.

Integration-style: run_scan() pulls in a lot (Bhavcopy, holdings, the agent
pipeline, email), so everything except the fetch/retry loop itself is
mocked out. No live network, no real email, no real sleep.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import scanner
import turso_sync
from market_calendar.staleness_check import DataProvenance


def _fresh_df(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100},
        index=idx,
    )


@pytest.fixture
def _bhavcopy_ok(monkeypatch):
    fake_bhav = MagicMock()
    fake_bhav.get_delivery_pct.return_value = (
        {},
        DataProvenance(source_name="bhavcopy", ok=True, reason="ok",
                        actual_date=date(2026, 8, 19)),
    )
    fake_bhav.full_df = None
    fake_bhav.bhavcopy_cmp_map = {}
    monkeypatch.setattr(scanner, "BhavcopyFetcher", lambda: fake_bhav)
    monkeypatch.setattr(scanner, "get_last_trading_day", lambda: date(2026, 8, 19))
    monkeypatch.setattr(scanner, "get_market_context",
                         lambda: {"nifty50": pd.DataFrame(), "banknifty": pd.DataFrame(),
                                   "nifty500": pd.DataFrame(), "vix": 15.0,
                                   "vix_is_fallback": False})
    monkeypatch.setattr(scanner, "get_code_provenance",
                         lambda: {"is_cloud_runner": False, "git_dirty": False,
                                   "git_commit": "abc123", "git_commit_ts": None,
                                   "git_branch": "main", "runner": "test"})
    monkeypatch.setattr(turso_sync, "get_holdings", lambda: {})
    monkeypatch.setattr(scanner, "init_tables", lambda: None)
    monkeypatch.setattr(scanner.time, "sleep", MagicMock())
    return fake_bhav


def test_retry_exhausted_all_attempts_holds_scan_and_alerts_once(_bhavcopy_ok, monkeypatch):
    """Coverage stays below the floor on every attempt -> 3 fetch attempts,
    2 sleeps (not after the last, failed attempt), one HELD alert email,
    run_scan() returns None without ever reaching the agent pipeline."""
    fetch_calls = []

    def fake_fetch(tickers, period="2y"):
        fetch_calls.append(1)
        return {t: _fresh_df() for t in tickers}  # "loaded" but every bar is stale

    monkeypatch.setattr(scanner, "fetch_batch_ohlcv", fake_fetch)
    monkeypatch.setattr(scanner, "find_stale_tickers",
                         lambda stock_data, expected_date: list(stock_data.keys()))

    alert_mock = MagicMock()
    monkeypatch.setattr(scanner, "send_data_freshness_alert_email", alert_mock)

    orchestrator_mock = MagicMock()
    monkeypatch.setattr(scanner, "AgentOrchestrator", orchestrator_mock)

    result = scanner.run_scan(dry_run=False, max_tickers=3)

    assert result is None
    assert len(fetch_calls) == 3, "must retry twice (3 total attempts) before giving up"
    assert scanner.time.sleep.call_count == 2, "must sleep between attempts, not after the last failed one"
    alert_mock.assert_called_once()
    reason = alert_mock.call_args[0][0][0]
    assert "3 attempts" in reason
    assert not orchestrator_mock.called, "must never reach the agent pipeline on a HELD scan"


def test_coverage_recovers_on_second_attempt_stops_retrying(_bhavcopy_ok, monkeypatch):
    """First attempt is stale, second attempt is fresh -> exactly 2 fetch
    attempts, exactly 1 sleep, no alert email, and the scan proceeds
    normally into the (mocked) agent pipeline."""
    attempts = {"n": 0}

    def fake_fetch(tickers, period="2y"):
        attempts["n"] += 1
        return {t: _fresh_df() for t in tickers}

    def fake_find_stale(stock_data, expected_date):
        # stale on attempt 1, all fresh from attempt 2 onward
        return list(stock_data.keys()) if attempts["n"] == 1 else []

    monkeypatch.setattr(scanner, "fetch_batch_ohlcv", fake_fetch)
    monkeypatch.setattr(scanner, "find_stale_tickers", fake_find_stale)

    alert_mock = MagicMock()
    monkeypatch.setattr(scanner, "send_data_freshness_alert_email", alert_mock)

    fake_orc = MagicMock()
    fake_orc.regime = "D"
    fake_orc.regime_name = "Correction"
    fake_orc.breadth_score = 4
    fake_orc.run_universe.return_value = {
        "tier1": [], "tier2": [], "tier3": [], "all_results": [],
        "regime": "D", "regime_name": "Correction", "breadth": 4,
    }
    monkeypatch.setattr(scanner, "AgentOrchestrator", lambda data_dict: fake_orc)
    monkeypatch.setattr(scanner, "send_email_report", MagicMock())

    result = scanner.run_scan(dry_run=False, max_tickers=3)

    assert attempts["n"] == 2
    assert scanner.time.sleep.call_count == 1
    alert_mock.assert_not_called()
    fake_orc.run_universe.assert_called_once()
    assert result is not None
