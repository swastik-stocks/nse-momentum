"""
[2026-08-18] Regression tests for the data-freshness/fail-closed fixes
made after the 2026-08-18 stale-data incidents (see market_calendar/
staleness_check.py, dhan_rvol.py, confirm_picks.py, data_fetcher.py,
sanity_gate.py). Each test mirrors a concrete failure mode this codebase
actually hit -- the point is to stop this bug CLASS from silently
recurring after a future refactor, not to be exhaustive.

Unit-level only: every external call (Dhan API, git, yfinance) is mocked.
No live network needed -- runs the same in this sandbox and in CI.
"""
import subprocess
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────
# 1. dhan_rvol.py empty-candle silent-zero bug (section 4a)
# ─────────────────────────────────────────────────────────────────────────

def test_dhan_empty_candle_response_is_an_error_not_zero_rvol():
    import dhan_rvol

    with patch.object(dhan_rvol, "_load_instrument_map", return_value={"RELIANCE": "1333"}), \
         patch.object(dhan_rvol, "_fetch_intraday_candles", return_value=[]), \
         patch.object(dhan_rvol.time, "sleep"):
        result = dhan_rvol.compute_rvol("RELIANCE.NS", as_of=datetime(2026, 8, 18, 9, 30))

    assert "error" in result, "empty candle array must be an explicit error, not a fabricated rvol"
    assert "rvol" not in result


def test_dhan_nonempty_candles_compute_normally():
    import dhan_rvol

    today_candles = [
        {"datetime": datetime(2026, 8, 18, 9, 15), "volume": 1000,
         "open": 100, "high": 101, "low": 99, "close": 100.5},
        {"datetime": datetime(2026, 8, 18, 9, 20), "volume": 1500,
         "open": 100.5, "high": 102, "low": 100, "close": 101},
    ]
    hist_day = date(2026, 8, 17)
    hist_candles = [
        {"datetime": datetime.combine(hist_day, datetime.min.time()) + timedelta(hours=9, minutes=15),
         "volume": 800, "open": 99, "high": 100, "low": 98, "close": 99.5},
        {"datetime": datetime.combine(hist_day, datetime.min.time()) + timedelta(hours=9, minutes=20),
         "volume": 900, "open": 99.5, "high": 100.5, "low": 99, "close": 100},
    ]

    def fake_fetch(security_id, from_date, to_date, exchange_segment="NSE_EQ", interval="5"):
        return today_candles if from_date == to_date == "2026-08-18" else hist_candles

    with patch.object(dhan_rvol, "_load_instrument_map", return_value={"RELIANCE": "1333"}), \
         patch.object(dhan_rvol, "_fetch_intraday_candles", side_effect=fake_fetch), \
         patch.object(dhan_rvol, "_get_trading_days_back", return_value=[hist_day] * 3), \
         patch.object(dhan_rvol.time, "sleep"):
        result = dhan_rvol.compute_rvol("RELIANCE.NS", as_of=datetime(2026, 8, 18, 9, 20))

    assert "error" not in result
    assert result["rvol"] > 0
    assert result["source"] == "dhan"


# ─────────────────────────────────────────────────────────────────────────
# 2. Duplicate/mis-ordered candle detection (section 5b-iv)
# ─────────────────────────────────────────────────────────────────────────

def test_duplicate_candle_detected():
    import dhan_rvol
    candles = [
        {"datetime": datetime(2026, 8, 17, 9, 15), "volume": 100, "open": 1, "high": 2, "low": 1, "close": 1.5},
        {"datetime": datetime(2026, 8, 17, 9, 15), "volume": 100, "open": 1, "high": 2, "low": 1, "close": 1.5},
    ]
    violations = dhan_rvol._check_candle_integrity(candles)
    assert any("duplicate" in v for v in violations)


def test_non_monotonic_candle_detected():
    import dhan_rvol
    candles = [
        {"datetime": datetime(2026, 8, 17, 9, 20), "volume": 100, "open": 1, "high": 2, "low": 1, "close": 1.5},
        {"datetime": datetime(2026, 8, 17, 9, 15), "volume": 100, "open": 1, "high": 2, "low": 1, "close": 1.5},
    ]
    violations = dhan_rvol._check_candle_integrity(candles)
    assert any("monotonically" in v for v in violations)


def test_broken_ohlc_bounds_detected():
    import dhan_rvol
    # high below close -- physically impossible
    candles = [{"datetime": datetime(2026, 8, 17, 9, 15), "volume": 100,
                "open": 10, "high": 9, "low": 8, "close": 11}]
    violations = dhan_rvol._check_candle_integrity(candles)
    assert any("inconsistent" in v for v in violations)


def test_clean_candles_no_violations():
    import dhan_rvol
    candles = [
        {"datetime": datetime(2026, 8, 17, 9, 15), "volume": 100, "open": 10, "high": 11, "low": 9, "close": 10.5},
        {"datetime": datetime(2026, 8, 17, 9, 20), "volume": 150, "open": 10.5, "high": 12, "low": 10, "close": 11.5},
    ]
    assert dhan_rvol._check_candle_integrity(candles) == []


# ─────────────────────────────────────────────────────────────────────────
# 2b. Per-ticker OHLCV freshness (section 5b-ii) -- aggregate coverage %
#     is the wrong primitive; this asserts a stale ticker is individually
#     named, not averaged away inside a healthy-looking overall number.
# ─────────────────────────────────────────────────────────────────────────

def _fake_df(last_date: date, rows: int = 5):
    import pandas as pd
    idx = pd.date_range(end=pd.Timestamp(last_date), periods=rows, freq="D")
    return pd.DataFrame({"Close": [100.0] * rows}, index=idx)


def test_stale_ticker_individually_flagged_not_averaged_away():
    import pandas as pd
    from scanner import find_stale_tickers

    expected = date(2026, 8, 18)
    stock_data = {
        "RELIANCE.NS": _fake_df(expected),                      # fresh
        "TCS.NS":      _fake_df(expected),                      # fresh
        "XYZ.NS":      _fake_df(expected - timedelta(days=1)),  # stale by 1 day
        "EMPTY.NS":    pd.DataFrame(),                           # not loaded at all -- skipped, not "stale"
    }
    stale = find_stale_tickers(stock_data, expected)
    assert stale == ["XYZ.NS"], "stale ticker must be named individually, not folded into an aggregate"


def test_no_stale_tickers_when_all_fresh():
    from scanner import find_stale_tickers
    expected = date(2026, 8, 18)
    stock_data = {"RELIANCE.NS": _fake_df(expected), "TCS.NS": _fake_df(expected)}
    assert find_stale_tickers(stock_data, expected) == []


# ─────────────────────────────────────────────────────────────────────────
# 3. Bhavcopy date-mismatch abort (section 3 / verify_bhavcopy_date)
# ─────────────────────────────────────────────────────────────────────────

def test_bhavcopy_exact_date_match_is_ok():
    from market_calendar.staleness_check import verify_bhavcopy_date
    prov = verify_bhavcopy_date("2026-08-18", date(2026, 8, 18))
    assert prov.ok is True


def test_bhavcopy_stale_date_is_not_ok():
    from market_calendar.staleness_check import verify_bhavcopy_date
    prov = verify_bhavcopy_date("2026-08-15", date(2026, 8, 18))
    assert prov.ok is False
    assert "2026-08-15" in prov.reason or prov.actual_date == date(2026, 8, 15)


def test_bhavcopy_missing_date_is_not_ok():
    from market_calendar.staleness_check import verify_bhavcopy_date
    prov = verify_bhavcopy_date(None, date(2026, 8, 18))
    assert prov.ok is False


# ─────────────────────────────────────────────────────────────────────────
# 4. Intraday freshness (get_live_price / RVOL staleness bound)
# ─────────────────────────────────────────────────────────────────────────

def test_intraday_freshness_within_bound_is_ok():
    from market_calendar.staleness_check import verify_intraday_freshness
    as_of = datetime(2026, 8, 18, 9, 30)
    fetched_at = as_of - timedelta(minutes=2)
    prov = verify_intraday_freshness(fetched_at, as_of, max_staleness_minutes=10)
    assert prov.ok is True


def test_intraday_freshness_stale_is_not_ok():
    from market_calendar.staleness_check import verify_intraday_freshness
    as_of = datetime(2026, 8, 18, 9, 30)
    fetched_at = as_of - timedelta(minutes=25)
    prov = verify_intraday_freshness(fetched_at, as_of, max_staleness_minutes=10)
    assert prov.ok is False


def test_intraday_freshness_no_timestamp_is_not_ok():
    """The yfinance-fallback case: no timestamp at all must never be
    silently treated as fresh -- 'unknown != current'."""
    from market_calendar.staleness_check import verify_intraday_freshness
    prov = verify_intraday_freshness(None, datetime(2026, 8, 18, 9, 30))
    assert prov.ok is False


# ─────────────────────────────────────────────────────────────────────────
# 5. Dirty git tree on the cloud runner (section 5b-i)
# ─────────────────────────────────────────────────────────────────────────

def _mock_git_run(porcelain_output: str):
    def fake_run(args, cwd=None, capture_output=None, text=None, timeout=None, check=None):
        result = MagicMock()
        if args[:2] == ["git", "status"]:
            result.stdout = porcelain_output
        elif args[:2] == ["git", "rev-parse"]:
            result.stdout = "abc1234"
        elif args[:2] == ["git", "log"]:
            result.stdout = "2026-08-18T10:00:00+05:30"
        elif args[:2] == ["git", "branch"]:
            result.stdout = "main"
        else:
            result.stdout = ""
        return result
    return fake_run


def test_get_code_provenance_detects_dirty_tree(monkeypatch):
    from market_calendar.staleness_check import get_code_provenance
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with patch("subprocess.run", side_effect=_mock_git_run(" M confirm_picks.py\n")):
        prov = get_code_provenance()
    assert prov["git_dirty"] is True
    assert prov["is_cloud_runner"] is True


def test_get_code_provenance_clean_tree(monkeypatch):
    from market_calendar.staleness_check import get_code_provenance
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with patch("subprocess.run", side_effect=_mock_git_run("")):
        prov = get_code_provenance()
    assert prov["git_dirty"] is False


def test_dirty_tree_on_cloud_runner_is_the_abort_condition(monkeypatch):
    """
    Mirrors the exact condition scanner.py's run_scan() checks before
    aborting: a fresh GitHub Actions checkout should never be dirty -- if
    it is, that's the anomaly worth failing loudly on.
    """
    from market_calendar.staleness_check import get_code_provenance
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with patch("subprocess.run", side_effect=_mock_git_run(" M scanner.py\n")):
        prov = get_code_provenance()
    should_abort = prov["is_cloud_runner"] and prov["git_dirty"]
    assert should_abort is True


def test_dirty_tree_on_local_runner_does_not_abort(monkeypatch):
    from market_calendar.staleness_check import get_code_provenance
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with patch("subprocess.run", side_effect=_mock_git_run(" M scanner.py\n")):
        prov = get_code_provenance()
    should_abort = prov["is_cloud_runner"] and prov["git_dirty"]
    assert should_abort is False, "a local ad-hoc run must not be blocked -- it's the documented escape hatch"


# ─────────────────────────────────────────────────────────────────────────
# 6. get_live_price: Dhan-primary, yfinance-fallback marked unverified
# ─────────────────────────────────────────────────────────────────────────

def test_get_live_price_uses_dhan_when_available():
    import confirm_picks
    fake_ts = datetime(2026, 8, 18, 9, 31, 5)
    with patch.object(confirm_picks.dhan_rvol, "get_ltp",
                       return_value={"symbol": "RELIANCE", "price": 1234.5,
                                     "fetched_at": fake_ts, "source": "dhan"}):
        price, fetched_at, source, suspect = confirm_picks.get_live_price("RELIANCE.NS")
    assert price == 1234.5
    assert fetched_at == fake_ts
    assert source == "dhan"
    assert suspect is False


def test_get_live_price_falls_back_to_yfinance_and_marks_unverified():
    import confirm_picks

    fast_info = MagicMock()
    fast_info.last_price = 500.0
    fast_info.previous_close = 490.0
    mock_ticker = MagicMock()
    mock_ticker.fast_info = fast_info

    with patch.object(confirm_picks.dhan_rvol, "get_ltp",
                       return_value={"symbol": "RELIANCE", "error": "security_id not found"}), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        price, fetched_at, source, suspect = confirm_picks.get_live_price("RELIANCE.NS")

    assert price == 500.0
    assert fetched_at is None, "yfinance path must never fabricate a timestamp it doesn't have"
    assert source == "yfinance_fallback"
    assert suspect is False


def test_get_live_price_yfinance_stuck_feed_is_flagged_suspect():
    import confirm_picks

    fast_info = MagicMock()
    fast_info.last_price = 500.0
    fast_info.previous_close = 500.0   # exactly equal -- the stuck-feed tell
    mock_ticker = MagicMock()
    mock_ticker.fast_info = fast_info

    with patch.object(confirm_picks.dhan_rvol, "get_ltp",
                       return_value={"symbol": "RELIANCE", "error": "unavailable"}), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        price, fetched_at, source, suspect = confirm_picks.get_live_price("RELIANCE.NS")

    assert suspect is True


def test_classify_routes_unverified_price_to_price_unverified_status():
    import confirm_picks
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0}
    result = confirm_picks.classify(pick, cmp=101.0, rvol=1.8, rvol_src="dhan",
                                     price_verified=False, price_source="yfinance_fallback")
    assert result["status"] == "PRICE_UNVERIFIED"
    assert result["status"] not in confirm_picks.NO_RESEND_STATUSES, \
        "must stay non-terminal so it gets re-checked next checkpoint"


# ─────────────────────────────────────────────────────────────────────────
# 7. BROKEN re-verification path (section 4c) -- classify()-level check
#    that a recovered price is no longer classified BROKEN. The full
#    cache-eviction wiring lives in main()'s checkpoint loop; this proves
#    the underlying classification logic the recovery path depends on.
# ─────────────────────────────────────────────────────────────────────────

def test_classify_broken_when_at_or_below_sl():
    import confirm_picks
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0}
    result = confirm_picks.classify(pick, cmp=94.0, rvol=1.0, rvol_src="dhan")
    assert result["status"] == "BROKEN"


def test_classify_not_broken_once_price_recovers_above_sl():
    import confirm_picks
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0}
    result = confirm_picks.classify(pick, cmp=99.5, rvol=1.8, rvol_src="dhan")
    assert result["status"] != "BROKEN"


def test_broken_is_not_in_no_resend_statuses():
    import confirm_picks
    assert "BROKEN" not in confirm_picks.NO_RESEND_STATUSES, \
        "BROKEN must be re-verified each checkpoint, not frozen forever -- " \
        "see the 2026-08-18 fix for the sibling BREAKOUT_NO_VOLUME caching bug"


# ─────────────────────────────────────────────────────────────────────────
# 8. classify_btst fails closed when day_high is unverifiable (section 4e)
# ─────────────────────────────────────────────────────────────────────────

def test_classify_btst_blocks_when_day_high_unavailable():
    import confirm_picks
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0, "ticker": "TEST"}
    result = confirm_picks.classify_btst(
        pick, cmp=110.0, day_high=None, rvol=2.0, rvol_src="dhan",
        asm_gsm_symbols=set(), days_to_earnings=-1,
    )
    assert result["status"] == "BTST_UNVERIFIED"


def test_classify_btst_faded_check_runs_when_day_high_available():
    import confirm_picks
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 200.0, "ticker": "TEST"}
    # cmp far enough off today's high to trip the faded gate
    result = confirm_picks.classify_btst(
        pick, cmp=100.0, day_high=110.0, rvol=2.0, rvol_src="dhan",
        asm_gsm_symbols=set(), days_to_earnings=-1,
    )
    assert result["status"] == "FADED"
