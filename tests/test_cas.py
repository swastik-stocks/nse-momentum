"""
[2026-08-19] Regression tests for Closing Auction Session (CAS) awareness.

NSE introduced CAS for F&O-eligible stocks (live since the circulars dated
2026-01-19 through 2026-05-29): continuous trading for those names stops at
15:15 IST, not 15:30, and the real closing price isn't set until the
15:30-15:35 call-auction match. These tests pin the behaviour added in
market_calendar/staleness_check.py (is_cas_eligible, verify_closing_price_
freshness) and confirm_picks.py's classify_btst gate 0c, so a future refactor
can't silently reintroduce "treat a frozen continuous-session print as the
real close" for F&O names.

Unit-level only, no live network or file I/O beyond a tiny in-memory CSV.
"""
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

IST = ZoneInfo("Asia/Kolkata")

_FAKE_SCRIP_MASTER_ROWS = [
    {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "D", "SEM_INSTRUMENT_NAME": "FUTSTK",
     "SEM_TRADING_SYMBOL": "RADICO-Aug2026-FUT"},
    {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "D", "SEM_INSTRUMENT_NAME": "OPTSTK",
     "SEM_TRADING_SYMBOL": "RADICO-Aug2026-5250-CE"},
    {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "D", "SEM_INSTRUMENT_NAME": "FUTIDX",
     "SEM_TRADING_SYMBOL": "NIFTY-Aug2026-FUT"},  # index derivative -- must NOT count
    {"SEM_EXM_EXCH_ID": "BSE", "SEM_SEGMENT": "D", "SEM_INSTRUMENT_NAME": "FUTSTK",
     "SEM_TRADING_SYMBOL": "RADICO-Aug2026-FUT"},  # wrong exchange -- must NOT count
    {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "E", "SEM_INSTRUMENT_NAME": "EQUITY",
     "SEM_TRADING_SYMBOL": "WELCORP"},  # equity row, not derivative -- must NOT count
]

_FIELDNAMES = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_INSTRUMENT_NAME",
               "SEM_EXPIRY_CODE", "SEM_TRADING_SYMBOL", "SEM_LOT_UNITS", "SEM_CUSTOM_SYMBOL",
               "SEM_EXPIRY_DATE", "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE", "SEM_TICK_SIZE",
               "SEM_EXPIRY_FLAG", "SEM_EXCH_INSTRUMENT_TYPE", "SEM_SERIES", "SM_SYMBOL_NAME"]


@pytest.fixture
def fake_scrip_master(tmp_path):
    path = tmp_path / "scrip_master.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for row in _FAKE_SCRIP_MASTER_ROWS:
            w.writerow({k: row.get(k, "") for k in _FIELDNAMES})
    return path


def test_load_cas_eligible_symbols_only_counts_nse_stock_derivatives(fake_scrip_master):
    from market_calendar.staleness_check import load_cas_eligible_symbols
    symbols = load_cas_eligible_symbols(fake_scrip_master)
    assert symbols == {"RADICO"}, "index derivatives, non-NSE rows, and equity rows must not count"


def test_load_cas_eligible_symbols_missing_file_returns_empty_not_raises(tmp_path):
    from market_calendar.staleness_check import load_cas_eligible_symbols
    symbols = load_cas_eligible_symbols(tmp_path / "does_not_exist.csv")
    assert symbols == set()


def test_is_cas_eligible_matches_with_or_without_ns_suffix():
    from market_calendar.staleness_check import is_cas_eligible
    cas_symbols = {"RADICO", "APLAPOLLO"}
    assert is_cas_eligible("RADICO.NS", cas_symbols) is True
    assert is_cas_eligible("RADICO", cas_symbols) is True
    assert is_cas_eligible("WELCORP.NS", cas_symbols) is False


def test_cas_name_during_auction_window_is_not_ok_even_if_fetch_is_fresh():
    from market_calendar.staleness_check import verify_closing_price_freshness
    cas_symbols = {"RADICO"}
    as_of = datetime(2026, 8, 19, 15, 20, tzinfo=IST)
    fetched_at = datetime(2026, 8, 19, 15, 19)  # 1 minute old -- would pass a plain freshness check
    prov = verify_closing_price_freshness("RADICO.NS", fetched_at, as_of, cas_symbols=cas_symbols)
    assert prov.ok is False
    assert "auction" in prov.reason.lower()


def test_non_cas_name_during_same_window_uses_normal_freshness_rules():
    from market_calendar.staleness_check import verify_closing_price_freshness
    cas_symbols = {"RADICO"}  # WELCORP is not in this set
    as_of = datetime(2026, 8, 19, 15, 20, tzinfo=IST)
    fetched_at = datetime(2026, 8, 19, 15, 19)
    prov = verify_closing_price_freshness("WELCORP.NS", fetched_at, as_of, cas_symbols=cas_symbols)
    assert prov.ok is True


def test_cas_name_after_auction_matches_uses_normal_freshness_rules():
    from market_calendar.staleness_check import verify_closing_price_freshness
    cas_symbols = {"RADICO"}
    as_of = datetime(2026, 8, 19, 15, 40, tzinfo=IST)
    fetched_at = datetime(2026, 8, 19, 15, 38)
    prov = verify_closing_price_freshness("RADICO.NS", fetched_at, as_of, cas_symbols=cas_symbols)
    assert prov.ok is True


def test_cas_name_before_continuous_close_uses_normal_freshness_rules():
    from market_calendar.staleness_check import verify_closing_price_freshness
    cas_symbols = {"RADICO"}
    as_of = datetime(2026, 8, 19, 11, 0, tzinfo=IST)
    fetched_at = datetime(2026, 8, 19, 10, 58)
    prov = verify_closing_price_freshness("RADICO.NS", fetched_at, as_of, cas_symbols=cas_symbols)
    assert prov.ok is True


# ─────────────────────────────────────────────────────────────────────────
# classify_btst gate 0c
# ─────────────────────────────────────────────────────────────────────────

def test_classify_btst_blocks_on_unverified_closing_price():
    from confirm_picks import classify_btst
    from market_calendar.staleness_check import DataProvenance

    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0, "ticker": "RADICO.NS"}
    bad_prov = DataProvenance(source_name="closing_price", ok=False,
                               reason="CAS auction hasn't matched yet", fetched_at=None)

    result = classify_btst(pick, cmp=110.0, day_high=112.0, rvol=2.0, rvol_src="dhan",
                            asm_gsm_symbols=set(), days_to_earnings=-1,
                            price_provenance=bad_prov)

    assert result["status"] == "BTST_UNVERIFIED"
    assert "auction" in result["action"].lower()


def test_classify_btst_unaffected_when_price_provenance_ok():
    from confirm_picks import classify_btst
    from market_calendar.staleness_check import DataProvenance

    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0, "ticker": "WELCORP.NS"}
    good_prov = DataProvenance(source_name="closing_price", ok=True,
                                reason="fresh", fetched_at=None)

    result = classify_btst(pick, cmp=110.0, day_high=112.0, rvol=2.0, rvol_src="dhan",
                            asm_gsm_symbols=set(), days_to_earnings=-1,
                            price_provenance=good_prov)

    assert result["status"] != "BTST_UNVERIFIED"


def test_classify_btst_default_none_provenance_does_not_block():
    """price_provenance is optional -- existing callers that don't pass it
    (none currently, but future ones might) must not be silently blocked."""
    from confirm_picks import classify_btst

    pick = {"entry": 100.0, "sl": 95.0, "pivot": 99.0, "t1": 120.0, "ticker": "WELCORP.NS"}
    result = classify_btst(pick, cmp=110.0, day_high=112.0, rvol=2.0, rvol_src="dhan",
                            asm_gsm_symbols=set(), days_to_earnings=-1)
    assert result["status"] != "BTST_UNVERIFIED"
