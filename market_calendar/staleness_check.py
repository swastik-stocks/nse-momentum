"""
Fix for the confirm_picks.py staleness-check bug.

BUG: The 10am confirmation was comparing picks_latest.json's embedded date
against TODAY's date, and flagging a mismatch as "stale". But by design,
the evening scan runs the NIGHT BEFORE and produces picks meant to be
confirmed the NEXT trading morning -- so picks_latest.json being dated
"yesterday" is the CORRECT, expected state, not an error.

FIX: Compare against the previous TRADING day (skipping weekends AND
NSE holidays), not literally "today".

Drop this into your repo (e.g. as nse_momentum/market_calendar.py) and
import previous_trading_day() + check_staleness() into confirm_picks.py.
"""

import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Official NSE 2026 trading holidays (equity segment), sourced from
# Zerodha's published holiday calendar. Update this list each December
# when NSE publishes the following year's calendar.
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 15),   # Municipal Corporation Elections in Maharashtra
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Eid
    date(2026, 6, 26),   # Moharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}


def is_trading_day(d: date, holidays: set = NSE_HOLIDAYS_2026) -> bool:
    """NSE is closed on weekends and the holidays listed above."""
    if d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    if d in holidays:
        return False
    return True


def previous_trading_day(ref_date: date, holidays: set = NSE_HOLIDAYS_2026) -> date:
    """Returns the most recent trading day strictly before ref_date."""
    d = ref_date - timedelta(days=1)
    while not is_trading_day(d, holidays):
        d -= timedelta(days=1)
    return d


def check_staleness(picks_date: date, today: date = None) -> None:
    """
    Raises StaleDataError only if picks_date does NOT match the last
    trading day before `today` -- i.e. only a REAL gap (evening scan
    genuinely missed running) triggers the alert.

    Usage in confirm_picks.py:
        picks_date = date.fromisoformat(picks_json["scan_date"])
        check_staleness(picks_date)
    """
    if today is None:
        today = date.today()

    expected_scan_date = previous_trading_day(today)

    if picks_date not in (expected_scan_date, today):
        raise StaleDataError(
            f"picks_latest.json is dated {picks_date} but the last "
            f"trading day before {today} was {expected_scan_date}. "
            f"Evening scan appears to have genuinely missed a run."
        )


class StaleDataError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# [2026-08-18] Shared fail-closed data-provenance contract.
#
# Root cause of the 2026-08-18 stale-data incidents: every fetch path in this
# codebase that could not get fresh data degraded SILENTLY -- a fabricated
# VIX default, a Bhavcopy file up to 4 days old accepted with no date check,
# a Dhan API 200-OK-with-empty-payload treated as a real rvol=0.0 reading.
# picks_latest.json's own "scan_date" field was wall-clock date.today(), not
# the actual vintage of the data inside it -- so nothing downstream could
# even detect the problem.
#
# DataProvenance is the one shape both the evening scan (scanner.py /
# orchestrator.py / data_fetcher.py) and the morning checker
# (confirm_picks.py / dhan_rvol.py) populate and consume, extending the
# "unverified must BLOCK, not silently pass" contract already used by
# get_asm_gsm_symbols()/days_to_next_earnings() in confirm_picks.py to every
# other fetch path.
# ─────────────────────────────────────────────────────────────────────────────

IST_OFFSET = timedelta(hours=5, minutes=30)


@dataclass
class DataProvenance:
    source_name: str
    ok: bool
    reason: str
    requested_date: Optional[date] = None
    actual_date:    Optional[date] = None
    fetched_at:     Optional[datetime] = None


def verify_bhavcopy_date(actual_iso_date: Optional[str], expected_date: date) -> DataProvenance:
    """
    Wraps the date-string-equality check the evening pipeline needs before
    trusting a Bhavcopy fetch. `actual_iso_date` is whatever date the
    Bhavcopy parser actually resolved (may be None if nothing parsed at
    all) -- ok=True only when it EXACTLY matches expected_date, never "close
    enough" / "most recent available". See BhavcopyFetcher.get_delivery_pct()
    in data_fetcher.py, which now loops candidate dates itself and only
    accepts the one that satisfies this check.
    """
    if not actual_iso_date:
        return DataProvenance(
            source_name="bhavcopy", ok=False,
            reason=f"no Bhavcopy file could be parsed for expected trading date {expected_date}",
            requested_date=expected_date, actual_date=None,
        )
    try:
        actual = date.fromisoformat(actual_iso_date)
    except ValueError:
        return DataProvenance(
            source_name="bhavcopy", ok=False,
            reason=f"Bhavcopy returned an unparseable date string {actual_iso_date!r}",
            requested_date=expected_date, actual_date=None,
        )
    if actual != expected_date:
        return DataProvenance(
            source_name="bhavcopy", ok=False,
            reason=f"Bhavcopy resolved to {actual} but expected trading date is {expected_date} -- "
                   f"today's Bhavcopy may not be published yet, refusing to silently use a stale one",
            requested_date=expected_date, actual_date=actual,
        )
    return DataProvenance(
        source_name="bhavcopy", ok=True, reason="Bhavcopy date matches expected trading date",
        requested_date=expected_date, actual_date=actual,
    )


def verify_intraday_freshness(fetched_at: Optional[datetime], as_of: datetime,
                               max_staleness_minutes: float = 10.0,
                               source_name: str = "intraday") -> DataProvenance:
    """
    Generic "is this timestamp too old to trust as live" check for the
    morning pipeline's price/volume fetches. `fetched_at` should be the
    real wall-clock time the underlying data point was captured (or None if
    the source can't provide one at all, e.g. yfinance's fast_info -- that
    case is always ok=False, since an unprovable timestamp is exactly the
    "unknown != current" case this exists to catch).
    """
    if fetched_at is None:
        return DataProvenance(
            source_name=source_name, ok=False,
            reason="no fetch timestamp available -- cannot prove this value is current",
            fetched_at=None,
        )
    # Callers pass a mix of tz-aware (datetime.now(IST)) and naive-but-IST
    # (dhan_rvol's candle timestamps, naive by deliberate convention -- see
    # dhan_rvol.py's fetch_intraday_candles) datetimes. Both represent the
    # same IST wall clock, so normalize to naive before subtracting rather
    # than raising on the aware/naive mismatch.
    if as_of.tzinfo is not None:
        as_of = as_of.replace(tzinfo=None)
    if fetched_at.tzinfo is not None:
        fetched_at = fetched_at.replace(tzinfo=None)
    age_minutes = (as_of - fetched_at).total_seconds() / 60.0
    if age_minutes > max_staleness_minutes:
        return DataProvenance(
            source_name=source_name, ok=False,
            reason=f"data is {age_minutes:.1f}min old, exceeds {max_staleness_minutes:.1f}min freshness bound",
            fetched_at=fetched_at,
        )
    if age_minutes < -1.0:
        # Fetched-in-the-future by more than clock-skew tolerance -- a sign
        # of a timezone bug, not real data. Fail closed rather than trust it.
        return DataProvenance(
            source_name=source_name, ok=False,
            reason=f"fetch timestamp is {abs(age_minutes):.1f}min in the future relative to as_of -- "
                   f"likely a timezone bug, not real data",
            fetched_at=fetched_at,
        )
    return DataProvenance(
        source_name=source_name, ok=True,
        reason=f"data is {age_minutes:.1f}min old, within {max_staleness_minutes:.1f}min bound",
        fetched_at=fetched_at,
    )


def get_code_provenance() -> dict:
    """
    [5b-i] Records which exact code produced a given run's output --
    directly closes the root cause of today's original incident: the
    evening scan's fixes existed only as uncommitted local changes, so the
    scheduled cloud run (a fresh checkout of committed code) silently
    produced different output than the local run, with nothing in
    picks_latest.json to reveal the discrepancy.

    Returns a dict (never raises -- git being unavailable is itself
    information, not a fatal error):
      git_commit, git_commit_ts, git_dirty, git_branch: from git directly.
      is_cloud_runner: True when GITHUB_ACTIONS=true is set (GitHub Actions
        sets this itself) -- a fresh checkout there should NEVER be dirty;
        callers use this to decide abort-vs-warn severity for git_dirty.
      runner: hostname, best-effort.
    """
    repo_dir = Path(__file__).resolve().parent.parent

    def _git(*args) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True,
                text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
        except Exception:
            return None

    commit     = _git("rev-parse", "HEAD")
    commit_ts  = _git("log", "-1", "--format=%cI")
    branch     = _git("branch", "--show-current")
    porcelain  = _git("status", "--porcelain")

    try:
        import socket
        runner = socket.gethostname()
    except Exception:
        runner = "unknown"

    return {
        "git_commit":     commit,
        "git_commit_ts":  commit_ts,
        "git_branch":     branch,
        # None (git unavailable) is treated as dirty=True by callers --
        # "can't verify clean" must never be silently read as "clean".
        "git_dirty":      True if porcelain is None else bool(porcelain),
        "is_cloud_runner": os.environ.get("GITHUB_ACTIONS", "").lower() == "true",
        "runner":         runner,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Closing Auction Session (CAS) awareness
# ─────────────────────────────────────────────────────────────────────────────
# NSE introduced CAS for F&O-eligible stocks (live since the circulars dated
# 2026-01-19 through 2026-05-29; see nseindia.com/static/products-services/
# closing-auction-session). For those names, continuous trading stops at
# 15:15 IST -- NOT 15:30 -- and the real closing price is only set by a
# 15:30-15:35 call-auction match. Reading a "live price" during 15:15-15:35
# for a CAS name is reading a frozen continuous-session print, not the
# actual close. Non-F&O names are unaffected and still trade continuously to
# 15:30 as before.

CAS_CONTINUOUS_CLOSE  = dtime(15, 15)  # continuous trading stops here for CAS names
CAS_AUCTION_MATCH_END = dtime(15, 35)  # auction match completes; real close now exists
NON_CAS_CLOSE         = dtime(15, 30)  # unaffected names: unchanged, existing assumption

_DEFAULT_SCRIP_MASTER = Path(__file__).resolve().parent.parent / "data" / "dhan_scrip_master.csv"


@lru_cache(maxsize=4)
def _load_cas_eligible_symbols_cached(scrip_master_path: str) -> frozenset:
    symbols = set()
    try:
        import csv
        with open(scrip_master_path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if row.get("SEM_EXM_EXCH_ID") != "NSE" or row.get("SEM_SEGMENT") != "D":
                    continue
                if row.get("SEM_INSTRUMENT_NAME") not in ("FUTSTK", "OPTSTK"):
                    continue
                underlying = row.get("SEM_TRADING_SYMBOL", "").split("-")[0].strip()
                if underlying:
                    symbols.add(underlying)
    except Exception:
        return frozenset()
    return frozenset(symbols)


def load_cas_eligible_symbols(scrip_master_path: Path = None) -> set:
    """
    Parses the Dhan scrip master for NSE stock-derivative rows (FUTSTK/
    OPTSTK -- index derivatives FUTIDX/OPTIDX don't imply CAS eligibility
    for an individual underlying) and returns the set of underlying trading
    symbols that carry F&O contracts, i.e. are CAS-eligible.

    Cached per resolved path (the scrip master is itself refreshed at most
    once a day elsewhere in the pipeline, so re-parsing per call would be
    pure waste) -- keyed on path specifically so a call with a different or
    missing path never silently returns a previous call's result for a
    different file. Returns an empty set (never raises) if the file is
    missing or unparseable -- callers must treat an empty result as "can't
    verify CAS eligibility", not "definitely not CAS-eligible", the same
    unverifiable-is-not-verified convention as the rest of this module.
    """
    path = scrip_master_path or _DEFAULT_SCRIP_MASTER
    return set(_load_cas_eligible_symbols_cached(str(path)))


def is_cas_eligible(ticker: str, cas_symbols: set = None) -> bool:
    """
    True if `ticker` (with or without the .NS suffix) has F&O contracts and
    is therefore subject to the Closing Auction Session. Uses the cached
    scrip-master-derived set from load_cas_eligible_symbols() unless a set
    is passed explicitly (tests, or a caller iterating many tickers that
    wants to load the set once itself).
    """
    symbols = cas_symbols if cas_symbols is not None else load_cas_eligible_symbols()
    underlying = ticker.replace(".NS", "").strip()
    return underlying in symbols


CLOSE_PRICE_TOLERANCE_MINUTES = 5  # a candle timestamped just before the close still counts as "the close"


def verify_closing_price_freshness(ticker: str, fetched_at: Optional[datetime],
                                    as_of: datetime, cas_symbols: set = None,
                                    max_staleness_minutes: float = 10.0) -> DataProvenance:
    """
    CAS-aware wrapper around verify_intraday_freshness() for callers that
    need today's CLOSING price specifically (BTST-style checks), not just
    "some recent price".

    Two special cases layered on top of verify_intraday_freshness():

    1. CAS-eligible name during the 15:15-15:35 auction window: continuous-
       session data is frozen and NOT yet the real close -- no fetch
       timestamp can make that trustworthy, so this returns ok=False
       unconditionally regardless of how fresh `fetched_at` looks.

    2. [2026-08-19, fix same day as the first version] Post-close grace:
       once a name's market has closed for the day (15:30 for non-CAS,
       15:35 once the CAS auction has matched), its LAST print for that
       session is, by definition, the close -- there will never be a
       fresher one. verify_intraday_freshness()'s generic "fetched within
       the last N minutes" rule would otherwise reject every post-close
       BTST check as "stale" purely because wall-clock time keeps moving
       while the price rightfully doesn't -- caught in production on
       2026-08-19 when this blocked all 3 non-CAS BTST candidates at
       15:51. A `fetched_at` dated today and timestamped at/after
       (close - CLOSE_PRICE_TOLERANCE_MINUTES) is accepted as the close
       regardless of elapsed time since. Before the close, or for a
       `fetched_at` from an earlier point in the day (e.g. a stuck feed
       that never advanced), this still delegates to the normal N-minute
       freshness bound -- only genuinely-at-the-close prints get the
       grace.
    """
    as_of_naive = as_of.replace(tzinfo=None) if as_of.tzinfo is not None else as_of
    is_cas = is_cas_eligible(ticker, cas_symbols)

    if is_cas and CAS_CONTINUOUS_CLOSE <= as_of_naive.time() < CAS_AUCTION_MATCH_END:
        return DataProvenance(
            source_name="closing_price", ok=False,
            reason=(f"{ticker} is CAS-eligible (has F&O contracts) and it's "
                    f"{as_of_naive.strftime('%H:%M')} -- the 15:15-15:35 closing "
                    f"auction hasn't matched yet, so no trustworthy closing price "
                    f"exists for this name right now"),
            fetched_at=fetched_at,
        )

    close_time = CAS_AUCTION_MATCH_END if is_cas else NON_CAS_CLOSE
    if fetched_at is not None:
        fetched_at_naive = fetched_at.replace(tzinfo=None) if fetched_at.tzinfo is not None else fetched_at
        close_tolerance = (datetime.combine(fetched_at_naive.date(), close_time)
                            - timedelta(minutes=CLOSE_PRICE_TOLERANCE_MINUTES)).time()
        if (as_of_naive.time() >= close_time
                and fetched_at_naive.date() == as_of_naive.date()
                and fetched_at_naive.time() >= close_tolerance):
            return DataProvenance(
                source_name="closing_price", ok=True,
                reason=(f"{fetched_at_naive.strftime('%H:%M')} print is at/after today's "
                        f"{'CAS auction' if is_cas else 'continuous-session'} close "
                        f"({close_time.strftime('%H:%M')}) -- this IS the closing price, "
                        f"elapsed time since then doesn't make it stale"),
                fetched_at=fetched_at,
            )

    return verify_intraday_freshness(fetched_at, as_of, max_staleness_minutes=max_staleness_minutes,
                                      source_name="closing_price")


