"""
RVOL (Relative Volume) calculator using Dhan's historical intraday API.

Replaces tvDatafeed entirely. Computes:
    RVOL = today's first-45-min volume (09:15-10:00 IST) / 20-day average
           of that same first-45-min window

Requires the Dhan Data API subscription (₹499+GST/month) -- the free
tier only provides live LTP, not historical intraday candles.

Requires: DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID (already set as GitHub
Actions secrets / in your local .env from the existing Dhan integration).

Rate limits (per Dhan docs): Data APIs allow 5 requests/sec, 100,000/day.
This module sleeps briefly between calls to stay well under that.
"""

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

# [FIX] datetime.fromtimestamp(ts) with no tz arg converts to the HOST
# MACHINE's local timezone. That's fine on a local Windows box set to IST,
# but GitHub Actions runners default to UTC -- every candle's wall-clock
# time would silently shift by 5.5 hours there, so the 09:15-10:00 window
# comparison would never match anything and this would always return
# "no historical volume data available" in CI, even though it looked fine
# in local testing. Always convert explicitly to IST instead of relying on
# host-local time.
IST = ZoneInfo("Asia/Kolkata")

DHAN_ACCESS_TOKEN = os.environ["DHAN_ACCESS_TOKEN"]
DHAN_CLIENT_ID = os.environ["DHAN_CLIENT_ID"]

DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

DHAN_HEADERS = {
    "Content-Type": "application/json",
    "access-token": DHAN_ACCESS_TOKEN,
    "client-id": DHAN_CLIENT_ID,
}

# First-45-minute window: market open 09:15 IST to 10:00 IST
WINDOW_START = "09:15"
WINDOW_END = "10:00"

# Reuse the same cached instrument map your Dhan integration already builds.
# If your project already defines this elsewhere (e.g. confirm_picks.py),
# you can delete this block and `from confirm_picks import _load_instrument_map`
# instead -- kept self-contained here so this module works standalone.
DHAN_MAPPING_FILE = Path(os.environ.get("DHAN_MAPPING_FILE", ".dhan_instrument_map.json"))


def _load_instrument_map() -> dict:
    if DHAN_MAPPING_FILE.exists():
        age_days = (datetime.now().timestamp() - DHAN_MAPPING_FILE.stat().st_mtime) / 86400
        if age_days < 7:
            with open(DHAN_MAPPING_FILE) as f:
                return json.load(f)
    return _download_dhan_instrument_map()


def _download_dhan_instrument_map() -> dict:
    req_headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv",
                         headers=req_headers, timeout=30)
    lines = resp.text.strip().split("\n")
    headers = [h.strip().strip('"') for h in lines[0].split(",")]
    sid_idx = next(i for i, h in enumerate(headers) if "SECURITY_ID" in h)
    sym_idx = next(i for i, h in enumerate(headers) if "TRADING_SYMBOL" in h)
    exch_idx = next(i for i, h in enumerate(headers) if "EXCH_ID" in h or "EXCHANGE" in h)
    inst_idx = next(i for i, h in enumerate(headers) if "INSTRUMENT" in h)

    mapping = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= max(sid_idx, sym_idx):
            continue
        exch = parts[exch_idx].strip().strip('"')
        inst = parts[inst_idx].strip().strip('"')
        sym = parts[sym_idx].strip().strip('"')
        sid = parts[sid_idx].strip().strip('"')
        if "NSE" in exch.upper() and "EQUITY" in inst.upper():
            mapping[sym] = sid

    with open(DHAN_MAPPING_FILE, "w") as f:
        json.dump(mapping, f)
    return mapping


DHAN_EPOCH = datetime(1980, 1, 1)  # Dhan historical-data timestamps are seconds
                                    # since this custom epoch (IST), NOT the
                                    # standard 1970 Unix epoch -- this was the
                                    # source of the RVOL=0 bug.


def _fetch_intraday_candles(security_id: str, from_date: str, to_date: str,
                             exchange_segment: str = "NSE_EQ", interval: str = "5") -> list:
    """
    Fetch intraday candles from Dhan for a date range.
    interval: candle size in minutes ("1", "5", "15", "25", "60").
    Returns a list of dicts with keys: datetime, volume.

    Note: securityId must be a STRING for this endpoint (Dhan's docs show
    "securityId": "1333" here, even though the marketfeed/quote endpoint
    wants security IDs as integers -- inconsistent across their own API).
    """
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": exchange_segment,
        "instrument": "EQUITY",
        "interval": interval,
        "oi": False,
        "fromDate": f"{from_date} 09:15:00",
        "toDate": f"{to_date} 15:30:00",
    }
    resp = requests.post(DHAN_INTRADAY_URL, headers=DHAN_HEADERS, json=payload, timeout=15)
    if resp.status_code != 200:
        print(f"DEBUG Dhan intraday error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    data = resp.json()

    # Dhan returns parallel arrays: {"open": [...], "high": [...], ..., "timestamp": [...]}
    timestamps = data.get("timestamp", [])
    volumes = data.get("volume", [])
    candles = []
    for ts, vol in zip(timestamps, volumes):
        # v2 /charts/intraday uses the standard Unix epoch (1970). The
        # "custom epoch since 1980" note in some Dhan docs applies to the
        # deprecated v1 API only -- using it here shifted every date
        # exactly 10 years into the future (confirmed by decoded dates
        # landing in 2036 instead of 2026).
        #
        # [FIX] Convert explicitly to IST, then drop tzinfo for naive
        # comparison against the naive IST wall-clock windows built below --
        # do NOT use fromtimestamp(ts) with no tz, which uses host-local
        # time and breaks on UTC CI runners (see IST comment above).
        candle_dt = datetime.fromtimestamp(ts, tz=IST).replace(tzinfo=None)
        candles.append({
            "datetime": candle_dt,
            "volume": vol,
        })
    return candles


# [FIX] Elapsed-time-matched window, replacing the old fixed 09:15-10:00
# window. The scheduled checkpoints (09:16/09:20/09:35/09:55) all fall
# BEFORE 10:00 -- a fixed 45-min window meant "today" would always have
# less volume summed than the historical 20-day average (which used the
# FULL 45 minutes every time), artificially depressing RVOL at every real
# checkpoint. This mirrors the same elapsed-matching approach already used
# by _rvol_tvdatafeed/_rvol_yfinance_fallback in confirm_picks.py, so all
# three RVOL sources are apples-to-apples regardless of what time a
# checkpoint runs.
MIN_ELAPSED_MINUTES = 5  # matches the "too early" guard used elsewhere


def _elapsed_window(as_of: datetime, target_date: date) -> tuple:
    """
    Returns (window_start_dt, window_end_dt) for target_date, spanning from
    market open to the SAME number of elapsed minutes as `as_of` is past
    market open today. Used for both today and each historical day, so the
    comparison is always minutes-elapsed vs minutes-elapsed, never a full
    45-min historical window vs a partial today window.
    """
    market_open_time = datetime.strptime(WINDOW_START, "%H:%M").time()
    open_dt_today     = datetime.combine(as_of.date(), market_open_time)
    elapsed           = as_of - open_dt_today
    if elapsed.total_seconds() < 0:
        elapsed = timedelta(0)

    window_start_dt = datetime.combine(target_date, market_open_time)
    window_end_dt   = window_start_dt + elapsed
    return window_start_dt, window_end_dt


def _sum_elapsed_volume(candles: list, window_start_dt: datetime,
                         window_end_dt: datetime) -> float:
    """Sum volume for candles falling within [window_start_dt, window_end_dt]."""
    total = 0.0
    for c in candles:
        if window_start_dt <= c["datetime"] <= window_end_dt:
            total += c["volume"]
    return total


def _get_trading_days_back(n: int) -> list:
    """Return the last n trading days (Mon-Fri, naive -- doesn't account
    for NSE holidays, which is fine for a 20-day average since a couple
    of missing days barely moves the average)."""
    days = []
    d = date.today() - timedelta(days=1)  # start from yesterday
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    return days


def compute_rvol(symbol: str, security_id: str = None, exchange_segment: str = "NSE_EQ",
                  as_of: datetime = None) -> dict:
    """
    Compute RVOL for one symbol, elapsed-time-matched against `as_of`
    (defaults to now, IST).

    Returns:
        {
            "symbol": str,
            "today_volume": float,
            "avg_20d_volume": float,
            "rvol": float,       # e.g. 1.8 means 1.8x normal volume
            "source": "dhan",
            "elapsed_minutes": float,
        }
    or {"symbol": ..., "error": "..."} if data was unavailable or it's too
    early in the session for a meaningful comparison.
    """
    # [FIX] as_of may arrive either naive (e.g. direct/local testing) or
    # timezone-aware (the real production path: confirm_picks.py's main()
    # builds now_ist = datetime.now(IST), which IS tz-aware, and passes it
    # straight through get_rvol() -> _rvol_dhan() -> here). Subtracting that
    # against the naive datetime.combine(...) below raised:
    #   TypeError: can't subtract offset-naive and offset-aware datetimes
    # for every single ticker in production. Always normalize to a naive
    # IST wall-clock datetime here, regardless of what was passed in.
    if as_of is None:
        as_of = datetime.now(IST).replace(tzinfo=None)
    elif as_of.tzinfo is not None:
        as_of = as_of.astimezone(IST).replace(tzinfo=None)

    if security_id is None:
        instrument_map = _load_instrument_map()
        clean_symbol = symbol.replace(".NS", "")
        security_id = instrument_map.get(clean_symbol)
        if security_id is None:
            return {"symbol": symbol, "error": "security_id not found in instrument map"}

    today = as_of.date()
    market_open_time = datetime.strptime(WINDOW_START, "%H:%M").time()
    elapsed_minutes = (as_of - datetime.combine(today, market_open_time)).total_seconds() / 60

    if elapsed_minutes < MIN_ELAPSED_MINUTES:
        return {"symbol": symbol,
                "error": f"only {elapsed_minutes:.0f}min elapsed since open -- "
                         f"too early for a meaningful RVOL"}

    # Today's elapsed-window volume
    today_candles = _fetch_intraday_candles(
        security_id,
        from_date=today.strftime("%Y-%m-%d"),
        to_date=today.strftime("%Y-%m-%d"),
        exchange_segment=exchange_segment,
    )
    today_start_dt, today_end_dt = _elapsed_window(as_of, today)
    today_volume = _sum_elapsed_volume(today_candles, today_start_dt, today_end_dt)
    time.sleep(0.25)  # stay under 5 req/sec

    # 20-day average of the SAME elapsed window on each historical day
    trading_days = _get_trading_days_back(20)
    oldest, newest = min(trading_days), max(trading_days)

    hist_candles = _fetch_intraday_candles(
        security_id,
        from_date=oldest.strftime("%Y-%m-%d"),
        to_date=newest.strftime("%Y-%m-%d"),
        exchange_segment=exchange_segment,
    )
    time.sleep(0.25)

    daily_volumes = []
    for d in trading_days:
        d_start_dt, d_end_dt = _elapsed_window(as_of, d)
        vol = _sum_elapsed_volume(hist_candles, d_start_dt, d_end_dt)
        if vol > 0:
            daily_volumes.append(vol)

    if not daily_volumes:
        return {"symbol": symbol, "error": "no historical volume data available"}

    avg_20d_volume = sum(daily_volumes) / len(daily_volumes)
    rvol = (today_volume / avg_20d_volume) if avg_20d_volume > 0 else 0.0

    return {
        "symbol": symbol,
        "today_volume": today_volume,
        "avg_20d_volume": avg_20d_volume,
        "rvol": round(rvol, 2),
        "source": "dhan",
        "elapsed_minutes": round(elapsed_minutes, 1),
    }


if __name__ == "__main__":
    # Quick smoke test on a liquid stock
    result = compute_rvol("HDFCBANK.NS", security_id="1333")
    print(json.dumps(result, indent=2, default=str))

    # Diagnostic: check if yesterday's data is available on its own
    # (helps determine if there's a lag before intraday data becomes available)
    yesterday = date.today() - timedelta(days=1)
    print(f"\nDEBUG checking yesterday ({yesterday}) in isolation...")
    yesterday_candles = _fetch_intraday_candles(
        "1333",
        from_date=yesterday.strftime("%Y-%m-%d"),
        to_date=yesterday.strftime("%Y-%m-%d"),
    )
    print(f"DEBUG yesterday candle count: {len(yesterday_candles)}")
    if yesterday_candles:
        print(f"DEBUG yesterday first candle: {yesterday_candles[0]}")
        print(f"DEBUG yesterday last candle: {yesterday_candles[-1]}")
