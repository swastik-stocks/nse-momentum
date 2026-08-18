"""
NSE Momentum v5.5 — Data Fetcher

ARCHITECTURE: Four-tier price source hierarchy
  TIER 0 (PREFERRED): Dhan       — real broker data via DhanHQ v2 API, used
                                    when a live access token is available
  TIER 1 (PRIMARY):   tvDatafeed — TradingView data, same prices trader sees
  TIER 2 (OVERRIDE):  Bhavcopy   — NSE official close, injected into SQLite
  TIER 3 (HISTORY):   Yahoo Finance — 2yr backfill only, never used for CMP

WHY DHAN IS TIER 0, NOT A REPLACEMENT FOR TIERS 1-3:
  Dhan access tokens expire in exactly 24 hours (SEBI-mandated, no
  exceptions, confirmed 2026-07-29) and this system deliberately does NOT
  auto-refresh them (the PIN+TOTP headless flow was tried and abandoned —
  storing trading-capable credentials in CI secrets for a read-only data
  need was judged not worth the blast radius). That means Dhan WILL go
  stale on days the token isn't manually regenerated, and the system must
  keep working on those days, just with lower data quality — hence Dhan
  is tried first, opportunistically, with automatic fallback to the
  existing tvDatafeed -> Yahoo chain on ANY failure (expired token,
  missing security_id mapping, network error). See _check_dhan_auth() and
  get_dhan_status() — the latter is meant to be surfaced in the evening
  email so a stale token is an active daily reminder, not a silent
  degradation.

  v2.0 scrip master column names are not fully consistent across Dhan's
  own documentation (SEM_SMST_SECURITY_ID/SEM_TRADING_SYMBOL in some
  real-world examples, SECURITY_ID/UNDERLYING_SYMBOL in Dhan's own release
  notes) -- _load_dhan_scrip_master() below searches multiple candidate
  names defensively, the same pattern BhavcopyFetcher._parse() already
  uses for NSE's own column-name drift, and logs the actual columns found
  if none match rather than silently mis-mapping tickers.

  UNVERIFIED: none of the Dhan-tier code below has been run against a
  live token from this environment (no network path to api.dhan.co here).
  Test with a small ticker set before trusting it in a real scan -- same
  discipline as scanner.py's own --dry-run --max-tickers flags.

WHY THE ORIGINAL THREE-TIER ORDER (unchanged, still applies below Dhan):
  Yahoo Finance for NSE: 15-min delayed, cache has no staleness check,
  adjusted prices have errors. POLYCAB showed Rs.10,083 vs actual Rs.9,531
  (5.8% error) — every downstream number (entry, SL, target, R:R) was wrong.

  tvDatafeed: same data source as TradingView. What you see on screen = what
  scanner uses. No delay, no adjustment errors.

  Bhavcopy CLOSE_PRICE: NSE official settlement price. Injected into SQLite
  after every download, so cache is always authoritative for T-1.

BUG FIXES (v5.4):
  BUG-1  Bhavcopy columns: code looked for CLOSE, CSV has CLOSE_PRICE.
         fix: search CLOSE_PRICE / CLOSE / CLOSING_PRICE variants.

  BUG-2  Cache loaded without date validation — returned stale rows if >= 100.
         fix: _is_cache_fresh() checks last bar date >= T-1 before trusting cache.

  BUG-3  Bhavcopy official close ignored for CMP — delivery% extracted, price discarded.
         fix: inject_bhavcopy_to_db() writes CLOSE_PRICE for all EQ stocks to SQLite.

  BUG-4  bhavcopy_full_df never stored — breadth calculator received None.
         fix: self.full_df populated in _parse() and returned via get_delivery_pct().
         (data_dict wiring is in scanner.py — already fixed there.)

  BUG-5  No CMP cross-validation — stale data propagated silently.
         fix: validate_cmp_vs_bhavcopy() logs WARNING + returns bhavcopy_cmp_map
         for any stock deviating > 1% from official close.

  BUG-6  Near-breakout CMP used stale Yahoo cache — gap % wrong.
         fix: build_bhavcopy_cmp_map() returns {SYMBOL.NS: close} dict.
         near_breakout.py accepts this and uses it for price instead of df[-1].
"""

import logging, sqlite3, requests, os, time
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH  = DATA_DIR / "momentum_v4.db"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_last_trading_day(from_date: date = None) -> date:
    """
    Most recent NSE trading day on/before from_date.

    [2026-08-18] Was weekday-only (ignored NSE holidays) -- delegates to
    market_calendar.staleness_check.is_trading_day() now, the single
    holiday-aware source both the evening and morning pipelines already
    use, so "what date is this" logic can't silently diverge between the
    two callers again. Name/signature unchanged -- existing callers
    (_is_cache_fresh, fetch_dhan, etc.) need no changes.
    """
    from market_calendar.staleness_check import is_trading_day
    d = from_date or date.today()
    for i in range(10):
        candidate = d - timedelta(days=i)
        if is_trading_day(candidate):
            return candidate
    return d


# ─────────────────────────────────────────────────────────────────────────────
# SQLite cache
# ─────────────────────────────────────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            ticker TEXT, date TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    return conn


def _store_ohlcv(ticker: str, df: pd.DataFrame):
    if df.empty:
        return
    conn = _get_db()
    rows = []
    for dt, row in df.iterrows():
        rows.append((
            ticker, str(dt.date()),
            float(row.get("Open",   0)), float(row.get("High",   0)),
            float(row.get("Low",    0)), float(row.get("Close",  0)),
            float(row.get("Volume", 0)),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO price_history "
        "(ticker,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()


def _load_ohlcv_from_db(ticker: str, days: int = 504) -> pd.DataFrame:
    conn   = _get_db()
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows   = conn.execute(
        "SELECT date,open,high,low,close,volume FROM price_history "
        "WHERE ticker=? AND date>=? ORDER BY date",
        (ticker, cutoff)
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def _is_cache_fresh(df: pd.DataFrame) -> bool:
    """BUG-2 FIX: cache is only trusted if last bar == T-1 trading day."""
    if df.empty or len(df) < 100:
        return False
    last_cached = df.index[-1].date()
    t_minus_1   = get_last_trading_day()
    return last_cached >= t_minus_1


# ─────────────────────────────────────────────────────────────────────────────
# TIER 0: Dhan — preferred, opportunistic, never a hard dependency
# ─────────────────────────────────────────────────────────────────────────────

def _symbol_from_ticker(ticker: str) -> str:
    """POLYCAB.NS  →  POLYCAB"""
    return ticker.replace(".NS", "").replace(".BO", "")


DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
DHAN_HISTORICAL_URL   = "https://api.dhan.co/v2/charts/historical"
DHAN_SCRIP_CACHE      = DATA_DIR / "dhan_scrip_master.csv"

# Module-level status, populated once per process by _check_dhan_auth().
# Read via get_dhan_status() by scanner.py/emailer.py so a stale token is
# an ACTIVE reminder in the daily email, not a silent degradation.
_dhan_status = {
    "checked":            False,
    "available":          False,
    "message":            "Not checked yet",
    "tickers_from_dhan":  0,
}


def get_dhan_status() -> dict:
    """Public accessor for scanner.py/emailer.py to surface Dhan's health
    for this run — see module docstring for why this matters."""
    return dict(_dhan_status)


def _dhan_credentials() -> Optional[tuple]:
    client_id = os.environ.get("DHAN_CLIENT_ID")
    token     = os.environ.get("DHAN_ACCESS_TOKEN")
    if not client_id or not token:
        return None
    return client_id, token


def _normalize_dhan_scrip_master(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive column matching, factored out so it runs identically whether
    the scrip master came from a fresh download or the on-disk cache —
    an earlier version of this function only normalized on fresh download,
    which meant a cache HIT silently returned raw, un-normalized columns
    and would have crashed downstream in _dhan_symbol_map() the very first
    time this ran twice in a row. Caught by testing against a deliberately
    malformed cached CSV before shipping, not found live.
    """
    if raw.empty:
        return pd.DataFrame()

    cols = list(raw.columns)
    sec_id_col = next((c for c in cols if c in
                       ("SEM_SMST_SECURITY_ID", "SECURITY_ID", "SEM_SECURITY_ID")), None)
    symbol_col = next((c for c in cols if c in
                       ("SEM_TRADING_SYMBOL", "UNDERLYING_SYMBOL", "SYMBOL_NAME",
                        "TRADING_SYMBOL", "SEM_CUSTOM_SYMBOL")), None)
    exch_col   = next((c for c in cols if c in
                       ("SEM_EXM_EXCH_ID", "EXCH_ID", "EXCHANGE")), None)
    seg_col    = next((c for c in cols if c in
                       ("SEM_SEGMENT", "SEGMENT")), None)
    inst_col   = next((c for c in cols if c in
                       ("SEM_INSTRUMENT_NAME", "INSTRUMENT_TYPE", "INSTRUMENT")), None)

    if not sec_id_col or not symbol_col:
        log.warning(
            f"Dhan scrip master: cannot find security_id/symbol columns. "
            f"Found: {cols[:20]}. Dhan tier will be skipped this run — "
            f"falling back to tvDatafeed/Yahoo."
        )
        return pd.DataFrame()

    df = raw.copy()
    if exch_col:
        df = df[df[exch_col].astype(str).str.upper().isin(["NSE"])]
    if seg_col:
        df = df[df[seg_col].astype(str).str.upper().isin(["E", "EQ", "EQUITY"])]
    if inst_col:
        df = df[df[inst_col].astype(str).str.upper().str.contains("EQUITY", na=False)]

    out = df[[sec_id_col, symbol_col]].rename(
        columns={sec_id_col: "security_id", symbol_col: "trading_symbol"}
    ).dropna()
    out["security_id"] = out["security_id"].astype(str)
    out["trading_symbol"] = out["trading_symbol"].astype(str).str.strip().str.upper()
    return out.drop_duplicates(subset=["trading_symbol"])


def _load_dhan_scrip_master(force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads (or loads cached) Dhan's scrip master CSV and returns a
    normalized DataFrame with columns ['security_id', 'trading_symbol']
    for NSE equity instruments only. Normalization (_normalize_dhan_scrip_master)
    runs identically on both the cache-hit and fresh-download paths.

    UNVERIFIED against a live download from this environment (no network
    path to images.dhan.co here) -- first real run should log and be
    checked, same as any new data source in this codebase.
    """
    if not force_refresh and DHAN_SCRIP_CACHE.exists():
        age_hours = (datetime.now().timestamp() - DHAN_SCRIP_CACHE.stat().st_mtime) / 3600
        if age_hours < 20:   # refresh at least once within the token's 24h life
            try:
                return _normalize_dhan_scrip_master(pd.read_csv(DHAN_SCRIP_CACHE, low_memory=False))
            except Exception as e:
                log.warning(f"Dhan scrip master cache unreadable ({e}), re-downloading")

    try:
        r = requests.get(DHAN_SCRIP_MASTER_URL, timeout=30)
        r.raise_for_status()
        DHAN_SCRIP_CACHE.write_bytes(r.content)
        raw = pd.read_csv(DHAN_SCRIP_CACHE, low_memory=False)
    except Exception as e:
        log.warning(f"Dhan scrip master download failed: {e}")
        if DHAN_SCRIP_CACHE.exists():
            log.warning("Falling back to stale cached scrip master")
            raw = pd.read_csv(DHAN_SCRIP_CACHE, low_memory=False)
            return _normalize_dhan_scrip_master(raw)
        return pd.DataFrame()

    return _normalize_dhan_scrip_master(raw)


_dhan_symbol_map_cache: Optional[Dict[str, str]] = None


def _dhan_symbol_map() -> Dict[str, str]:
    """{bare symbol (no .NS): dhan security_id}, built once per process."""
    global _dhan_symbol_map_cache
    if _dhan_symbol_map_cache is not None:
        return _dhan_symbol_map_cache
    master = _load_dhan_scrip_master()
    if master.empty:
        _dhan_symbol_map_cache = {}
    else:
        _dhan_symbol_map_cache = dict(zip(master["trading_symbol"], master["security_id"]))
    return _dhan_symbol_map_cache


def _check_dhan_auth() -> bool:
    """
    ONE cheap test call at the start of a batch run to verify the token is
    still alive, rather than discovering it's dead on ticker #1 of 500
    (or worse, ticker #250, after 249 wasted calls). Uses RELIANCE
    (security_id 2885, a stable, always-listed large-cap) as the canary.
    Sets module-level _dhan_status so get_dhan_status() reflects this run.
    """
    global _dhan_status
    creds = _dhan_credentials()
    if not creds:
        _dhan_status = {"checked": True, "available": False,
                         "message": "DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN not set — skipping Dhan tier",
                         "tickers_from_dhan": 0}
        log.info(f"  Dhan: {_dhan_status['message']}")
        return False

    client_id, token = creds
    to_date   = get_last_trading_day()
    from_date = to_date - timedelta(days=10)
    try:
        resp = requests.post(
            DHAN_HISTORICAL_URL,
            headers={"Content-Type": "application/json", "access-token": token},
            json={
                "securityId": "2885",   # RELIANCE — canary probe, not used for data
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
            timeout=15,
        )
        if resp.status_code == 200:
            _dhan_status = {"checked": True, "available": True,
                             "message": "Dhan token valid", "tickers_from_dhan": 0}
            log.info("  Dhan: token valid, using as preferred data source this run")
            return True
        else:
            _dhan_status = {
                "checked": True, "available": False,
                "message": f"Dhan auth check failed: HTTP {resp.status_code} "
                           f"— {resp.text[:200]}. Falling back to tvDatafeed/Yahoo "
                           f"for this entire run. Refresh the token at web.dhan.co.",
                "tickers_from_dhan": 0,
            }
            log.warning(f"  ⚠ {_dhan_status['message']}")
            return False
    except Exception as e:
        _dhan_status = {
            "checked": True, "available": False,
            "message": f"Dhan auth check errored: {type(e).__name__}: {e}. "
                       f"Falling back to tvDatafeed/Yahoo for this entire run.",
            "tickers_from_dhan": 0,
        }
        log.warning(f"  ⚠ {_dhan_status['message']}")
        return False


def fetch_dhan(ticker: str, n_bars: int = 520) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Dhan for one ticker. Returns empty DataFrame
    on ANY failure (missing security_id mapping, HTTP error, malformed
    response) — callers must treat this identically to fetch_tv()
    returning empty, i.e. fall through to the next tier. Never raises.
    """
    creds = _dhan_credentials()
    if not creds or not _dhan_status.get("available"):
        return pd.DataFrame()
    client_id, token = creds

    symbol = _symbol_from_ticker(ticker)
    sec_id = _dhan_symbol_map().get(symbol)
    if not sec_id:
        log.debug(f"Dhan: no security_id mapping for {symbol} — falling back")
        return pd.DataFrame()

    to_date   = get_last_trading_day()
    # Daily bars, ~n_bars trading days back; padded to calendar days since
    # weekends/holidays aren't trading days.
    from_date = to_date - timedelta(days=int(n_bars * 1.55) + 10)

    try:
        resp = requests.post(
            DHAN_HISTORICAL_URL,
            headers={"Content-Type": "application/json", "access-token": token},
            json={
                "securityId": sec_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "expiryCode": 0,
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
            timeout=20,
        )
        if resp.status_code == 429:
            log.debug(f"Dhan fetch {ticker}: rate-limited (429) — falling back this ticker")
            return pd.DataFrame()
        if resp.status_code != 200:
            log.debug(f"Dhan fetch {ticker}: HTTP {resp.status_code} — {resp.text[:150]}")
            return pd.DataFrame()
        data = resp.json()
        required = ("open", "high", "low", "close", "volume", "timestamp")
        if not all(k in data for k in required):
            log.debug(f"Dhan fetch {ticker}: response missing expected keys, "
                       f"got {list(data.keys())[:10]}")
            return pd.DataFrame()

        df = pd.DataFrame({
            "Open":   data["open"],
            "High":   data["high"],
            "Low":    data["low"],
            "Close":  data["close"],
            "Volume": data["volume"],
        })
        df.index = pd.to_datetime(data["timestamp"], unit="s")
        df = df.sort_index()
        return df if len(df) >= 20 else pd.DataFrame()
    except Exception as e:
        log.debug(f"Dhan fetch {ticker} error: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: tvDatafeed — primary fallback OHLCV source (used when Dhan
# unavailable or doesn't have a given ticker)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tv(ticker: str, n_bars: int = 520) -> pd.DataFrame:
    """
    Fetch daily OHLCV from tvDatafeed for one ticker.
    Returns DataFrame with DatetimeIndex and OHLCV columns,
    or empty DataFrame on failure.
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv     = TvDatafeed()          # no login needed for NSE daily data
        symbol = _symbol_from_ticker(ticker)
        raw    = tv.get_hist(
            symbol=symbol, exchange="NSE",
            interval=Interval.in_daily, n_bars=n_bars
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close", "volume": "Volume"
        })
        raw = raw[["Open","High","Low","Close","Volume"]].copy()
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw = raw.sort_index()
        return raw
    except Exception as e:
        log.debug(f"tvDatafeed error {ticker}: {e}")
        return pd.DataFrame()


def fetch_batch_tv(tickers: List[str], n_bars: int = 520) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple tickers. Tries Dhan first (if a live token is
    available — checked ONCE per batch, not per ticker), then tvDatafeed
    for anything Dhan didn't provide, then Yahoo for anything tvDatafeed
    didn't provide either. Stores each result in SQLite so future runs use
    cache regardless of which tier provided it.
    """
    result      = {}
    tv_failed   = []

    # Check cache freshness first — avoids unnecessary fetch calls of any kind
    for ticker in tickers:
        cached = _load_ohlcv_from_db(ticker)
        if _is_cache_fresh(cached):
            result[ticker] = cached
        else:
            tv_failed.append(ticker)   # needs fresh fetch

    if not tv_failed:
        log.info(f"      All {len(tickers)} tickers fresh in cache — no fetch needed")
        return result

    # [NEW] TIER 0: Dhan — one auth check for the whole batch, not per ticker
    dhan_ok = _check_dhan_auth()
    if dhan_ok:
        log.info(f"      Fetching {len(tv_failed)} tickers via Dhan...")
        dhan_success = 0
        dhan_still_failed = []
        for ticker in tv_failed:
            # [FIX] Dhan's historical endpoint caps at 5 req/sec. All 500
            # tickers mapped correctly (confirmed via diagnose_dhan_mapping.py)
            # yet only 219/500 succeeded in an unthrottled run — the gap was
            # rate-limiting, not missing security_ids. 0.25s/call keeps this
            # at 4/sec, safely under the cap.
            time.sleep(0.25)
            df = fetch_dhan(ticker, n_bars=n_bars)
            if not df.empty and len(df) >= 20:
                _store_ohlcv(ticker, df)
                result[ticker] = df
                dhan_success += 1
            else:
                dhan_still_failed.append(ticker)
        _dhan_status["tickers_from_dhan"] = dhan_success
        if dhan_success:
            log.info(f"      Dhan: {dhan_success} fetched successfully")
        tv_failed = dhan_still_failed
        if not tv_failed:
            return result   # Dhan covered everything — no need to touch tvDatafeed/Yahoo
        log.info(f"      {len(tv_failed)} tickers not covered by Dhan "
                 f"(missing security_id mapping or per-ticker error) — "
                 f"falling through to tvDatafeed")

    log.info(f"      Fetching {len(tv_failed)} tickers via tvDatafeed...")

    tv_success = 0
    still_failed = []

    for ticker in tv_failed:
        df = fetch_tv(ticker, n_bars=n_bars)
        if not df.empty and len(df) >= 20:
            _store_ohlcv(ticker, df)
            result[ticker] = df
            tv_success += 1
        else:
            still_failed.append(ticker)

    if tv_success:
        log.info(f"      tvDatafeed: {tv_success} fetched successfully")

    # Fallback to Yahoo for any tvDatafeed failures
    if still_failed:
        log.info(f"      Yahoo fallback for {len(still_failed)} tickers "
                 f"tvDatafeed couldn't provide...")
        yahoo_result = _fetch_batch_yahoo(still_failed)
        result.update(yahoo_result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3: Yahoo Finance — history builder / fallback only
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_batch_yahoo(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    result = {}
    try:
        import yfinance as yf
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            chunk = tickers[i:i + batch_size]
            raw   = yf.download(
                chunk, period=period, auto_adjust=True,
                progress=False, group_by="ticker", threads=True
            )
            for ticker in chunk:
                try:
                    df = (raw[["Open","High","Low","Close","Volume"]].dropna()
                          if len(chunk) == 1
                          else raw[ticker][["Open","High","Low","Close","Volume"]].dropna())
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    if len(df) >= 20:
                        _store_ohlcv(ticker, df)
                        result[ticker] = df
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"Yahoo batch fetch error: {e}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public fetch API (called by scanner.py)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_batch_ohlcv(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """
    Main entry point for batch OHLCV fetch.
    Uses tvDatafeed as primary, Yahoo as fallback.
    """
    return fetch_batch_tv(tickers)


def fetch_single(ticker: str, period: str = "2y") -> pd.DataFrame:
    cached = _load_ohlcv_from_db(ticker)
    if _is_cache_fresh(cached):
        return cached
    df = fetch_tv(ticker)
    if not df.empty:
        _store_ohlcv(ticker, df)
        return df
    # Yahoo fallback
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if not df.empty:
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            _store_ohlcv(ticker, df)
            return df
    except Exception as e:
        log.debug(f"Yahoo single fetch error {ticker}: {e}")
    return cached   # return stale cache as last resort


# ─────────────────────────────────────────────────────────────────────────────
# Market context (indices + VIX — Yahoo only, no cache needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_market_context() -> dict:
    """
    [2026-08-18] ctx["vix"] used to default to a hardcoded 15.0 with no way
    for a caller to tell "real VIX happened to be 15.0" apart from "fetch
    failed, used the fallback constant" -- that fabricated value fed
    straight into regime classification scoring as if it were real. Now
    carries ctx["vix_is_fallback"]: True until a real ^INDIAVIX fetch
    actually succeeds; callers (orchestrator.py's regime step, emailer.py's
    market-intelligence section) must render/log this explicitly rather
    than trusting a bare vix number.
    """
    ctx = {
        "nifty50": pd.DataFrame(), "banknifty": pd.DataFrame(),
        "nifty500": pd.DataFrame(), "vix": 15.0, "vix_is_fallback": True,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import yfinance as yf
        n50  = yf.download("^NSEI",     period="2y", auto_adjust=True, progress=False)
        bn   = yf.download("^NSEBANK",  period="2y", auto_adjust=True, progress=False)
        vix  = yf.download("^INDIAVIX", period="5d", auto_adjust=True, progress=False)
        n500 = yf.download("^CRSLDX",   period="2y", auto_adjust=True, progress=False)
        for df in [n50, bn, vix, n500]:
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
        ctx["nifty50"]        = n50;   ctx["nifty50_data"]   = n50
        ctx["banknifty"]      = bn;    ctx["banknifty_data"] = bn
        ctx["nifty500"]       = n500;  ctx["nifty500_data"]  = n500
        if not vix.empty and "Close" in vix.columns:
            v = vix["Close"].squeeze()
            ctx["vix"] = float(v.iloc[-1]) if hasattr(v, "iloc") else float(v)
            ctx["vix_is_fallback"] = False
    except Exception as e:
        log.warning(f"Market context fetch error: {e} — VIX will use fallback "
                     f"default 15.0, flagged via ctx['vix_is_fallback']=True")
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3 FIX: Bhavcopy → SQLite injection (authoritative CMP)
# ─────────────────────────────────────────────────────────────────────────────

def inject_bhavcopy_to_db(full_df: pd.DataFrame, trading_date: str) -> int:
    """
    Write NSE official closing prices into SQLite for every EQ-series stock.
    Called automatically after each Bhavcopy parse.
    Makes Bhavcopy the authoritative CMP source — overrides any stale Yahoo data.
    """
    if full_df is None or full_df.empty:
        return 0

    cols      = list(full_df.columns)
    sym_col   = next((c for c in cols if c == "SYMBOL"),  None)
    ser_col   = next((c for c in cols if c == "SERIES"),  None)
    open_col  = next((c for c in cols if c in ("OPEN_PRICE",  "OPEN")),  None)
    high_col  = next((c for c in cols if c in ("HIGH_PRICE",  "HIGH")),  None)
    low_col   = next((c for c in cols if c in ("LOW_PRICE",   "LOW")),   None)
    close_col = next((c for c in cols if c in ("CLOSE_PRICE", "CLOSE", "CLOSING_PRICE")), None)
    vol_col   = next((c for c in cols if c in ("TTL_TRD_QNTY", "TOTAL_TRADED_QUANTITY",
                                                "TOTTRDQTY", "VOLUME")), None)

    if not sym_col or not close_col:
        log.warning(f"inject_bhavcopy_to_db: cannot find SYMBOL or CLOSE_PRICE. "
                    f"Cols: {cols[:15]}")
        return 0

    df = full_df.copy()
    if ser_col:
        df = df[df[ser_col].str.strip() == "EQ"]

    for col in [open_col, high_col, low_col, close_col, vol_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[close_col])

    rows = []
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip()
        rows.append((
            sym + ".NS",
            trading_date,
            float(row[open_col])  if open_col  and pd.notna(row.get(open_col))  else 0.0,
            float(row[high_col])  if high_col  and pd.notna(row.get(high_col))  else 0.0,
            float(row[low_col])   if low_col   and pd.notna(row.get(low_col))   else 0.0,
            float(row[close_col]),
            float(row[vol_col])   if vol_col   and pd.notna(row.get(vol_col))   else 0.0,
        ))

    if not rows:
        return 0

    conn = _get_db()
    conn.executemany(
        "INSERT OR REPLACE INTO price_history "
        "(ticker,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-5 FIX: CMP cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_cmp_vs_bhavcopy(stock_data: dict, bhavcopy_cmp_map: dict,
                              threshold_pct: float = 1.0) -> dict:
    """
    BUG-5 FIX: Cross-check each stock's last close against Bhavcopy official close.
    Logs WARNING for deviations > threshold_pct.
    Returns failures dict for audit.
    """
    failures = {}
    for ticker, df in stock_data.items():
        if df.empty:
            continue
        yahoo_close = float(df["Close"].iloc[-1])
        bhavcopy    = bhavcopy_cmp_map.get(ticker)
        if bhavcopy and bhavcopy > 0:
            delta_pct = abs(yahoo_close - bhavcopy) / bhavcopy * 100
            if delta_pct > threshold_pct:
                failures[ticker] = {
                    "cache":     round(yahoo_close, 2),
                    "bhavcopy":  round(bhavcopy,    2),
                    "delta_pct": round(delta_pct,   2),
                }
                log.warning(
                    f"  CMP MISMATCH {ticker}: "
                    f"cache={yahoo_close:.1f} bhavcopy={bhavcopy:.1f} "
                    f"Δ={delta_pct:.1f}%"
                )

    if failures:
        log.warning(f"  {len(failures)} CMP mismatches > {threshold_pct}%. "
                    f"Bhavcopy injection should have corrected these in DB.")
    else:
        log.info(f"  CMP validation: all stocks within {threshold_pct}% ✓")
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# BUG-6 SUPPORT: bhavcopy_cmp_map builder (used by near_breakout.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_bhavcopy_cmp_map(full_df: pd.DataFrame) -> Dict[str, float]:
    """
    BUG-6 FIX support: build {TICKER.NS: close_price} from Bhavcopy full_df.
    Passed to find_near_breakout_stocks() so it uses official CMP not stale cache.
    """
    if full_df is None or full_df.empty:
        return {}

    cols      = list(full_df.columns)
    sym_col   = next((c for c in cols if c == "SYMBOL"), None)
    close_col = next((c for c in cols if c in ("CLOSE_PRICE","CLOSE","CLOSING_PRICE")), None)

    if not sym_col or not close_col:
        return {}

    result = {}
    for _, row in full_df.iterrows():
        try:
            sym   = str(row[sym_col]).strip()
            price = float(row[close_col])
            if price > 0:
                result[sym + ".NS"] = price
        except (ValueError, TypeError):
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BhavcopyFetcher
# ─────────────────────────────────────────────────────────────────────────────

class BhavcopyFetcher:
    """
    Downloads NSE Bhavcopy (sec_bhavdata_full).

    v5.4: After parsing, automatically:
      - Populates self.full_df (fixed column names — BUG-1)
      - Injects official closes into SQLite (BUG-3)
      - Exposes self.bhavcopy_cmp_map for near-breakout use (BUG-6)
      - Exposes self.trading_date for audit
    """
    CACHE_DIR = DATA_DIR / "bhavcopy_cache"
    BASE_URL  = ("https://archives.nseindia.com/products/content/"
                 "sec_bhavdata_full_{date}.csv")

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.full_df:          Optional[pd.DataFrame]  = None
        self.trading_date:     Optional[str]           = None
        self.bhavcopy_cmp_map: Dict[str, float]        = {}

    def get_delivery_pct(self, expected_trading_date: date = None) -> tuple:
        """
        [2026-08-18] Was: loop back up to 5 calendar days and silently
        return the FIRST file that parsed, regardless of which date it
        actually was -- self.trading_date was computed but never gated on,
        so a stale 3-4-day-old Bhavcopy (e.g. today's not published yet)
        was indistinguishable from a fresh one to every caller.

        Now: requires an exact match against `expected_trading_date`
        (defaults to get_last_trading_day() -- the holiday-aware "what
        date should this data be" answer). Still checks the same 5-day
        cache/download window (a file for the right date might be sitting
        in cache under an earlier offset, or need downloading), but only
        ACCEPTS a result whose parsed iso_date equals expected_trading_date
        exactly -- "close" is not "correct" for data a trading decision is
        based on.

        Returns (delivery_dict, DataProvenance) -- callers MUST check
        provenance.ok before trusting delivery_dict; an empty {} on
        failure looks identical to "no delivery data today" unless the
        provenance is inspected too.
        """
        from market_calendar.staleness_check import verify_bhavcopy_date
        expected = expected_trading_date or get_last_trading_day()
        expected_str = expected.strftime("%Y-%m-%d")

        for offset in range(0, 5):
            dt = datetime.today() - timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            date_str   = dt.strftime("%d%m%Y")
            iso_date   = dt.strftime("%Y-%m-%d")
            if iso_date != expected_str:
                continue   # not the date we need -- don't even bother parsing it
            cache_file = self.CACHE_DIR / f"bhav_{date_str}.csv"

            if cache_file.exists():
                result = self._parse(cache_file, iso_date)
                if result:
                    provenance = verify_bhavcopy_date(self.trading_date, expected)
                    return result, provenance

            url = self.BASE_URL.format(date=date_str)
            try:
                r = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    timeout=15
                )
                if r.status_code == 200 and len(r.content) > 1000:
                    cache_file.write_bytes(r.content)
                    result = self._parse(cache_file, iso_date)
                    if result:
                        provenance = verify_bhavcopy_date(self.trading_date, expected)
                        return result, provenance
            except Exception as e:
                log.debug(f"Bhavcopy fetch {date_str}: {e}")

        log.warning(f"Bhavcopy for expected trading date {expected_str} unavailable "
                     f"(not published yet, or archive unreachable) -- refusing to "
                     f"silently substitute an older file")
        return {}, verify_bhavcopy_date(None, expected)

    def _parse(self, path: Path, iso_date: str) -> Dict[str, float]:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.upper()
            cols = list(df.columns)

            # BUG-1 FIX: correct column search (CLOSE_PRICE not CLOSE)
            close_col = next((c for c in cols if c in
                              ("CLOSE_PRICE", "CLOSE", "CLOSING_PRICE")), None)
            prev_col  = next((c for c in cols if c in
                              ("PREV_CLOSE", "PREVCLOSE", "PCLOSE",
                               "PREV_CLOSING_PRICE")), None)

            if close_col and prev_col:
                full = df.copy()
                full[close_col] = pd.to_numeric(full[close_col], errors="coerce")
                full[prev_col]  = pd.to_numeric(full[prev_col],  errors="coerce")
                self.full_df      = full.dropna(subset=[close_col, prev_col])
                self.trading_date = iso_date

                # BUG-3 FIX: inject authoritative closes into SQLite
                injected = inject_bhavcopy_to_db(self.full_df, iso_date)

                # BUG-6 FIX: build CMP map for near-breakout scanner
                self.bhavcopy_cmp_map = build_bhavcopy_cmp_map(self.full_df)

                log.info(
                    f"      Bhavcopy: {len(self.full_df)} symbols | "
                    f"cols=[{close_col},{prev_col}] | "
                    f"date={iso_date} | "
                    f"injected={injected} EQ stocks into SQLite"
                )
            else:
                log.warning(
                    f"Bhavcopy missing CLOSE_PRICE or PREV_CLOSE. "
                    f"Found: {cols[:15]}"
                )
                self.full_df = None

            # Delivery % extraction (unchanged)
            sym_col = next((c for c in cols if "SYMBOL" in c), None)
            del_col = next((c for c in cols if "DELIV" in c and "PER" in c), None)
            if not sym_col or not del_col:
                return {}
            sub = df[[sym_col, del_col]].dropna()
            sub[del_col] = pd.to_numeric(sub[del_col], errors="coerce").fillna(0)
            return dict(zip(sub[sym_col].str.strip(), sub[del_col].astype(float)))

        except Exception as e:
            log.error(f"Bhavcopy parse error: {e}")
            self.full_df = None
            return {}
