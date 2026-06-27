"""
NSE Momentum v5.4 — Data Fetcher

ARCHITECTURE: Three-tier price source hierarchy
  TIER 1 (PRIMARY):   tvDatafeed  — TradingView data, same prices trader sees
  TIER 2 (OVERRIDE):  Bhavcopy    — NSE official close, injected into SQLite daily
  TIER 3 (HISTORY):   Yahoo Finance — 2yr backfill only, never used for CMP

WHY THIS ORDER:
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

import logging, sqlite3, requests
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
    """Most recent NSE trading weekday (Mon–Fri). Ignores NSE holidays."""
    d = from_date or date.today()
    for i in range(7):
        candidate = d - timedelta(days=i)
        if candidate.weekday() < 5:
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
# TIER 1: tvDatafeed — primary OHLCV source
# ─────────────────────────────────────────────────────────────────────────────

def _symbol_from_ticker(ticker: str) -> str:
    """POLYCAB.NS  →  POLYCAB"""
    return ticker.replace(".NS", "").replace(".BO", "")


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
    Fetch OHLCV for multiple tickers via tvDatafeed.
    Stores each result in SQLite so future runs use cache.
    Falls back to Yahoo for any ticker tvDatafeed fails on.
    """
    result      = {}
    tv_failed   = []

    # Check cache freshness first — avoids unnecessary TV calls
    for ticker in tickers:
        cached = _load_ohlcv_from_db(ticker)
        if _is_cache_fresh(cached):
            result[ticker] = cached
        else:
            tv_failed.append(ticker)   # needs fresh fetch

    if not tv_failed:
        log.info(f"      All {len(tickers)} tickers fresh in cache — no fetch needed")
        return result

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
    ctx = {
        "nifty50": pd.DataFrame(), "banknifty": pd.DataFrame(),
        "nifty500": pd.DataFrame(), "vix": 15.0,
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
    except Exception as e:
        log.warning(f"Market context fetch error: {e}")
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

    def get_delivery_pct(self) -> Dict[str, float]:
        for offset in range(0, 5):
            dt = datetime.today() - timedelta(days=offset)
            if dt.weekday() >= 5:
                continue
            date_str   = dt.strftime("%d%m%Y")
            iso_date   = dt.strftime("%Y-%m-%d")
            cache_file = self.CACHE_DIR / f"bhav_{date_str}.csv"

            if cache_file.exists():
                result = self._parse(cache_file, iso_date)
                if result:
                    return result

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
                        return result
            except Exception as e:
                log.debug(f"Bhavcopy fetch {date_str}: {e}")

        log.warning("Bhavcopy unavailable — delivery % defaulting to 0")
        return {}

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
