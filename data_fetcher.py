"""
NSE Momentum v4.3 — Data Fetcher
Handles: Yahoo Finance OHLCV, market context (Nifty/BankNifty/VIX), Bhavcopy delivery %
Local DB stores 2yr price history to avoid re-fetching.
"""

import os, logging, sqlite3, requests, zipfile, io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "momentum_v4.db"

# ── Local SQLite price cache ───────────────────────────────────────────────────

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            ticker TEXT,
            date   TEXT,
            open   REAL, high REAL, low REAL, close REAL, volume REAL,
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
            float(row.get("Open", 0)), float(row.get("High", 0)),
            float(row.get("Low", 0)),  float(row.get("Close", 0)),
            float(row.get("Volume", 0)),
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO price_history (ticker,date,open,high,low,close,volume)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


def _load_ohlcv_from_db(ticker: str, days: int = 504) -> pd.DataFrame:
    conn = _get_db()
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date,open,high,low,close,volume FROM price_history "
        "WHERE ticker=? AND date>=? ORDER BY date",
        (ticker, cutoff)
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def fetch_single(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV for one ticker. Tries local DB first, falls back to yfinance."""
    # Try local DB
    df = _load_ohlcv_from_db(ticker)
    if len(df) >= 100:
        return df

    # Fetch from Yahoo Finance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # Strip partial intraday bar
        from datetime import datetime as dt
        now = dt.now()
        market_open = now.hour < 15 or (now.hour == 15 and now.minute < 35)
        if market_open and not df.empty:
            today_str = now.strftime("%Y-%m-%d")
            df = df[df.index.strftime("%Y-%m-%d") < today_str]
        _store_ohlcv(ticker, df)
        return df
    except Exception as e:
        log.debug(f"yfinance error {ticker}: {e}")
        return pd.DataFrame()


def fetch_batch_ohlcv(tickers: list, period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Efficient batch with local cache."""
    result = {}
    need_fetch = []

    for ticker in tickers:
        df = _load_ohlcv_from_db(ticker)
        if len(df) >= 100:
            result[ticker] = df
        else:
            need_fetch.append(ticker)

    if need_fetch:
        log.info(f"  Fetching {len(need_fetch)} tickers from Yahoo Finance...")
        try:
            import yfinance as yf
            batch_size = 50
            for i in range(0, len(need_fetch), batch_size):
                chunk = need_fetch[i:i+batch_size]
                raw = yf.download(
                    chunk, period=period, auto_adjust=True, progress=False,
                    group_by="ticker", threads=True
                )
                for ticker in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw[["Open","High","Low","Close","Volume"]].dropna()
                        else:
                            df = raw[ticker][["Open","High","Low","Close","Volume"]].dropna()
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        # Strip partial intraday bar (market open = vol not final)
                        from datetime import datetime as dt
                        now = dt.now()
                        market_open = now.hour < 15 or (now.hour == 15 and now.minute < 35)
                        if market_open and not df.empty:
                            today_str = now.strftime("%Y-%m-%d")
                            df = df[df.index.strftime("%Y-%m-%d") < today_str]
                        if len(df) >= 20:
                            _store_ohlcv(ticker, df)
                            result[ticker] = df
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"Batch fetch error: {e}")

    return result


def get_market_context() -> dict:
    """Fetch Nifty50, BankNifty, VIX for regime detection."""
    ctx = {
        "nifty50": pd.DataFrame(),
        "banknifty": pd.DataFrame(),
        "nifty500": pd.DataFrame(),
        "vix": 15.0,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import yfinance as yf
        n50 = yf.download("^NSEI", period="2y", auto_adjust=True, progress=False)
        bn  = yf.download("^NSEBANK", period="2y", auto_adjust=True, progress=False)
        vix = yf.download("^INDIAVIX", period="5d", auto_adjust=True, progress=False)
        n500= yf.download("^CRSLDX", period="2y", auto_adjust=True, progress=False)

        for df in [n50, bn, vix, n500]:
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)

        ctx["nifty50"]      = n50
        ctx["nifty50_data"] = n50   # alias for orchestrator
        ctx["banknifty"]    = bn
        ctx["banknifty_data"] = bn  # alias
        ctx["nifty500"]     = n500
        ctx["nifty500_data"] = n500 # alias
        if not vix.empty and "Close" in vix.columns:
            vix_val = vix["Close"].squeeze()
            ctx["vix"] = float(vix_val.iloc[-1]) if hasattr(vix_val, "iloc") else float(vix_val)
    except Exception as e:
        log.warning(f"Market context fetch error: {e}")
    return ctx


class BhavcopyFetcher:
    """
    Downloads NSE Bhavcopy (sec_bhavdata_full) to get delivery %.
    Falls back to graceful empty dict if download fails.
    """
    CACHE_DIR = DATA_DIR / "bhavcopy_cache"
    BASE_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

    def __init__(self):
        self.CACHE_DIR.mkdir(exist_ok=True)

    def get_delivery_pct(self) -> Dict[str, float]:
        """Returns {SYMBOL: delivery_pct} for today or most recent available."""
        for offset in range(0, 5):
            dt = datetime.today() - timedelta(days=offset)
            if dt.weekday() >= 5:  # skip weekends
                continue
            date_str = dt.strftime("%d%m%Y")
            cache_file = self.CACHE_DIR / f"bhav_{date_str}.csv"

            # Try cache
            if cache_file.exists():
                return self._parse(cache_file)

            # Try download
            url = self.BASE_URL.format(date=date_str)
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    cache_file.write_bytes(r.content)
                    return self._parse(cache_file)
            except Exception as e:
                log.debug(f"Bhavcopy fetch {date_str}: {e}")
                continue

        log.warning("Bhavcopy unavailable — delivery % defaulting to 0")
        return {}

    def _parse(self, path: Path) -> Dict[str, float]:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            # Column names vary; find the right ones
            sym_col = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
            del_col = next((c for c in df.columns if "DELIV" in c.upper() and "PER" in c.upper()), None)
            if not sym_col or not del_col:
                return {}
            df = df[[sym_col, del_col]].dropna()
            df[del_col] = pd.to_numeric(df[del_col], errors="coerce").fillna(0)
            return dict(zip(df[sym_col].str.strip(), df[del_col].astype(float)))
        except Exception as e:
            log.debug(f"Bhavcopy parse error: {e}")
            return {}
