#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.4 — Daily Portfolio Watch
==========================================
Reads a broker holdings export (XLSX or CSV — Axis Direct, Yes Securities/Omni,
or any similar format) and evaluates every holding for trend turns, then gives
plain-language protect/build-capital advice per position.

DESIGN NOTE:
  Reuses data_fetcher.fetch_batch_ohlcv() — the SAME tvDatafeed -> yfinance ->
  bhavcopy hierarchy the scanner uses. This module does not introduce a new,
  unvetted price source; it evaluates your actual holdings against the same
  data pipeline you already trust for signals.

  This is intentionally a LOCAL script, not a GitHub Actions job — it depends
  on a file you manually download from your broker each morning. There is no
  public API for individual Axis Direct / Yes Securities holdings, so this
  can't run unattended in the cloud the way daily_scan.yml does.

Usage:
  python portfolio_watch.py --file "C:\\path\\to\\holdings.xlsx" --account HUF
  python portfolio_watch.py --file huf.xlsx --account HUF --file maya.xlsx --account Maya
  python portfolio_watch.py --file huf.xlsx --account HUF --email
  python portfolio_watch.py --file huf.xlsx --account HUF --out report.html

Requires: pandas, numpy, openpyxl (for .xlsx), plus your existing data_fetcher.py
  pip install openpyxl
"""

import argparse
import difflib
import os
import sys
import smtplib
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    from data_fetcher import fetch_batch_ohlcv
except ImportError:
    print("ERROR: could not import fetch_batch_ohlcv from data_fetcher.py.")
    print("Place this file in the same folder as data_fetcher.py (your repo root).")
    sys.exit(1)

try:
    from loguru import logger as log
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

VERSION = "1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Broker file parsing — column names vary by broker, so match flexibly
# ─────────────────────────────────────────────────────────────────────────────

TICKER_COLS = ["symbol", "scrip", "scrip name", "ticker", "stock name", "stock",
               "instrument", "instrument name", "trading symbol"]
QTY_COLS    = ["qty", "quantity", "holding qty", "net qty", "quantity available", "open qty"]
AVG_COLS    = ["avg cost", "avg price", "buy avg", "average price",
               "avg. cost price", "average cost price", "buy average"]
ISIN_COLS   = ["isin", "isin number", "isin code"]

# Rows that are totals/footers, not real holdings — never treat as a ticker
_JUNK_TICKERS = {"", "NAN", "TOTAL", "GRAND TOTAL", "NET TOTAL", "SUBTOTAL"}

NSE_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
_MASTER_CACHE_PATH = BASE_DIR / "data" / "nse_equity_master.csv"
_ETF_CACHE_PATH = BASE_DIR / "data" / "nse_etf_master.csv"
_MASTER_CACHE_MAX_AGE_DAYS = 7


def _download_and_cache(url: str, cache_path: Path) -> pd.DataFrame | None:
    try:
        if cache_path.exists():
            age_days = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 86400
            if age_days < _MASTER_CACHE_MAX_AGE_DAYS:
                return _read_csv_flexible_encoding(cache_path)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_bytes(resp.content)
        return _read_csv_flexible_encoding(cache_path)

    except Exception as e:
        log.warning(f"  Could not fetch {url} ({e}).")
        if cache_path.exists():
            try:
                return _read_csv_flexible_encoding(cache_path)
            except Exception:
                pass
        return None


def _read_csv_flexible_encoding(path: Path) -> pd.DataFrame:
    """NSE's CSVs aren't always pure UTF-8 (stray Windows-1252 characters
    like em-dashes show up occasionally) — try utf-8 first, fall back to
    cp1252/latin-1 rather than crashing the whole ticker-resolution step
    over one bad byte in an unrelated column."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def _load_nse_master() -> tuple:
    """
    Broker exports (Axis Direct in particular) give company names and ISINs,
    not NSE ticker symbols — 'Anant Raj Ltd', not 'ANANTRAJ'. Passing a company
    name straight to fetch_batch_ohlcv() would silently fetch nothing.

    Downloads TWO NSE master lists — equities (EQUITY_L.csv) and ETFs
    (eq_etfseclist.csv) — since ETF holdings (e.g. SILVERETF) don't appear
    in the equity-only list. Caches both locally for a week.

    IMPORTANT: builds isin_map/name_map SEPARATELY per source and merges the
    resulting DICTS, rather than pd.concat-ing the two raw DataFrames. The
    two files use slightly different column casing (e.g. 'SYMBOL' vs
    'Symbol'), and concat treats those as distinct columns — a prior version
    of this function did concat first, which caused a column-name collision
    in _find_col that silently wiped out equity resolution entirely. Merging
    already-built dicts sidesteps that class of bug completely.

    Returns (isin_map, name_map) — equity entries take priority; ETF entries
    fill in any keys equities didn't already provide. Returns ({}, {}) if
    nothing could be loaded — callers must handle that gracefully.
    """
    log.info("  Loading NSE securities master (equities + ETFs, cached weekly)...")
    equity_df = _download_and_cache(NSE_MASTER_URL, _MASTER_CACHE_PATH)
    etf_df    = _download_and_cache(NSE_ETF_MASTER_URL, _ETF_CACHE_PATH)

    isin_map, name_map = {}, {}
    for df in (etf_df, equity_df):  # ETF first, equity second — equity wins on key collision
        if df is None:
            continue
        i_map, n_map = _build_isin_and_name_maps(df)
        isin_map.update(i_map)
        name_map.update(n_map)

    return isin_map, name_map


def _build_isin_and_name_maps(master: pd.DataFrame) -> tuple:
    isin_col = _find_col(master, ["isin number", "isin"])
    sym_col  = _find_col(master, ["symbol"])
    name_col = _find_col(master, ["name of company", "underlying", "company name", "name"])
    isin_map = {}
    name_map = {}

    if isin_col and sym_col:
        sub = master[[isin_col, sym_col]].dropna()
        isin_map = dict(zip(sub[isin_col].astype(str).str.strip(),
                             sub[sym_col].astype(str).str.strip()))
    if name_col and sym_col:
        sub = master[[name_col, sym_col]].dropna()
        # Drop blank/whitespace-only names too — an empty string is a valid
        # dict key that would otherwise fuzzy-match against everything.
        sub = sub[sub[name_col].astype(str).str.strip() != ""]
        name_map = dict(zip(sub[name_col].astype(str).str.strip(),
                             sub[sym_col].astype(str).str.strip()))
    return isin_map, name_map


def resolve_tickers(raw_names: pd.Series, isins: pd.Series = None) -> tuple:
    """
    Resolve broker-supplied company names (+ optional ISINs) to real NSE
    ticker symbols. ISIN match is exact and preferred; fuzzy company-name
    matching is a fallback ONLY — and is explicitly flagged as such, because
    a silent wrong-stock match on a capital-protection tool is a serious
    problem, not a cosmetic one (e.g. 'ICICI Prudential Life Insurance' vs
    'ICICI Prudential Asset Management', or 'Bajaj Finance' vs 'Bajaj
    Finserv', are different companies that share a name prefix — fuzzy
    matching can confuse them).

    AMBIGUITY CHECK: the 0.75 cutoff alone only tells you the BEST match
    cleared a bar — it says nothing about whether a close runner-up existed
    that could just as easily have been picked. This pulls the top 2
    candidates and flags AMBIGUOUS_FUZZY when they're within 0.08 similarity
    of each other, which is the actual signature of a same-prefix collision
    risk, not just "did the top match clear the threshold."

    Returns (resolved_tickers: pd.Series, methods: pd.Series) where methods
    is one of: ISIN_EXACT, FUZZY_NAME, AMBIGUOUS_FUZZY, UNRESOLVED — always
    inspect FUZZY_NAME and AMBIGUOUS_FUZZY rows before trusting their advice.
    """
    isin_map, name_map = _load_nse_master()
    if not isin_map and not name_map:
        log.warning("  No NSE master list available — using broker names as-is.")
        return raw_names, pd.Series(["UNRESOLVED"] * len(raw_names), index=raw_names.index)

    name_keys = list(name_map.keys())
    AMBIGUITY_MARGIN = 0.08   # runner-up within this of the top match = flag it

    resolved, methods = [], []
    for i, raw_name in enumerate(raw_names):
        isin = str(isins.iloc[i]).strip() if isins is not None else None
        symbol, method = None, "UNRESOLVED"

        if isin and isin in isin_map:
            symbol, method = isin_map[isin], "ISIN_EXACT"
        else:
            # Stricter cutoff (0.75, up from 0.6) — a same-prefix collision
            # like the ICICI Prudential example above scores dangerously
            # high on loose fuzzy matching. Better to leave it UNRESOLVED
            # and force a manual check than silently pick the wrong company.
            candidates = difflib.get_close_matches(str(raw_name), name_keys, n=2, cutoff=0.75)
            if candidates:
                symbol = name_map[candidates[0]]
                if len(candidates) == 2:
                    top_ratio = difflib.SequenceMatcher(None, str(raw_name), candidates[0]).ratio()
                    runner_ratio = difflib.SequenceMatcher(None, str(raw_name), candidates[1]).ratio()
                    if (top_ratio - runner_ratio) <= AMBIGUITY_MARGIN:
                        method = "AMBIGUOUS_FUZZY"
                        log.warning(f"  AMBIGUOUS match for '{raw_name}': top candidate "
                                    f"'{candidates[0]}' ({top_ratio:.2f}) vs runner-up "
                                    f"'{candidates[1]}' ({runner_ratio:.2f}) — too close to "
                                    f"trust automatically, flagging for manual review.")
                    else:
                        method = "FUZZY_NAME"
                else:
                    method = "FUZZY_NAME"

        if symbol:
            resolved.append(symbol)
            methods.append(method)
        else:
            log.warning(f"  Could not resolve ticker for '{raw_name}' "
                        f"(ISIN={isin}) — using raw name, price fetch will likely fail.")
            resolved.append(raw_name)
            methods.append(method)

    return pd.Series(resolved, index=raw_names.index), pd.Series(methods, index=raw_names.index)


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for cand in candidates:
        for lc, orig in cols_lower.items():
            if cand in lc:
                return orig
    return None


def _detect_header_row(path: Path, max_scan_rows: int = 25) -> int:
    """
    Broker exports (Axis Direct in particular) often have several title/
    metadata rows — report name, account number, generation date — sitting
    above the real header row. Scan the first max_scan_rows+1 rows (no header
    assumed) and return the index of the first row containing a recognizable
    ticker/qty header keyword. Returns 0 if nothing matches, so well-formed
    files are unaffected.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        preview = pd.read_excel(path, header=None, nrows=max_scan_rows + 1)
    else:
        preview = pd.read_csv(path, header=None, nrows=max_scan_rows + 1)

    all_candidates = TICKER_COLS + QTY_COLS
    for i in range(len(preview)):
        row_cells = [str(c).lower().strip() for c in preview.iloc[i].tolist()]
        for cell in row_cells:
            if any(cand in cell for cand in all_candidates):
                return i
    return 0


def _read_table(path: Path, header_row: int) -> pd.DataFrame:
    """Read xlsx/xls/csv with the detected header row, with a forgiving
    fallback for malformed CSVs (ragged rows, embedded commas in free text)."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header_row)
    try:
        return pd.read_csv(path, header=header_row)
    except pd.errors.ParserError as e:
        log.warning(f"  [{path.name}] Standard CSV parse failed ({e}). "
                    f"Retrying with a more forgiving parser...")
        try:
            return pd.read_csv(path, header=header_row, engine="python", on_bad_lines="skip")
        except Exception:
            raise ValueError(
                f"Could not parse {path.name} as a holdings file — the row "
                f"structure is inconsistent even with forgiving parsing. "
                f"This often means the file is a TRADE/TRANSACTION log "
                f"(buy/sell history) rather than a HOLDINGS/POSITIONS export "
                f"— they have different structures. Check your broker portal "
                f"for a 'Holdings' or 'Portfolio' report instead of a "
                f"'Trade Summary' or 'Contract Note' report."
            )


def load_holdings(filepath: str) -> pd.DataFrame:
    """Load a broker holdings export (xlsx/xls/csv) into a normalized DataFrame
    with columns: ticker (resolved NSE symbol), qty, avg_price."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        header_row = _detect_header_row(path)
    except pd.errors.ParserError:
        header_row = 0  # let _read_table's fallback logic handle it below

    if header_row > 0:
        log.info(f"  [{path.name}] Detected real header at row {header_row} "
                 f"(skipped {header_row} title/metadata row(s)).")

    raw = _read_table(path, header_row)

    ticker_col = _find_col(raw, TICKER_COLS)
    qty_col    = _find_col(raw, QTY_COLS)
    avg_col    = _find_col(raw, AVG_COLS)
    isin_col   = _find_col(raw, ISIN_COLS)

    if not ticker_col or not qty_col:
        raise ValueError(
            f"Could not detect ticker/quantity columns in {path.name} "
            f"(tried header row {header_row}).\n"
            f"Columns found: {list(raw.columns)}\n"
            f"This may be a trade-log/transaction export rather than a "
            f"holdings/positions export — check your broker portal for the "
            f"correct report type, or tell me the actual header row number."
        )

    # ── Filter to valid rows FIRST, before ticker resolution ────────────────
    # Broker exports have blank padding rows and multi-paragraph disclaimer
    # text below the real data — resolving tickers for those wastes time and
    # produces noisy warnings for rows that get dropped anyway. Filter on
    # qty being genuinely numeric and positive before doing any name/ISIN
    # lookup work.
    qty_numeric = pd.to_numeric(raw[qty_col], errors="coerce")
    valid_mask = qty_numeric.notna() & (qty_numeric > 0)

    raw_names = raw.loc[valid_mask, ticker_col].astype(str).str.strip()
    isins     = raw.loc[valid_mask, isin_col].astype(str).str.strip() if isin_col else None
    qty       = qty_numeric.loc[valid_mask]
    avg_price = pd.to_numeric(raw.loc[valid_mask, avg_col], errors="coerce") if avg_col else pd.Series(np.nan, index=raw_names.index)

    # Drop obvious junk rows (totals, blanks) that happened to have a
    # numeric-looking qty coincidentally, before spending API/network calls
    # resolving them.
    junk_mask = raw_names.str.upper().isin(_JUNK_TICKERS) | (raw_names.str.len() > 60)
    raw_names, isins, qty, avg_price = (
        raw_names[~junk_mask], isins[~junk_mask] if isins is not None else None,
        qty[~junk_mask], avg_price[~junk_mask]
    )

    if raw_names.empty:
        log.warning(f"  [{path.name}] No valid holdings rows found after filtering.")
        return pd.DataFrame(columns=["ticker", "qty", "avg_price"])

    # If the ticker column already looks like real NSE symbols (short, no
    # spaces, uppercase-ish — e.g. from a broker that exports symbols
    # directly), skip resolution. Otherwise resolve company names -> tickers.
    looks_like_symbols = raw_names.str.match(r"^[A-Za-z0-9&\-]{1,20}$").mean() > 0.8
    if looks_like_symbols:
        tickers = raw_names.str.upper()
        methods = pd.Series(["DIRECT_SYMBOL"] * len(raw_names), index=raw_names.index)
    else:
        log.info(f"  [{path.name}] Ticker column looks like company names, not "
                 f"symbols — resolving via NSE master list...")
        tickers, methods = resolve_tickers(raw_names, isins)

    out = pd.DataFrame({
        "ticker":            tickers.str.strip().str.upper().values,
        "raw_name":          raw_names.values,
        "qty":               qty.values,
        "avg_price":         avg_price.values,
        "resolution_method": methods.values,
    })
    out = out.dropna(subset=["ticker", "qty"])
    out = out[out["qty"] > 0]
    out = out[~out["ticker"].isin(_JUNK_TICKERS)]
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators — plain pandas/numpy, no extra dependencies
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _get_ohlc_cols(df: pd.DataFrame) -> tuple:
    """tvDatafeed uses lowercase, yfinance uses capitalized — handle both."""
    close_col  = "close" if "close" in df.columns else "Close"
    volume_col = "volume" if "volume" in df.columns else "Volume"
    return close_col, volume_col


def classify_trend(df: pd.DataFrame) -> dict:
    """
    Rule-based trend classifier. Returns status, numeric score, and a list of
    plain-language reasons — every classification is explainable, not a black box.
    """
    if df is None or len(df) < 60:
        return {"status": "NO_DATA", "score": 0, "reasons": ["Insufficient price history (<60 bars)"]}

    close_col, vol_col = _get_ohlc_cols(df)
    close  = df[close_col].astype(float)
    volume = df[vol_col].astype(float) if vol_col in df.columns else pd.Series([0] * len(df))

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi14 = _rsi(close, 14)
    macd  = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd, 9)
    macd_hist   = macd - macd_signal

    cmp       = float(close.iloc[-1])
    e20       = float(ema20.iloc[-1])
    e50       = float(ema50.iloc[-1])
    rsi       = float(rsi14.iloc[-1])
    hist_now  = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-4]) if len(macd_hist) > 4 else hist_now

    vol5  = float(volume.tail(5).mean())
    vol20 = float(volume.tail(20).mean())
    vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0

    high20 = float(close.tail(20).max())
    low20  = float(close.tail(20).min())
    pct_from_high = (cmp - high20) / high20 * 100
    pct_from_low  = (cmp - low20) / low20 * 100

    score, reasons = 0.0, []

    # 1. Trend structure (EMA stack)
    if cmp > e20 > e50:
        score += 2; reasons.append("Price above both 20 & 50-EMA — uptrend intact")
    elif cmp < e20 < e50:
        score -= 2; reasons.append("Price below both 20 & 50-EMA — downtrend intact")
    elif cmp > e50 >= e20:
        score -= 1; reasons.append("Lost 20-EMA support, still above 50-EMA")
    else:
        reasons.append("Mixed EMA structure — no clear trend")

    # 2. Momentum direction (MACD histogram slope)
    if hist_now > 0 and hist_now < hist_prev:
        score -= 1; reasons.append("MACD histogram positive but fading — momentum slowing")
    elif hist_now < 0 and hist_now > hist_prev:
        score += 1; reasons.append("MACD histogram negative but improving — possible bottoming")
    elif hist_now > hist_prev:
        score += 1
    else:
        score -= 1

    # 3. RSI context
    if rsi >= 70:
        score -= 0.5; reasons.append(f"RSI {rsi:.0f} — overbought / extended")
    elif rsi <= 35:
        score -= 1; reasons.append(f"RSI {rsi:.0f} — weak momentum")
    elif rsi >= 55:
        score += 1

    # 4. Volume confirmation (is the move backed by participation?)
    if vol_ratio > 1.3 and cmp > e20:
        score += 1; reasons.append(f"Volume expanding ({vol_ratio:.1f}x 20-day avg) on strength")
    elif vol_ratio > 1.3 and cmp < e20:
        score -= 1; reasons.append(f"Volume expanding ({vol_ratio:.1f}x 20-day avg) on weakness — distribution")

    # 5. Proximity to recent range extremes
    if pct_from_low < 2:
        score -= 1; reasons.append(f"Within 2% of 20-day low ({pct_from_low:.1f}%) — breakdown risk")
    if pct_from_high > -2:
        score += 0.5; reasons.append("Within 2% of 20-day high — near breakout zone")

    if score >= 3:
        status = "BULLISH"
    elif score >= 1:
        status = "NEUTRAL_BULLISH"
    elif score > -1:
        status = "NEUTRAL"
    elif score > -3:
        status = "WEAKENING"
    else:
        status = "BEARISH"

    return {
        "status": status, "score": round(score, 1), "reasons": reasons,
        "cmp": cmp, "ema20": round(e20, 2), "ema50": round(e50, 2),
        "rsi": round(rsi, 1), "vol_ratio": round(vol_ratio, 2),
        "pct_from_20d_high": round(pct_from_high, 1), "pct_from_20d_low": round(pct_from_low, 1),
    }


def generate_advice(trend: dict, avg_price: float) -> str:
    """Combine trend status with your actual entry price to give
    protect-capital vs build-capital guidance — same trend can mean opposite
    actions depending on whether you're sitting on profit or loss."""
    status = trend["status"]
    cmp    = trend.get("cmp")
    if cmp is None or not avg_price or pd.isna(avg_price):
        pnl_pct = None
    else:
        pnl_pct = (cmp - avg_price) / avg_price * 100

    if status == "NO_DATA":
        return "Could not fetch price data — verify the ticker symbol."

    if status == "BULLISH":
        if pnl_pct is not None and pnl_pct > 15:
            return "Trend intact, strong profit — trail stop up to lock gains. No need to exit."
        return "Trend intact — hold. Room to build if conviction is high and sizing allows."

    if status == "NEUTRAL_BULLISH":
        return "Constructive but not confirmed — hold, wait for volume before adding."

    if status == "NEUTRAL":
        return "No clear edge either way — hold with existing stop, don't add here."

    if status == "WEAKENING":
        if pnl_pct is not None and pnl_pct > 0:
            return "Momentum fading while still in profit — consider trimming 30-50% to protect gains."
        return "Momentum fading and underwater — tighten stop, do not average down."

    if status == "BEARISH":
        if pnl_pct is not None and pnl_pct > 0:
            return "Trend has turned down despite being in profit — protect capital: exit or hard stop at cost."
        return "Trend down and in loss — capital-protection situation. Exit unless this is a deliberate thesis hold."

    return "Unable to classify."


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_portfolio(filepath: str, account_name: str = "") -> pd.DataFrame:
    holdings = load_holdings(filepath)
    if holdings.empty:
        log.warning(f"No holdings parsed from {filepath}")
        return pd.DataFrame()

    tickers_ns = [t if t.endswith(".NS") else t + ".NS" for t in holdings["ticker"]]
    log.info(f"[{account_name or filepath}] Fetching OHLCV for {len(tickers_ns)} tickers...")
    ohlcv = fetch_batch_ohlcv(tickers_ns, period="6mo")

    rows = []
    for _, row in holdings.iterrows():
        t_ns  = row["ticker"] if row["ticker"].endswith(".NS") else row["ticker"] + ".NS"
        df    = ohlcv.get(t_ns)
        trend = classify_trend(df)
        advice = generate_advice(trend, row["avg_price"])
        cmp = trend.get("cmp")
        pnl_pct = None
        if cmp is not None and row["avg_price"] and not pd.isna(row["avg_price"]):
            pnl_pct = round((cmp - row["avg_price"]) / row["avg_price"] * 100, 1)

        rows.append({
            "account":            account_name,
            "ticker":             row["ticker"],
            "raw_name":           row.get("raw_name", row["ticker"]),
            "resolution_method":  row.get("resolution_method", "DIRECT_SYMBOL"),
            "qty":                row["qty"],
            "avg_price":          row["avg_price"],
            "cmp":                cmp,
            "pnl_pct":            pnl_pct,
            "status":             trend["status"],
            "score":              trend.get("score"),
            "rsi":                trend.get("rsi"),
            "reasons":            "; ".join(trend.get("reasons", [])),
            "advice":             advice,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting — console + HTML
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ORDER = {"BEARISH": 0, "WEAKENING": 1, "NEUTRAL": 2, "NEUTRAL_BULLISH": 3, "BULLISH": 4, "NO_DATA": 5}
_STATUS_COLOR = {
    "BEARISH": "#c0392b", "WEAKENING": "#e67e22", "NEUTRAL": "#7f8c8d",
    "NEUTRAL_BULLISH": "#27ae60", "BULLISH": "#1e8449", "NO_DATA": "#95a5a6",
}


def find_prefix_collisions(df: pd.DataFrame, min_prefix_len: int = 6) -> list:
    """
    Second, INDEPENDENT sanity net — separate from resolution-method
    tracking. Even a correctly-resolved ticker can sit right next to a
    company with a genuinely similar name (Bajaj Finance vs Bajaj Finserv,
    ICICI Prudential Life vs ICICI Prudential AMC) — worth a second look
    regardless of whether resolution used ISIN or fuzzy matching, since a
    broker holdings file with two similarly-named positions is exactly the
    scenario where a human reviewing the report (not just the resolver)
    might misread one for the other.

    Scans all RAW broker-supplied names in the report and flags any pair
    sharing a common prefix of at least min_prefix_len characters. Does not
    distinguish resolution method — this catches cases the ISIN-vs-fuzzy
    tracking structurally can't, since it operates on what the broker file
    actually said, not on how confidently it was resolved.

    Returns a list of (name_a, ticker_a, name_b, ticker_b, shared_prefix) tuples.
    """
    if df.empty or "raw_name" not in df.columns:
        return []

    collisions = []
    names = df[["raw_name", "ticker"]].drop_duplicates().reset_index(drop=True)
    upper_names = names["raw_name"].astype(str).str.upper().str.strip()

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = upper_names.iloc[i], upper_names.iloc[j]
            if names["ticker"].iloc[i] == names["ticker"].iloc[j]:
                continue  # same resolved stock, not a collision — e.g. odd-lot rows
            prefix_len = 0
            for ca, cb in zip(a, b):
                if ca == cb:
                    prefix_len += 1
                else:
                    break
            if prefix_len >= min_prefix_len:
                collisions.append((
                    names["raw_name"].iloc[i], names["ticker"].iloc[i],
                    names["raw_name"].iloc[j], names["ticker"].iloc[j],
                    a[:prefix_len],
                ))
    return collisions


def print_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("\n  No holdings to report.\n")
        return

    fuzzy = df[df.get("resolution_method").isin(["FUZZY_NAME", "AMBIGUOUS_FUZZY"])]
    if not fuzzy.empty:
        print("\n" + "!" * 100)
        print("  ⚠️  VERIFY THESE — resolved by approximate name match, not exact ISIN.")
        print("  Fuzzy company-name matching can confuse similarly-named companies")
        print("  (e.g. 'ICICI Prudential Life Insurance' vs 'ICICI Prudential Asset")
        print("  Management', or 'Bajaj Finance' vs 'Bajaj Finserv'). Check each one")
        print("  is the stock you actually hold before acting on its advice below.")
        print("!" * 100)
        for _, r in fuzzy.iterrows():
            tag = " *** AMBIGUOUS — close runner-up candidate existed ***" if r.get("resolution_method") == "AMBIGUOUS_FUZZY" else ""
            print(f"    '{r['raw_name']}'  →  resolved as {r['ticker']}{tag}")
        print("!" * 100)

    collisions = find_prefix_collisions(df)
    if collisions:
        print("\n" + "!" * 100)
        print("  ⚠️  SIMILAR NAMES IN YOUR PORTFOLIO — independent second check, regardless")
        print("  of how each ticker was resolved. Two holdings share a long common name")
        print("  prefix — worth a manual glance to be sure neither was misread.")
        print("!" * 100)
        for name_a, ticker_a, name_b, ticker_b, prefix in collisions:
            print(f"    '{name_a}' ({ticker_a})  vs  '{name_b}' ({ticker_b})  — shared prefix: '{prefix}'")
        print("!" * 100)

    df = df.sort_values(by="status", key=lambda s: s.map(_STATUS_ORDER))
    print("\n" + "=" * 100)
    print(f"  PORTFOLIO WATCH — {datetime.today().strftime('%d %b %Y')}")
    print("=" * 100)
    for _, r in df.iterrows():
        pnl_str = f"{r['pnl_pct']:+.1f}%" if r['pnl_pct'] is not None and pd.notna(r['pnl_pct']) else "N/A"
        cmp_str = f"{r['cmp']:.2f}" if r['cmp'] is not None and pd.notna(r['cmp']) else "N/A"
        avg_str = f"{r['avg_price']:.2f}" if r['avg_price'] is not None and pd.notna(r['avg_price']) else "N/A"
        rsi_str = f"{r['rsi']}" if r['rsi'] is not None and pd.notna(r['rsi']) else "N/A"
        method = r.get("resolution_method")
        flag = "  ⚠️ AMBIGUOUS-MATCHED" if method == "AMBIGUOUS_FUZZY" else ("  ⚠️ FUZZY-MATCHED" if method == "FUZZY_NAME" else "")
        print(f"\n  {r['ticker']:<14} [{r['account']}]   {r['status']:<17} "
              f"CMP {cmp_str}  Avg {avg_str}  P&L {pnl_str}  RSI {rsi_str}{flag}")
        print(f"    → {r['advice']}")
        print(f"    ({r['reasons']})")
    print("\n" + "=" * 100 + "\n")


def build_html_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No holdings to report.</p>"
    df = df.sort_values(by="status", key=lambda s: s.map(_STATUS_ORDER))

    rows_html = ""
    for _, r in df.iterrows():
        color   = _STATUS_COLOR.get(r["status"], "#7f8c8d")
        pnl_str = f"{r['pnl_pct']:+.1f}%" if r['pnl_pct'] is not None and pd.notna(r['pnl_pct']) else "N/A"
        cmp_str = f"{r['cmp']:.2f}" if r['cmp'] is not None and pd.notna(r['cmp']) else "N/A"
        avg_str = f"{r['avg_price']:.2f}" if r['avg_price'] is not None and pd.notna(r['avg_price']) else "N/A"
        rsi_str = f"{r['rsi']}" if r['rsi'] is not None and pd.notna(r['rsi']) else "N/A"
        rows_html += f"""
        <div style="background:#fff;border-left:4px solid {color};border-radius:4px;
                    padding:14px 18px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:15px;">{r['ticker']} <span style="color:#888;font-weight:normal;">[{r['account']}]</span></strong>
            <span style="background:{color};color:#fff;font-size:11px;padding:3px 8px;border-radius:10px;">{r['status']}</span>
          </div>
          <div style="font-size:12px;color:#555;margin-top:6px;">
            CMP ₹{cmp_str} &nbsp;|&nbsp; Avg ₹{avg_str} &nbsp;|&nbsp;
            P&amp;L {pnl_str} &nbsp;|&nbsp; RSI {rsi_str}
          </div>
          <div style="font-size:13px;margin-top:8px;color:#222;"><strong>{r['advice']}</strong></div>
          <div style="font-size:11px;color:#888;margin-top:4px;">{r['reasons']}</div>
        </div>"""

    return f"""
    <div style="background:#0d1b2a;color:#fff;padding:18px 22px;border-radius:6px 6px 0 0;">
      <div style="font-size:11px;letter-spacing:1px;color:#8fa3bf;">NSE MOMENTUM — PORTFOLIO WATCH v{VERSION}</div>
      <div style="font-size:22px;font-weight:bold;margin-top:4px;">Daily Holdings Review</div>
      <div style="font-size:12px;color:#8fa3bf;margin-top:2px;">
        {datetime.today().strftime('%d %b %Y')}
      </div>
    </div>
    <div style="background:#f4f6f8;padding:18px 22px;border-radius:0 0 6px 6px;">
      {rows_html}
    </div>
    """


def send_email(html_body: str, recipients: list) -> None:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        log.warning("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — skipping email.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Portfolio Watch — {datetime.today().strftime('%d %b %Y')}"
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, recipients, msg.as_string())
    log.info(f"Email sent to {recipients}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily portfolio trend watch")
    parser.add_argument("--file", action="append", required=True,
                         help="Path to a broker holdings export (xlsx/csv). Repeatable.")
    parser.add_argument("--account", action="append", default=[],
                         help="Label for each --file, same order (e.g. HUF, Maya).")
    parser.add_argument("--email", action="store_true", help="Send the report by email.")
    parser.add_argument("--out", default=None, help="Write HTML report to this path.")
    args = parser.parse_args()

    accounts = args.account + [""] * (len(args.file) - len(args.account))

    all_results = []
    for filepath, account in zip(args.file, accounts):
        result = evaluate_portfolio(filepath, account)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print_report(combined)

    if args.out or args.email:
        html = build_html_report(combined)
        if args.out:
            Path(args.out).write_text(html, encoding="utf-8")
            log.info(f"HTML report written to {args.out}")
        if args.email:
            recipients_path = BASE_DIR / "recipients.txt"
            if recipients_path.exists():
                recipients = [l.strip() for l in recipients_path.read_text().splitlines()
                              if l.strip() and not l.startswith("#")]
            else:
                recipients = [GMAIL_ADDRESS] if GMAIL_ADDRESS else []
            if recipients:
                send_email(html, recipients)
            else:
                log.warning("No recipients configured (recipients.txt missing and GMAIL_ADDRESS unset).")


if __name__ == "__main__":
    main()
