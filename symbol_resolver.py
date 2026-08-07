#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbol_resolver.py — P1-05

Shared NSE symbol resolution: given a broker/user-supplied company name (or
ISIN), resolve it to a real NSE ticker symbol. Extracted from
portfolio_watch.py (which had this logic already, proven against real broker
exports) so it can also be used by publish_symbol_master.py and
backfill_holdings_symbols.py without duplicating the download/match logic.

ISIN match is exact and preferred; fuzzy company-name matching is a fallback
only, with an explicit ambiguity check — see resolve_tickers()'s docstring
for why a silent wrong-stock match matters here.
"""

import difflib
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

try:
    from loguru import logger as log
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent

NSE_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
_MASTER_CACHE_PATH = BASE_DIR / "data" / "nse_equity_master.csv"
_ETF_CACHE_PATH = BASE_DIR / "data" / "nse_etf_master.csv"
_MASTER_CACHE_MAX_AGE_DAYS = 7

AMBIGUITY_MARGIN = 0.08   # runner-up within this of the top match = flag it
FUZZY_CUTOFF = 0.75       # stricter than difflib's 0.6 default — see resolve_tickers()


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


def load_nse_master() -> tuple:
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
    is one of: EXACT_SYMBOL, ISIN_EXACT, FUZZY_NAME, AMBIGUOUS_FUZZY,
    UNRESOLVED — always inspect FUZZY_NAME and AMBIGUOUS_FUZZY rows before
    trusting their advice.
    """
    isin_map, name_map = load_nse_master()
    if not isin_map and not name_map:
        log.warning("  No NSE master list available — using broker names as-is.")
        return raw_names, pd.Series(["UNRESOLVED"] * len(raw_names), index=raw_names.index)

    upper_index = build_upper_index(name_map)
    symbol_set = set(name_map.values()) | set(isin_map.values())

    resolved, methods = [], []
    for i, raw_name in enumerate(raw_names):
        isin = str(isins.iloc[i]).strip() if isins is not None else None
        symbol, method = resolve_one(str(raw_name), isin, isin_map, upper_index, symbol_set)
        resolved.append(symbol if symbol else raw_name)
        methods.append(method)
        if not symbol:
            log.warning(f"  Could not resolve ticker for '{raw_name}' "
                        f"(ISIN={isin}) — using raw name, price fetch will likely fail.")

    return pd.Series(resolved, index=raw_names.index), pd.Series(methods, index=raw_names.index)


def build_upper_index(name_map: dict) -> dict:
    """
    Uppercase-keyed lookup over name_map, built once per resolution batch.

    Broker/holdings data is routinely all-caps (Portfolio Dashboard's
    add_holding() does `.strip().upper()` on every symbol/name, and broker
    exports are frequently upper too), while NSE's master list uses title
    case ('Aster DM Quality Care Limited'). difflib.SequenceMatcher is
    case-sensitive, so matching raw vs mixed case scores ~0.3-0.4 even for
    an exact-modulo-case match — well below FUZZY_CUTOFF — while the
    uppercased comparison scores ~0.8+. Without this, the resolver silently
    fails on exactly the real-world defect case it exists to fix (holdings
    stored as 'ASTER DM QUALITY CARE' vs NSE's 'Aster DM Quality Care
    Limited'). Values are (original_name, symbol) so callers can recover
    the master's real casing for display.
    """
    index = {}
    for name, symbol in name_map.items():
        key = name.upper()
        if key not in index:   # first entry wins on a case-collision
            index[key] = (name, symbol)
    return index


def resolve_one(raw_name: str, isin: str | None, isin_map: dict, upper_index: dict,
                  symbol_set: set = frozenset()) -> tuple:
    raw_upper = raw_name.strip().upper()

    # Already a real NSE symbol (broker exports symbols directly, or a
    # holding was entered correctly to begin with) — no fuzzy matching
    # needed and none of its false-positive risk. Checked before ISIN so a
    # holding that's already a clean symbol never takes a slower path.
    if raw_upper in symbol_set:
        return raw_upper, "EXACT_SYMBOL"

    if isin and isin in isin_map:
        return isin_map[isin], "ISIN_EXACT"

    upper_keys = list(upper_index.keys())
    candidates = difflib.get_close_matches(raw_upper, upper_keys, n=2, cutoff=FUZZY_CUTOFF)
    if not candidates:
        return None, "UNRESOLVED"

    top_name, symbol = upper_index[candidates[0]]
    if len(candidates) == 2:
        top_ratio    = difflib.SequenceMatcher(None, raw_upper, candidates[0]).ratio()
        runner_ratio = difflib.SequenceMatcher(None, raw_upper, candidates[1]).ratio()
        if (top_ratio - runner_ratio) <= AMBIGUITY_MARGIN:
            runner_name = upper_index[candidates[1]][0]
            log.warning(f"  AMBIGUOUS match for '{raw_name}': top candidate "
                        f"'{top_name}' ({top_ratio:.2f}) vs runner-up "
                        f"'{runner_name}' ({runner_ratio:.2f}) — too close to "
                        f"trust automatically, flagging for manual review.")
            return symbol, "AMBIGUOUS_FUZZY"
    return symbol, "FUZZY_NAME"


def resolve_symbol(raw_name: str, isin: str = None) -> dict:
    """
    Single-input convenience wrapper around resolve_tickers() for callers
    that have one holding at a time (backfill script, publish jobs) rather
    than a whole DataFrame column. Loads the NSE master fresh each call —
    callers resolving many symbols in a loop should use resolve_tickers()
    directly (or load_nse_master() once and call resolve_one() themselves)
    to avoid redundant downloads.

    Returns {"symbol": str, "method": str, "company_name": str | None} —
    method is one of EXACT_SYMBOL/ISIN_EXACT/FUZZY_NAME/AMBIGUOUS_FUZZY/
    UNRESOLVED, same vocabulary as resolve_tickers(). company_name is the
    NSE master's own name for the resolved symbol (None if UNRESOLVED).
    """
    isin_map, name_map = load_nse_master()
    upper_index = build_upper_index(name_map)
    symbol_set = set(name_map.values()) | set(isin_map.values())
    symbol, method = resolve_one(raw_name, isin, isin_map, upper_index, symbol_set)

    company_name = None
    if symbol:
        # name_map is name -> symbol; reverse-look-up the matched name back
        # out rather than rebuild a symbol->name map for one lookup.
        for name, sym in name_map.items():
            if sym == symbol:
                company_name = name
                break

    return {
        "symbol": symbol or raw_name,
        "method": method,
        "company_name": company_name,
    }
