#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.4 — Multi-Checkpoint Confirmation Checker

CHANGES vs v5.3:
  [FIX-1] RVOL window was a fixed 09:15-10:15 (60-min) calendar slice
          regardless of when the script actually ran -- the email footer
          claimed "first-45min" (wrong on two counts: window is 60 min,
          and even at the stated 45min a 9:55am run is 5min early against
          it). Replaced with an ELAPSED-TIME-MATCHED window: today's
          volume from market open (09:15) to the current run time,
          compared against the SAME elapsed-time window on historical
          days. This is correct at ANY run time, not just after a fixed
          45/60-min mark -- which is what makes genuinely earlier
          checkpoints (see FIX-2) valid instead of just re-running the
          same broken fixed window sooner.
  [FIX-2] MULTIPLE CHECKPOINTS PER MORNING instead of one fixed 9:55am
          run. On strong-trend days, fast movers can run well past entry
          before a single 9:55am check ever sees them -- by construction,
          a pick gets exactly one chance to be caught. Now supports
          several scheduled runs through the first hour (e.g. 09:20,
          09:35, 09:55). Terminal-status picks (CONFIRMED/LOW_VOL/
          NO_VOLUME/MISSED/BROKEN) from an earlier checkpoint are cached
          in a per-day state file and NOT re-classified or re-emailed --
          only newly-terminal picks and the final summary get sent, so
          multiple checkpoints don't mean multiple duplicate emails.
  [FIX-3] REGIME-AWARE drift/extension tolerance. MAX_ENTRY_DRIFT_PCT and
          MAX_PIVOT_EXTENSION_PCT were fixed constants regardless of
          market regime -- meaning a genuine Regime A (STRONG BULL)
          momentum continuation got graded by the same "don't chase"
          tolerance as a choppy Regime C day. Now these thresholds vary
          by the evening scan's regime (see REGIME_TOLERANCE below).
          CONFIRMED 2026-07-28: picks_latest.json never actually included
          a "regime" field at all (verified against orchestrator.py's
          picks_json construction) -- not a wrong key name, the data
          simply didn't exist yet. Fixed by adding "regime": self.regime
          to orchestrator.py's per-pick dict (same evening-scan regime
          letter scanner.py already logs as orc.regime). This file's
          picks_meta.get("regime") lookup needed no changes -- it was
          already reading the right key, just waiting on real data.
"""

import os, json, smtplib, time
import requests
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from market_calendar.staleness_check import (check_staleness, StaleDataError,
                                              get_code_provenance, verify_intraday_freshness)
from database.schema import get_connection, init_all_tables
import dhan_rvol
from agents.intraday_vol_adjusted_agent import IntradayVolAdjustedAgent

try:
    from loguru import logger as log
except ImportError:
    import logging
    log = logging.getLogger(__name__)

load_dotenv(override=True)

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
PICKS_JSON_PATH    = os.getenv("PICKS_JSON_PATH", "picks_latest.json")

# ── Version — single source of truth ─────────────────────────────────────────
VERSION = "5.4"

# ── Thresholds ────────────────────────────────────────────────────────────────
RVOL_CONFIRM_MIN        = 1.5   # RVOL >= this → CONFIRMED (full size)
RVOL_LOW_VOL_MIN        = 1.0   # RVOL >= this → CONFIRMED_LOW_VOL (half size)
                                 # RVOL < 1.0 above pivot → BREAKOUT_NO_VOLUME
MAX_ENTRY_DRIFT_PCT     = 2.0   # fallback default if regime lookup unavailable
MAX_PIVOT_EXTENSION_PCT = 3.0   # fallback default if regime lookup unavailable
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN_TIME        = datetime.strptime("09:15", "%H:%M").time()

# [FIX-3] Regime-aware chase tolerance. Wider on strong-trend days (momentum
# continuation is a different risk profile than chasing in a choppy market),
# tighter on weak/negative regimes. Letter codes match orchestrator.py's
# regime output (A=strongest ... E=weakest) -- see docstring ASSUMPTION note.
REGIME_TOLERANCE = {
    "A": {"max_entry_drift_pct": 4.0, "max_pivot_extension_pct": 6.0},   # STRONG BULL
    "B": {"max_entry_drift_pct": 3.0, "max_pivot_extension_pct": 4.5},
    "C": {"max_entry_drift_pct": 2.0, "max_pivot_extension_pct": 3.0},   # = old fixed defaults
    "D": {"max_entry_drift_pct": 1.5, "max_pivot_extension_pct": 2.5},
    "E": {"max_entry_drift_pct": 1.0, "max_pivot_extension_pct": 2.0},
}

# [FIX-2] Statuses that don't need re-checking once reached -- a pick that's
# already CONFIRMED, MISSED, etc. this morning stays that way; only PENDING
# (still below pivot) and DATA_ERROR (retry-worthy) get re-classified on the
# next checkpoint.
#
# [FIX-5 2026-08-18] BREAKOUT_NO_VOLUME removed from this set. It was treated
# as terminal, same as a real entry/exit decision -- but "no volume YET" is
# exactly as reversible as CONFIRMED_PENDING_RVOL's "no RVOL data yet" (which
# was correctly kept non-terminal, see classify()'s [FIX-4]). Confirmed live
# 2026-08-18: GLAXO got locked into BREAKOUT_NO_VOLUME at 0.53x RVOL mid-
# morning and was never re-checked again -- real RVOL reached 6.24x by 15:24
# IST (institutional volume showed up late) but the cached morning verdict
# kept overriding it all day, and the BTST scan (which only ever looks at
# tickers CONFIRMED/CONFIRMED_LOW_VOL at some earlier checkpoint) never got
# a chance to see it. Leaving it non-terminal means it gets re-classified
# each checkpoint like PENDING/CONFIRMED_PENDING_RVOL already do, so genuine
# late-session volume can still flip it to CONFIRMED while price is still
# within chase tolerance -- and if price has since run away, classify()'s
# existing entry-drift/pivot-extension checks correctly catch that as MISSED
# instead, so there's no risk of belatedly recommending a chased entry.
TERMINAL_STATUSES = {"CONFIRMED", "CONFIRMED_LOW_VOL", "MISSED", "BROKEN"}

# [2026-08-18, 5b/4c] NO_RESEND_STATUSES drives the "don't spam duplicate
# emails" dedup only -- a separate concern from data correctness. BROKEN
# is deliberately EXCLUDED here (unlike TERMINAL_STATUSES above, kept for
# any external reference) because it was found to have the same caching
# exposure as the BREAKOUT_NO_VOLUME bug fixed earlier today: once cached,
# a ticker was NEVER re-fetched for the rest of the day, so a single
# bad/stale price tick dipping to/below SL could permanently freeze a
# stock as "broken" all day with zero re-verification path (worse than
# BREAKOUT_NO_VOLUME's old bug, since there wasn't even a partial re-check
# like the BTST scan gives CONFIRMED picks). See the cached-BROKEN
# re-verification branch in main()'s checkpoint loop below.
NO_RESEND_STATUSES = {"CONFIRMED", "CONFIRMED_LOW_VOL", "MISSED"}

# [NEW 2026-08-11] 15:15 BTST (Buy Today Sell Tomorrow) scan thresholds.
# See classify_btst() for how these two gates (closing strength, R:R
# favorability) are applied -- design discussed after the 10:05 GLAND example.
BTST_MIN_RVOL             = 1.5    # full-day-elapsed RVOL must still clear the same bar as a fresh morning entry
BTST_MAX_PCT_OFF_HIGH     = 2.0    # CMP must be within this % of today's high -- further off = momentum stalling into the close
BTST_MAX_PCT_T1_CAPTURED  = 70.0   # skip if >=70% of the entry->T1 move is already used up -- not enough R:R left to justify overnight risk
# ─────────────────────────────────────────────────────────────────────────────


def _load_recipients() -> list:
    path = "recipients.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return [GMAIL_ADDRESS]


def _load_cc_recipients() -> list:
    """Optional CC list -- recipients_cc.txt, one address per line. Applies to
    every email this script sends (all checkpoints + the BTST scan)."""
    path = "recipients_cc.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# RVOL — real first-45min calculation
# ─────────────────────────────────────────────────────────────────────────────

def _rvol_tvdatafeed(ticker_nse: str, as_of: datetime = None) -> float:
    """
    Primary RVOL source: tvDatafeed (TradingView 5-min bars).

    [FIX-1] Window is now ELAPSED-TIME-MATCHED: volume from market open
    (09:15) to `as_of` (defaults to now), compared against the SAME
    elapsed-time window on historical days -- not a fixed 09:15-10:15
    calendar slice. This makes RVOL meaningful at any checkpoint time,
    not just after a fixed 45/60-min mark has fully elapsed.

    Returns RVOL float >= 0, or -1.0 on failure / insufficient elapsed time.
    ALL failures are logged explicitly — no silent fallthrough.
    """
    as_of = as_of or datetime.now(IST)
    try:
        from tvDatafeed import TvDatafeed, Interval
        import pandas as pd

        symbol = ticker_nse.replace(".NS", "")
        log.info(f"    [tvDatafeed] Requesting {symbol} 5-min bars...")

        tv = TvDatafeed()
        df = tv.get_hist(
            symbol=symbol, exchange="NSE",
            interval=Interval.in_15_minute, n_bars=500
        )

        if df is None:
            log.warning(f"    [tvDatafeed] {symbol}: get_hist returned None")
            return -1.0

        log.info(f"    [tvDatafeed] {symbol}: got {df.shape[0]} bars, "
                 f"index range {df.index[0]} → {df.index[-1]}")

        # --- FIX: tvDatafeed returns naive timestamps in UTC, not IST.
        # Must localize to UTC first, then convert to IST — localizing
        # directly to IST just relabels the UTC clock time without
        # shifting it, causing the elapsed-time window search to miss
        # every bar (they land ~5.5hrs off from where they should be).
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            try:
                df.index = df.index.tz_convert("Asia/Kolkata")
            except Exception as e:
                # [2026-08-18, 4e] Was a bare silent pass -- see the sibling
                # fix in get_opening_range/get_intraday_high_low/
                # get_intraday_bars for why this needs at least debug
                # logging rather than a silent continue.
                log.debug(f"    tz conversion failed both ways for {ticker_nse}: {e}")
        # --- END FIX

        today = as_of.date()
        elapsed_minutes = (as_of - datetime.combine(today, MARKET_OPEN_TIME, tzinfo=IST)).total_seconds() / 60
        if elapsed_minutes < 5:
            log.warning(f"    [tvDatafeed] {symbol}: only {elapsed_minutes:.0f}min elapsed "
                        f"since open -- too early for a meaningful RVOL")
            return -1.0

        today_window = df[
            (df.index.date == today) &
            (df.index.time >= MARKET_OPEN_TIME) &
            (df.index.time <= as_of.time())
        ]

        if today_window.empty:
            log.warning(f"    [tvDatafeed] {symbol}: no bars for today "
                        f"({today}) in 09:15-{as_of.strftime('%H:%M')} window. "
                        f"Latest bar date: {df.index[-1].date()}")
            return -1.0

        today_vol = float(today_window["volume"].sum())
        log.info(f"    [tvDatafeed] {symbol}: today {elapsed_minutes:.0f}min vol = {today_vol:,.0f}")

        hist = df[df.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_window = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= MARKET_OPEN_TIME) &
                (hist.index.time <= as_of.time())
            ]
            if not day_window.empty:
                hist_by_day[row_date] = float(day_window["volume"].sum())

        if len(hist_by_day) < 3:
            log.warning(f"    [tvDatafeed] {symbol}: only {len(hist_by_day)} "
                        f"historical days — insufficient for avg")
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        rvol = round(today_vol / avg_hist, 2) if avg_hist > 0 else -1.0
        log.info(f"    [tvDatafeed] {symbol}: RVOL = {rvol:.2f}x "
                 f"(today {today_vol:,.0f} / avg {avg_hist:,.0f}, {elapsed_minutes:.0f}min elapsed)")
        return rvol

    except ImportError:
        log.warning("    [tvDatafeed] Not installed — pip install tvDatafeed")
        return -1.0
    except Exception as e:
        log.warning(f"    [tvDatafeed] {ticker_nse}: FAILED — {type(e).__name__}: {e}")
        return -1.0


def _rvol_yfinance_fallback(ticker_nse: str, as_of: datetime = None) -> float:
    """
    Fallback RVOL: yfinance 1-min data. Same elapsed-time-matched window
    fix as _rvol_tvdatafeed -- see FIX-1 in module docstring.
    Returns RVOL float >= 0, or -1.0 on failure.
    """
    as_of = as_of or datetime.now(IST)
    try:
        import yfinance as yf
        import pandas as pd

        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        log.info(f"    [yfinance-fallback] Fetching 1-min data for {t}...")
        df_1m = yf.Ticker(t).history(period="5d", interval="1m", prepost=False)

        if df_1m.empty:
            log.warning(f"    [yfinance-fallback] {t}: empty response")
            return -1.0

        try:
            df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
        except Exception:
            pass

        today = as_of.date()
        elapsed_minutes = (as_of - datetime.combine(today, MARKET_OPEN_TIME, tzinfo=IST)).total_seconds() / 60
        if elapsed_minutes < 5:
            log.warning(f"    [yfinance-fallback] {t}: only {elapsed_minutes:.0f}min elapsed "
                        f"since open -- too early for a meaningful RVOL")
            return -1.0

        today_window = df_1m[
            (df_1m.index.date == today) &
            (df_1m.index.time >= MARKET_OPEN_TIME) &
            (df_1m.index.time <= as_of.time())
        ]
        if today_window.empty:
            log.warning(f"    [yfinance-fallback] {t}: no bars for today in window")
            return -1.0

        today_vol = float(today_window["Volume"].sum())
        hist = df_1m[df_1m.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_window = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= MARKET_OPEN_TIME) &
                (hist.index.time <= as_of.time())
            ]
            if not day_window.empty:
                hist_by_day[row_date] = float(day_window["Volume"].sum())

        if len(hist_by_day) < 3:
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        rvol = round(today_vol / avg_hist, 2) if avg_hist > 0 else -1.0
        log.info(f"    [yfinance-fallback] {t}: RVOL = {rvol:.2f}x ({elapsed_minutes:.0f}min elapsed)")
        return rvol

    except Exception as e:
        log.warning(f"    [yfinance-fallback] {ticker_nse}: FAILED — {e}")
        return -1.0


def _rvol_dhan(ticker_raw: str, as_of: datetime = None) -> tuple:
    """
    [NEW] Primary RVOL source — Dhan's paid Data API, elapsed-time-matched
    (see dhan_rvol.py). Returns (rvol_float, elapsed_minutes_or_None); rvol
    is -1.0 on any failure (too early in session, security_id not mapped,
    network/API error, no historical data, candle-integrity violation) so
    it slots into the same fallback chain as the tvDatafeed/yfinance
    functions below.
    """
    symbol = ticker_raw.replace(".NS", "")
    try:
        result = dhan_rvol.compute_rvol(symbol, as_of=as_of)
    except Exception as e:
        log.warning(f"    [Dhan] {symbol}: RVOL call raised {type(e).__name__}: {e}")
        return -1.0, None

    if "error" in result:
        log.info(f"    [Dhan] {symbol}: {result['error']}")
        return -1.0, None

    log.info(f"    [Dhan] {symbol}: RVOL = {result['rvol']:.2f}x "
             f"({result.get('elapsed_minutes', '?')}min elapsed)")
    return result["rvol"], result.get("elapsed_minutes")


def get_rvol(ticker: str, as_of: datetime = None) -> tuple:
    """
    Returns (rvol_float, source_label, elapsed_minutes_or_None).
    Tries Dhan first (paid, elapsed-time-matched, most reliable), falls
    back to tvDatafeed, then yfinance, then N/A. elapsed_minutes is only
    populated for the Dhan path today (tvDatafeed/yfinance fallbacks don't
    thread it through) -- still a real improvement over no provenance at
    all for the common case, since Dhan is tried first.
    """
    ticker_raw = ticker if ticker.endswith(".NS") else ticker + ".NS"

    rvol, elapsed = _rvol_dhan(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "Dhan", elapsed

    log.info(f"    Dhan failed for {ticker_raw} — trying tvDatafeed fallback")
    rvol = _rvol_tvdatafeed(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "tvDatafeed", None

    log.info(f"    tvDatafeed failed for {ticker_raw} — trying yfinance fallback")
    rvol = _rvol_yfinance_fallback(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "yfinance", None

    log.warning(f"    All RVOL sources failed for {ticker_raw} — returning N/A")
    return -1.0, "N/A", None


# ─────────────────────────────────────────────────────────────────────────────
# Live price
# ─────────────────────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> tuple:
    """
    [2026-08-18, 5b-iii] Dhan-primary, yfinance-fallback-only. yfinance's
    fast_info exposes no trustworthy trade timestamp at all (a documented,
    unfixable gap in yfinance itself -- confirmed no reliable metadata
    field across versions) -- so it can never be trusted as the
    authoritative "is this price current" answer on its own. Dhan's
    intraday candles carry a real per-bar timestamp, so Dhan is tried
    first via dhan_rvol.get_ltp() (nearly free: the same underlying call
    is already made for RVOL on this ticker this checkpoint). Falls back
    to yfinance ONLY if Dhan fails, and that fallback is explicitly tagged
    lower-confidence rather than silently presented as equally trustworthy.

    Returns (price_or_None, fetched_at_or_None, source_str, suspect_bool).
    fetched_at is None for the yfinance path by construction (it cannot
    prove one) -- callers apply verify_intraday_freshness() to this value,
    which correctly marks a None timestamp as unverified rather than
    silently trusting it. suspect_bool is a cheap probabilistic tell on
    the yfinance path only: LTP exactly equal to previous close during
    market hours is a real signature of a stuck/delayed feed.
    """
    symbol = ticker.replace(".NS", "")
    try:
        result = dhan_rvol.get_ltp(symbol)
        if "error" not in result:
            return result["price"], result["fetched_at"], "dhan", False
        log.info(f"    [Dhan] LTP unavailable for {symbol}: {result['error']} "
                 f"— falling back to yfinance")
    except Exception as e:
        log.warning(f"    [Dhan] LTP call raised {type(e).__name__}: {e} — falling back to yfinance")

    try:
        import yfinance as yf
        t    = ticker if ticker.endswith(".NS") else ticker + ".NS"
        fast = yf.Ticker(t).fast_info
        ltp  = float(fast.last_price)
        if ltp <= 0:
            return None, None, "yfinance_fallback", False
        suspect = False
        try:
            suspect = abs(ltp - float(fast.previous_close)) < 1e-9
        except Exception:
            pass
        return ltp, None, "yfinance_fallback", suspect
    except Exception as e:
        log.warning(f"  [WARN] live price failed for {ticker}: {e}")
        return None, None, "yfinance_fallback", False


# ─────────────────────────────────────────────────────────────────────────────
# Classification — v5.3 status set
# ─────────────────────────────────────────────────────────────────────────────

OPENING_RANGE_MINUTES = 15


def get_opening_range(ticker_nse: str, as_of: datetime = None,
                       minutes: int = OPENING_RANGE_MINUTES) -> tuple:
    """
    Returns (opening_range_high, opening_range_low, source) for today's
    first `minutes` of trading, or (None, None, "unavailable") on failure.

    WHY: classify()'s drift/extension checks compare CMP against the
    STATIC pivot/entry set the evening before. On a genuine gap-up day,
    price may have already cleared that pivot within the first few
    minutes and been consolidating near its OWN opening range since --
    measuring "how extended" against last night's number on a day like
    that reproduces the exact false-MISSED pattern that started this
    whole investigation (13/13 picks missed on 29 Jul, all with real
    RVOL). Anchoring to the day's actual opening range fixes that,
    without touching entry/SL/T1 themselves.

    Deliberately a SEPARATE fetch from get_rvol() -- not refactored to
    share one call with the already-fixed, already-tested elapsed-time
    RVOL logic, to keep this isolated and low-risk. Costs one extra API
    call per ticker.
    """
    as_of = as_of or datetime.now(IST)
    today = as_of.date()
    window_end = (datetime.combine(today, MARKET_OPEN_TIME, tzinfo=IST)
                  + timedelta(minutes=minutes)).time()

    try:
        from tvDatafeed import TvDatafeed, Interval
        symbol = ticker_nse.replace(".NS", "")
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange="NSE",
                          interval=Interval.in_5_minute, n_bars=100)
        if df is not None and not df.empty:
            try:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            except Exception:
                try:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                except Exception as e:
                    # [2026-08-18, 4e] Was a bare silent pass -- if BOTH tz
                    # conversions fail, execution used to continue with an
                    # unconverted index with no trace at all. In practice
                    # this self-corrects into an empty-window -1.0 (the
                    # date/time slice below won't match anything), but that
                    # was accidental, not by design -- log it so a genuinely
                    # corrupted-index scenario is at least visible.
                    log.debug(f"    tz conversion failed both ways for this frame: {e}")
            win = df[(df.index.date == today) &
                     (df.index.time >= MARKET_OPEN_TIME) &
                     (df.index.time <= window_end)]
            if not win.empty:
                orh, orl = float(win["high"].max()), float(win["low"].min())
                log.info(f"    [tvDatafeed] {symbol}: opening range ({minutes}min) "
                         f"= {orl:.1f}-{orh:.1f}")
                return orh, orl, "tvDatafeed"
    except Exception as e:
        log.debug(f"    [tvDatafeed] opening range {ticker_nse}: {type(e).__name__}: {e}")

    try:
        import yfinance as yf
        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        df_1m = yf.Ticker(t).history(period="5d", interval="1m", prepost=False)
        if not df_1m.empty:
            try:
                df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
            win = df_1m[(df_1m.index.date == today) &
                        (df_1m.index.time >= MARKET_OPEN_TIME) &
                        (df_1m.index.time <= window_end)]
            if not win.empty:
                orh, orl = float(win["High"].max()), float(win["Low"].min())
                log.info(f"    [yfinance-fallback] {t}: opening range ({minutes}min) "
                         f"= {orl:.1f}-{orh:.1f}")
                return orh, orl, "yfinance"
    except Exception as e:
        log.debug(f"    [yfinance-fallback] opening range {ticker_nse}: {type(e).__name__}: {e}")

    log.warning(f"    Opening range unavailable for {ticker_nse} — "
                f"classify() falls back to static-pivot-only behavior")
    return None, None, "unavailable"


def get_intraday_high_low(ticker_nse: str, as_of: datetime = None) -> tuple:
    """
    Returns (day_high, day_low, source) for today's session from market open
    (09:15) up to `as_of` -- used by the 15:15 BTST scan's closing-strength
    gate (how far CMP has faded from today's high). Same tvDatafeed ->
    yfinance fallback chain as get_opening_range(), but the window is
    open-to-as_of rather than a fixed first-N-minutes slice.
    """
    as_of = as_of or datetime.now(IST)
    today = as_of.date()

    try:
        from tvDatafeed import TvDatafeed, Interval
        symbol = ticker_nse.replace(".NS", "")
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange="NSE",
                          interval=Interval.in_5_minute, n_bars=100)
        if df is not None and not df.empty:
            try:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            except Exception:
                try:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                except Exception as e:
                    # [2026-08-18, 4e] Was a bare silent pass -- if BOTH tz
                    # conversions fail, execution used to continue with an
                    # unconverted index with no trace at all. In practice
                    # this self-corrects into an empty-window -1.0 (the
                    # date/time slice below won't match anything), but that
                    # was accidental, not by design -- log it so a genuinely
                    # corrupted-index scenario is at least visible.
                    log.debug(f"    tz conversion failed both ways for this frame: {e}")
            win = df[(df.index.date == today) &
                     (df.index.time >= MARKET_OPEN_TIME) &
                     (df.index.time <= as_of.time())]
            if not win.empty:
                dh, dl = float(win["high"].max()), float(win["low"].min())
                log.info(f"    [tvDatafeed] {symbol}: day range so far = {dl:.1f}-{dh:.1f}")
                return dh, dl, "tvDatafeed"
    except Exception as e:
        log.debug(f"    [tvDatafeed] day range {ticker_nse}: {type(e).__name__}: {e}")

    try:
        import yfinance as yf
        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        df_1m = yf.Ticker(t).history(period="5d", interval="1m", prepost=False)
        if not df_1m.empty:
            try:
                df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
            win = df_1m[(df_1m.index.date == today) &
                        (df_1m.index.time >= MARKET_OPEN_TIME) &
                        (df_1m.index.time <= as_of.time())]
            if not win.empty:
                dh, dl = float(win["High"].max()), float(win["Low"].min())
                log.info(f"    [yfinance-fallback] {t}: day range so far = {dl:.1f}-{dh:.1f}")
                return dh, dl, "yfinance"
    except Exception as e:
        log.debug(f"    [yfinance-fallback] day range {ticker_nse}: {type(e).__name__}: {e}")

    log.warning(f"    Day high/low unavailable for {ticker_nse} — BTST closing-strength check skipped")
    return None, None, "unavailable"


def get_intraday_bars(ticker_nse: str, as_of: datetime = None):
    """
    [2026-08-13] Same tvDatafeed -> yfinance fallback chain and open-to-
    as_of window as get_intraday_high_low() above, but returns the actual
    bar series (a DataFrame with a 'Close' column, open-to-as_of) instead
    of collapsing it to day_high/day_low. Feeds
    agents.intraday_vol_adjusted_agent.IntradayVolAdjustedAgent — see
    compute_intraday_diagnostic() below. Returns None on any failure
    (paper-mode diagnostic, must never block the real BTST classification).
    """
    as_of = as_of or datetime.now(IST)
    today = as_of.date()

    try:
        from tvDatafeed import TvDatafeed, Interval
        symbol = ticker_nse.replace(".NS", "")
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange="NSE",
                          interval=Interval.in_5_minute, n_bars=100)
        if df is not None and not df.empty:
            try:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            except Exception:
                try:
                    df.index = df.index.tz_convert("Asia/Kolkata")
                except Exception as e:
                    # [2026-08-18, 4e] Was a bare silent pass -- if BOTH tz
                    # conversions fail, execution used to continue with an
                    # unconverted index with no trace at all. In practice
                    # this self-corrects into an empty-window -1.0 (the
                    # date/time slice below won't match anything), but that
                    # was accidental, not by design -- log it so a genuinely
                    # corrupted-index scenario is at least visible.
                    log.debug(f"    tz conversion failed both ways for this frame: {e}")
            win = df[(df.index.date == today) &
                     (df.index.time >= MARKET_OPEN_TIME) &
                     (df.index.time <= as_of.time())]
            if not win.empty:
                return win.rename(columns={"close": "Close", "open": "Open"})
    except Exception as e:
        log.debug(f"    [tvDatafeed] intraday bars {ticker_nse}: {type(e).__name__}: {e}")

    try:
        import yfinance as yf
        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        df_1m = yf.Ticker(t).history(period="5d", interval="1m", prepost=False)
        if not df_1m.empty:
            try:
                df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
            win = df_1m[(df_1m.index.date == today) &
                        (df_1m.index.time >= MARKET_OPEN_TIME) &
                        (df_1m.index.time <= as_of.time())]
            if not win.empty:
                return win
    except Exception as e:
        log.debug(f"    [yfinance-fallback] intraday bars {ticker_nse}: {type(e).__name__}: {e}")

    return None


def get_trailing_daily_vol_pct(ticker_nse: str) -> float:
    """
    [2026-08-13] Stddev of daily returns over the trailing ~20 trading
    days (excluding today), for IntradayVolAdjustedAgent's Z calculation
    — same non-annualized convention as the backtest that validated this
    approach (see validation/intraday_vol_adjusted_validate.py). Returns
    0.0 on any failure (agent treats that as "insufficient data," not a
    crash — same fail-open convention as every prototype agent).
    """
    try:
        import yfinance as yf
        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        hist = yf.Ticker(t).history(period="2mo", interval="1d")
        if hist.empty or len(hist) < 15:
            return 0.0
        today_str = datetime.now(IST).date().isoformat()
        hist = hist[hist.index.strftime("%Y-%m-%d") != today_str]
        closes = hist["Close"].to_numpy(dtype=float)[-21:]
        if len(closes) < 15:
            return 0.0
        rets = closes[1:] / closes[:-1] - 1
        return float(np.std(rets, ddof=1))
    except Exception as e:
        log.debug(f"    trailing daily vol {ticker_nse}: {type(e).__name__}: {e}")
        return 0.0


def compute_intraday_diagnostic(ticker_nse: str, as_of: datetime = None) -> dict:
    """
    [2026-08-13, PAPER-MODE — informational only, does not gate anything]

    Factor-library "intraday vol-adjusted move" idea, scoped out and
    prototyped after a choppy-volume session raised the question of
    whether RVOL alone can tell a clean continuation apart from a loud
    whipsaw. Validated against validation/intraday_vol_adjusted_validate.py
    (N=153 historical BTST-confirmed signals, 2015-2026): results were
    UNDERPOWERED, not proven — no checkpoint reached significance, and the
    direction flipped between checkpoints on the smallest buckets. Wired
    in here purely to ACCUMULATE real observations going forward, the same
    "observe, don't pay" pattern this codebase already uses for the P2-08
    ADD-ON premise (see orchestrator.py's ADDON_LIVE_EXECUTION). Must
    never gate BTST classification or change any status — see
    classify_btst(), which does not call this function.

    Returns a dict (z, efficiency, label) or None if live data was
    unavailable — always fails open, never raises.
    """
    try:
        bars = get_intraday_bars(ticker_nse, as_of=as_of)
        if bars is None or bars.empty or "Close" not in bars.columns:
            return None
        day_open = float(bars["Open"].iloc[0]) if "Open" in bars.columns else float(bars["Close"].iloc[0])
        trailing_vol = get_trailing_daily_vol_pct(ticker_nse)

        agent = IntradayVolAdjustedAgent(
            bars[["Close"]], day_open=day_open, trailing_daily_vol_pct=trailing_vol
        )
        if agent.get_bars_available() < 2:
            return None

        z, eff = agent.get_z(), agent.get_efficiency()
        if abs(z) >= 1.0 and eff >= 0.6:
            label = "Clean/Strong"
        elif abs(z) >= 1.0 and eff < 0.6:
            label = "Choppy"
        elif abs(z) < 1.0 and eff >= 0.6:
            label = "Quiet/Orderly"
        else:
            label = "Quiet"

        return {"z": z, "efficiency": eff, "label": label}
    except Exception as e:
        log.debug(f"    intraday diagnostic {ticker_nse}: {type(e).__name__}: {e}")
        return None


def classify(pick: dict, cmp: float | None, rvol: float, rvol_src: str,
             max_entry_drift_pct: float = MAX_ENTRY_DRIFT_PCT,
             max_pivot_extension_pct: float = MAX_PIVOT_EXTENSION_PCT,
             opening_range: tuple = (None, None, None),
             price_verified: bool = True, price_source: str = "unknown") -> dict:
    """
    Status set (v5.3):
      CONFIRMED           — above pivot, RVOL >= 1.5x
      CONFIRMED_LOW_VOL   — above pivot, RVOL 1.0-1.5x
      BREAKOUT_NO_VOLUME  — above pivot, RVOL < 1.0x  ← NEW
      MISSED              — entry drift > threshold, OR pivot extension > threshold with low RVOL
      PENDING             — below pivot
      BROKEN              — at or below stop loss
      DATA_ERROR          — price feed failed entirely (no number at all)
      PRICE_UNVERIFIED    — a price WAS returned, but its freshness could ← NEW 2026-08-18
                             not be proven (e.g. yfinance fallback with no
                             trade timestamp) -- "unknown != current", see
                             get_live_price()/5b-iii. Non-terminal: this
                             ticker specifically gets re-checked next
                             checkpoint, without blocking the rest of the
                             email over one flaky source.

    [FIX-3] max_entry_drift_pct / max_pivot_extension_pct are now passed
    in per-call (regime-aware, see REGIME_TOLERANCE) rather than always
    reading the module-level defaults directly.
    """
    entry = pick["entry"]
    sl    = pick["sl"]
    pivot = pick.get("pivot", entry)
    t1    = pick["t1"]

    # [NEW] Opening-range anchoring — see get_opening_range() docstring.
    # If today's actual opening range already cleared the static pivot,
    # judge "how extended" against the OPENING RANGE, not last night's
    # number. entry/sl/t1 are NEVER changed by this — only the chase
    # tolerance's reference point is.
    orh, orl, or_src = opening_range if opening_range else (None, None, None)
    anchor_pivot = pivot
    anchor_entry = entry
    or_anchored  = False
    if orh is not None and orh > pivot:
        anchor_pivot = orh
        anchor_entry = max(entry, orh)
        or_anchored  = True

    if cmp is not None and not price_verified:
        return {
            "status": "PRICE_UNVERIFIED",
            "ltp": cmp, "gap_pct": None, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": None, "is_chasing": False,
            "action": (f"Got a price (₹{cmp:,.1f} via {price_source}) but could not "
                       f"prove it's current — no reliable trade timestamp available. "
                       f"Not trusted for a classification. Will re-check next checkpoint."),
            "label":  "PRICE UNVERIFIED",
            "color":  "#ffffff", "bg": "#78716c",
        }

    if cmp is None:
        return {
            "status": "DATA_ERROR",
            "ltp": None, "gap_pct": None, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": None, "is_chasing": False,
            "action": "Could not fetch live price — check manually",
            "label":  "DATA ERROR",
            "color":  "#6b7280", "bg": "#f3f4f6",
        }

    gap_pct = round(((cmp - pivot) / pivot) * 100, 1) if pivot > 0 else 0.0

    # Entry drift vs planned entry price (opening-range-anchored when applicable)
    drift_pct       = round(((cmp - anchor_entry) / anchor_entry) * 100, 1) if anchor_entry > 0 else 0.0
    entry_chasing   = drift_pct > max_entry_drift_pct

    # [BUG-4/5] Secondary: pivot extension with low volume (opening-range-anchored)
    pivot_ext_pct   = round(((cmp - anchor_pivot) / anchor_pivot) * 100, 1) if anchor_pivot > 0 else 0.0
    pivot_extended  = (pivot_ext_pct > max_pivot_extension_pct
                       and rvol >= 0 and rvol < RVOL_CONFIRM_MIN)

    vol_ok_confirm  = rvol >= RVOL_CONFIRM_MIN
    vol_ok_low      = rvol >= RVOL_LOW_VOL_MIN

    # ── BROKEN ───────────────────────────────────────────────────────────────
    if cmp <= sl:
        return {
            "status": "BROKEN",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": False,
            "action": f"SL breached — DO NOT enter. CMP ₹{cmp:,.1f} ≤ SL ₹{sl:,.1f}",
            "label":  "SKIP — BROKEN",
            "color":  "#ffffff", "bg": "#dc2626",
        }

    # ── MISSED — entry drift OR pivot too extended with no volume ─────────────
    if entry_chasing or pivot_extended:
        or_note = f" (opening-range anchored: {orl:.1f}-{orh:.1f})" if or_anchored else ""
        if pivot_extended and not entry_chasing:
            reason = (f"Breakout extended {pivot_ext_pct:.1f}% above pivot{or_note} "
                      f"with only {rvol:.1f}x volume — trap, not entry")
        else:
            reason = (f"Price drifted {drift_pct:+.1f}% above planned entry "
                      f"₹{entry:,.1f}{or_note} — R:R destroyed")
        return {
            "status": "MISSED",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": True,
            "action": f"{reason} — DO NOT chase",
            "label":  f"MISSED",
            "color":  "#ffffff", "bg": "#7c3aed",
        }

    # ── Above pivot ───────────────────────────────────────────────────────────
    if cmp >= pivot:
        rvol_disp = f"{rvol:.1f}x" if rvol >= 0 else "N/A"

        # [FIX-4] RVOL genuinely unresolved (both sources failed) — do NOT
        # fail open into CONFIRMED. Stay non-terminal so this ticker gets
        # re-checked at the next checkpoint instead of being locked in on
        # a guess.
        if rvol < 0:
            return {
                "status": "CONFIRMED_PENDING_RVOL",
                "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
                "drift_pct": drift_pct, "is_chasing": False,
                "action": (f"Above pivot (CMP ₹{cmp:,.1f}) but RVOL not yet "
                           f"available — price confirmed, volume unconfirmed. "
                           f"Will re-check next checkpoint."),
                "label":  "PRICE OK — VOL PENDING",
                "color":  "#ffffff", "bg": "#2563eb",
            }

        if vol_ok_confirm:
            return {
                "status": "CONFIRMED",
                "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
                "drift_pct": drift_pct, "is_chasing": False,
                "action": (f"Enter — CMP ₹{cmp:,.1f} | Entry ₹{entry:,.1f} | "
                           f"SL ₹{sl:,.1f} | T1 ₹{t1:,.1f}"),
                "label":  "CONFIRMED — ENTER",
                "color":  "#ffffff", "bg": "#16a34a",
            }

        elif vol_ok_low:
            return {
                "status": "CONFIRMED_LOW_VOL",
                "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
                "drift_pct": drift_pct, "is_chasing": False,
                "action": (f"Above pivot but below-average volume ({rvol_disp}) — "
                           f"enter half size only"),
                "label":  f"LOW VOLUME ({rvol_disp})",
                "color":  "#ffffff", "bg": "#ca8a04",
            }

        else:
            # [BUG-2] NEW STATUS: above pivot but RVOL < 1.0x
            return {
                "status": "BREAKOUT_NO_VOLUME",
                "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
                "drift_pct": drift_pct, "is_chasing": False,
                "action": (f"Above pivot but RVOL only {rvol_disp} — "
                           f"breakout without volume is a trap. Do NOT enter."),
                "label":  f"NO VOLUME ({rvol_disp})",
                "color":  "#ffffff", "bg": "#dc2626",
            }

    # ── PENDING — below pivot ─────────────────────────────────────────────────
    dist     = pivot - cmp
    dist_pct = round((dist / pivot) * 100, 1) if pivot > 0 else 0.0
    return {
        "status": "PENDING",
        "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
        "drift_pct": drift_pct, "is_chasing": False,
        "action": (f"Below pivot by ₹{dist:,.1f} ({dist_pct:.1f}%) — "
                   f"set alert at ₹{pivot:,.1f}"),
        "label":  "PENDING",
        "color":  "#1e40af", "bg": "#dbeafe",
    }


# ─────────────────────────────────────────────────────────────────────────────
# [NEW 2026-08-11] BTST safety gates — NSE ASM/GSM + per-ticker earnings
# ─────────────────────────────────────────────────────────────────────────────

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/reports/asm-gsm",
}


def get_asm_gsm_symbols() -> set | None:
    """
    Live NSE surveillance list (ASM long+short term + GSM), fetched once per
    BTST scan -- not per ticker. Confirmed working 2026-08-11 against
    https://www.nseindia.com/api/reportASM and .../reportGSM (both return
    the current day's list, keyed by symbol, no auth/cookie needed).

    Returns None on any fetch/parse failure (network, NSE format change,
    blocked request) -- callers must treat None as "could not verify," NOT
    as "clear." BTST is fail-SAFE here (block if unverifiable), matching
    the existing RVOL<0 pattern in classify() -- staying non-terminal /
    unconfirmed on missing data, never silently assuming safety.
    """
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        session.get("https://www.nseindia.com/", timeout=8)  # cookie warm-up

        flagged = set()

        r_asm = session.get("https://www.nseindia.com/api/reportASM", timeout=10)
        r_asm.raise_for_status()
        asm = r_asm.json()
        for bucket in ("longterm", "shortterm"):
            for row in asm.get(bucket, {}).get("data", []):
                sym = row.get("symbol")
                if sym:
                    flagged.add(sym.upper())

        r_gsm = session.get("https://www.nseindia.com/api/reportGSM", timeout=10)
        r_gsm.raise_for_status()
        for row in r_gsm.json():
            sym = row.get("symbol")
            if sym:
                flagged.add(sym.upper())

        log.info(f"  ASM/GSM list fetched: {len(flagged)} symbol(s) currently flagged")
        return flagged
    except Exception as e:
        log.warning(f"  ASM/GSM fetch failed ({type(e).__name__}: {e}) — "
                    f"BTST will treat all tickers as UNVERIFIED, not clear")
        return None


def days_to_next_earnings(ticker_raw: str, as_of: date = None) -> int | None:
    """
    Days until the nearest upcoming (or today's) NSE-listed board meeting /
    financial-results event for this ticker, via the same
    https://www.nseindia.com/api/event-calendar endpoint already used (with
    identical parsing) by agents/fundamental_proxy_agent.py's Agent 9B --
    confirmed live 2026-08-11 (e.g. GLAND.NS shows a real 10-Aug-2026
    "Financial Results" event, one day before this check was built).

    Three distinct outcomes, NOT collapsed into one:
      - int >= 0: a qualifying future event is confirmed that many days out.
      - -1: fetch succeeded, no qualifying event is currently announced --
        this is the NORMAL, SAFE case most days (NSE only publishes board-
        meeting dates roughly 5-7 days ahead per SEBI norms, so "nothing
        announced" does not mean "nothing coming," it means the near-term
        window is genuinely clear as far as anyone can know right now).
      - None: the fetch itself failed (network/parse/blocked) -- genuinely
        unverifiable, and the only case callers should treat as unsafe.
      Collapsing -1 and None into one "unknown" value would make this gate
      block almost every day for no real reason, since most tickers have
      no announced event on most days.
    """
    as_of = as_of or datetime.now(IST).date()
    symbol = ticker_raw.replace(".NS", "")
    try:
        r = requests.get(
            f"https://www.nseindia.com/api/event-calendar?symbol={symbol}",
            headers=_NSE_HEADERS, timeout=8,
        )
        r.raise_for_status()
        events = r.json()
        if not isinstance(events, list):
            return None

        best_days = None
        for event in events:
            date_str = event.get("date", "") or event.get("bm_date", "")
            purpose  = (event.get("purpose", "") or "").lower()
            desc     = (event.get("bm_desc", "") or "").lower()
            if not date_str:
                continue
            if not any(kw in purpose or kw in desc
                       for kw in ("result", "quarterly", "financial results")):
                continue
            try:
                ev_date = datetime.strptime(date_str, "%d-%b-%Y").date()
            except Exception:
                continue
            days_away = (ev_date - as_of).days
            if days_away < 0:
                continue
            if best_days is None or days_away < best_days:
                best_days = days_away
        return best_days if best_days is not None else -1
    except Exception as e:
        log.debug(f"    event-calendar fetch failed for {symbol}: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# [NEW 2026-08-11] BTST (Buy Today Sell Tomorrow) classification — 15:15 scan
# ─────────────────────────────────────────────────────────────────────────────

def classify_btst(pick: dict, cmp: float | None, day_high: float | None,
                   rvol: float, rvol_src: str,
                   asm_gsm_symbols: set | None = None,
                   days_to_earnings: int | None = None) -> dict:
    """
    15:15 BTST scan classification -- separate from the morning entry
    classify() above. Only ever called on tickers that were already
    CONFIRMED / CONFIRMED_LOW_VOL at an earlier checkpoint today (see
    run_btst_scan) -- this answers "hold overnight?", not "enter now?".

    Two SAFETY GATES checked FIRST, ahead of everything else -- these are
    hard vetoes on holding overnight at all, not signal-quality checks:
      0a. ASM/GSM -- NSE surveillance-listed stocks carry real trading
          restrictions (margin, price bands, sometimes trade-for-trade
          only). asm_gsm_symbols=None means the fetch failed and this
          could not be verified -- BLOCKS just like a real flag, since
          "unverifiable" must never be treated as "safe" for a regulatory
          check (see get_asm_gsm_symbols() docstring).
      0b. Earnings-eve -- days_to_earnings 0 or 1 means results/board
          meeting today or tomorrow: hold-overnight risk is asymmetric
          gap risk, not the normal "was the pattern right" risk, so this
          is a hard reject, not the soft score penalty the morning scan's
          fundamental_proxy_agent.py uses. days_to_earnings=None (fetch
          failed) also blocks, same reasoning as ASM/GSM. -1 means
          "checked, nothing announced" -- the normal, safe case (see
          days_to_next_earnings() docstring) -- NOT treated as a block.

    Then the two original gates (Closing Strength + R:R Favorability, per
    the GLAND example that motivated this):
      1. Closing strength -- CMP within BTST_MAX_PCT_OFF_HIGH of today's
         high AND full-day-elapsed RVOL still >= BTST_MIN_RVOL. A stock
         that faded off its highs or saw volume dry up into the close is
         NOT a BTST candidate, regardless of how the morning entry went.
      2. R:R favorability -- less than BTST_MAX_PCT_T1_CAPTURED% of the
         entry->T1 move already captured. A stock 90% of the way to T1
         has little room left to reward the overnight gap risk.
    """
    entry = pick["entry"]
    sl    = pick["sl"]
    pivot = pick.get("pivot", entry)
    t1    = pick["t1"]
    ticker_disp = pick.get("ticker", "").replace(".NS", "")

    if cmp is None:
        return {
            "status": "DATA_ERROR", "ltp": None, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": None, "pct_captured": None,
            "action": "Could not fetch live price — check manually before market close",
            "label": "DATA ERROR", "color": "#6b7280", "bg": "#f3f4f6",
        }

    if asm_gsm_symbols is None or ticker_disp.upper() in asm_gsm_symbols:
        verified = asm_gsm_symbols is not None
        return {
            "status": "ASM_GSM_BLOCKED", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": None, "pct_captured": None,
            "action": (f"{'Under NSE ASM/GSM surveillance' if verified else 'ASM/GSM status could not be verified'} "
                       f"— do not hold overnight, restrictions apply. Square off today."),
            "label": "ASM/GSM" if verified else "ASM/GSM UNVERIFIED",
            "color": "#ffffff", "bg": "#dc2626",
        }

    if days_to_earnings is None or 0 <= days_to_earnings <= 1:
        verified = days_to_earnings is not None
        note = (f"results due in {days_to_earnings}d" if verified
                else "earnings calendar could not be verified")
        return {
            "status": "EARNINGS_BLOCKED", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": None, "pct_captured": None,
            "action": f"Earnings-eve risk ({note}) — asymmetric gap risk, do not hold overnight. Square off today.",
            "label": "EARNINGS EVE" if verified else "EARNINGS UNVERIFIED",
            "color": "#ffffff", "bg": "#dc2626",
        }

    if cmp <= sl:
        return {
            "status": "STOPPED_OUT", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": None, "pct_captured": None,
            "action": (f"SL breached intraday (CMP ₹{cmp:,.1f} ≤ SL ₹{sl:,.1f}) — "
                       f"position should already be closed, nothing to hold"),
            "label": "STOPPED OUT", "color": "#ffffff", "bg": "#dc2626",
        }

    pct_off_high = round(((day_high - cmp) / day_high) * 100, 1) if day_high and day_high > 0 else None
    pct_captured = round(((cmp - entry) / (t1 - entry)) * 100, 1) if t1 != entry else None

    # [2026-08-18, 4e] Was: day_high=None (get_intraday_high_low() fetch
    # failure) silently made `faded` always False -- a candidate could
    # reach BTST_CANDIDATE status having never actually been checked for
    # fading off its high, with the email showing "—" as the only
    # (easy-to-miss) hint the check didn't run. Per the "unverified must
    # BLOCK, not silently pass" contract used elsewhere in this file
    # (get_asm_gsm_symbols/days_to_next_earnings), an unverifiable
    # closing-strength check must explicitly block BTST candidacy, not
    # quietly default to "not faded".
    if day_high is None or day_high <= 0:
        return {
            "status": "BTST_UNVERIFIED", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": None, "pct_captured": pct_captured,
            "action": ("Could not fetch today's intraday high — the closing-strength "
                       "(fade-off-high) safety gate could not be checked. Not cleared "
                       "for BTST until this can be verified."),
            "label": "DAY-HIGH UNVERIFIED", "color": "#ffffff", "bg": "#78716c",
        }

    below_pivot = cmp < pivot
    exhausted   = pct_captured is not None and pct_captured >= BTST_MAX_PCT_T1_CAPTURED
    faded       = pct_off_high is not None and pct_off_high > BTST_MAX_PCT_OFF_HIGH
    vol_light   = rvol < 0 or rvol < BTST_MIN_RVOL

    if below_pivot:
        return {
            "status": "NOT_HELD", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": pct_off_high, "pct_captured": pct_captured,
            "action": (f"CMP ₹{cmp:,.1f} back below pivot ₹{pivot:,.1f} — "
                       f"momentum gone, do not hold overnight"),
            "label": "BELOW PIVOT", "color": "#ffffff", "bg": "#7c3aed",
        }

    if exhausted:
        return {
            "status": "NEAR_T1", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": pct_off_high, "pct_captured": pct_captured,
            "action": (f"Already {pct_captured:.0f}% of the way to T1 ₹{t1:,.1f} — "
                       f"book profit today, little room left for BTST"),
            "label": f"NEAR T1 ({pct_captured:.0f}%)", "color": "#ffffff", "bg": "#ca8a04",
        }

    if faded:
        return {
            "status": "FADED", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": pct_off_high, "pct_captured": pct_captured,
            "action": (f"{pct_off_high:.1f}% off today's high ₹{day_high:,.1f} — "
                       f"momentum stalling into the close, skip BTST"),
            "label": f"FADED ({pct_off_high:.1f}% off high)", "color": "#ffffff", "bg": "#dc2626",
        }

    if vol_light:
        rvol_disp = f"{rvol:.1f}x" if rvol >= 0 else "N/A"
        return {
            "status": "VOL_LIGHT", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
            "pct_off_high": pct_off_high, "pct_captured": pct_captured,
            "action": f"Full-day RVOL only {rvol_disp} — conviction faded through the day, skip BTST",
            "label": f"LIGHT VOL ({rvol_disp})", "color": "#ffffff", "bg": "#dc2626",
        }

    # Both gates cleared
    off_high_disp = f"{pct_off_high:.1f}%" if pct_off_high is not None else "N/A"
    remaining_pct = round(((t1 - cmp) / cmp) * 100, 1) if cmp > 0 else 0.0
    return {
        # [2026-08-11] Label/action deliberately informational, not a "hold
        # overnight" instruction -- validation/btst_backtest.py (1,707
        # historical signals) found this filter does NOT show a validated
        # positive edge (permutation p=0.094 after correcting for testing 2
        # exit metrics -- not significant). See NEXT_STEPS.md. Meeting these
        # criteria is real, checkable information; it is not yet evidence
        # this makes money. Use judgment, don't treat this as a signal.
        "status": "BTST_CANDIDATE", "ltp": cmp, "rvol": rvol, "rvol_src": rvol_src,
        "pct_off_high": pct_off_high, "pct_captured": pct_captured,
        "action": (f"Meets closing-strength criteria (edge not yet validated — see footer) — "
                   f"CMP ₹{cmp:,.1f}, {off_high_disp} off high, RVOL {rvol:.1f}x, "
                   f"{remaining_pct:.1f}% still to T1 ₹{t1:,.1f}. SL stays ₹{sl:,.1f}. "
                   f"Use your own judgment before holding overnight."),
        "label": "BTST CANDIDATE", "color": "#ffffff", "bg": "#16a34a",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Email HTML
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_rvol(rvol: float, src: str, elapsed_minutes: float = None) -> str:
    """
    [2026-08-18] Was: hardcoded "(Live)" on every resolved RVOL regardless
    of actual verification status -- misleading given a fabricated
    rvol=0.0 from an empty Dhan payload (now fixed, but the display bug
    was independent) would have shown "❌ 0.0x (Live)" as if verified.
    Now renders the real elapsed-time provenance dhan_rvol.compute_rvol()
    already calculates, instead of a blanket claim.
    """
    if rvol < 0:
        return "<span style='color:#9ca3af;'>N/A</span>"
    color = "#16a34a" if rvol >= 1.5 else "#ca8a04" if rvol >= 1.0 else "#dc2626"
    icon  = "✅" if rvol >= 1.5 else "⚠️" if rvol >= 1.0 else "❌"
    provenance = f"{elapsed_minutes:.0f}min elapsed" if elapsed_minutes is not None else "elapsed time unknown"
    return (
        f"<span style='color:{color};font-weight:700;'>{icon} {rvol:.1f}x</span>"
        f"<br><span style='font-size:10px;color:#9ca3af;'>{src} · {provenance}</span>"
    )


def _row(r: dict) -> str:
    p           = r["pick"]
    c           = r["classification"]
    ticker_disp = p.get("ticker", "").replace(".NS", "")
    ltp         = f"₹{c['ltp']:,.1f}" if c["ltp"] is not None else "N/A"
    # [2026-08-18] Show real per-datum provenance instead of a bare number
    # with no indication of source or freshness -- "traceable to the
    # underlying source data and its calculation date/time" per the
    # non-negotiable data-accuracy requirement.
    price_fetched_at = c.get("price_fetched_at")
    price_source     = c.get("price_source", "")
    if c["ltp"] is not None:
        ltp_prov = (f"{price_source} · {price_fetched_at} IST" if price_fetched_at
                    else (f"{price_source} · unverified timestamp" if price_source
                          else "unverified timestamp"))
        if c.get("price_suspect"):
            ltp_prov += " · <span style='color:#dc2626'>suspect (=prev close)</span>"
        ltp = f"{ltp}<br><span style='font-size:10px;color:#9ca3af;'>{ltp_prov}</span>"
    gap         = f"{c['gap_pct']:+.1f}%" if c.get("gap_pct") is not None else "N/A"
    drift_cell  = ""
    if c.get("is_chasing"):
        drift_cell = (
            f"<span style='color:#7c3aed;font-weight:700;font-size:10px;'>"
            f"{c['drift_pct']:+.1f}% vs entry</span>"
        )
    elif c.get("drift_pct") is not None:
        drift_cell = (
            f"<span style='color:#9ca3af;font-size:10px;'>"
            f"{c['drift_pct']:+.1f}% vs entry</span>"
        )
    tier = f"T{p.get('tier','?')}"

    # [P1] SL/T1/T2/RR shown on all rows — critical for intraday decisions
    sl  = p.get("sl",  p.get("stop_loss", 0)) or 0
    t1  = p.get("t1",  p.get("target1",   0)) or 0
    t2  = p.get("t2",  p.get("target2",   0)) or 0
    rr  = p.get("rr",  p.get("rrr",       0)) or 0
    entry = p.get("entry", 0) or 0

    sl_cell  = f"<span style='color:#dc2626;font-weight:600;'>₹{sl:,.1f}</span>"  if sl  else "—"
    t1_cell  = f"<span style='color:#16a34a;font-weight:600;'>₹{t1:,.1f}</span>"  if t1  else "—"
    t2_cell  = f"<span style='color:#15803d;font-size:11px;'>₹{t2:,.1f}</span>"   if t2  else "—"
    rr_cell  = f"<span style='color:#1e40af;font-weight:600;'>{rr:.1f}x</span>"   if rr  else "—"

    return f"""
    <tr style="border-bottom:1px solid #e5e7eb;">
      <td style="padding:10px 8px;">
        <span style="font-weight:700;font-size:14px;color:#111827;">{ticker_disp}</span><br>
        <span style="font-size:11px;color:#9ca3af;">{tier} · Score {p.get('score','')}</span>
      </td>
      <td style="padding:10px 8px;font-size:12px;color:#6b7280;">{p.get('sector','')}</td>
      <td style="padding:10px 8px;text-align:center;">
        <span style="background:{c['bg']};color:{c['color']};padding:4px 10px;
            border-radius:4px;font-size:11px;font-weight:700;white-space:nowrap;">
            {c['label']}</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-weight:700;color:#111827;">{ltp}</td>
      <td style="padding:10px 8px;text-align:right;">{_fmt_rvol(c['rvol'], c.get('rvol_src',''), c.get('rvol_elapsed_minutes'))}</td>
      <td style="padding:10px 8px;text-align:right;color:#6b7280;">{gap}</td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        {sl_cell}<br><span style='color:#9ca3af;font-size:10px;'>SL</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        {t1_cell}<br><span style='color:#9ca3af;font-size:10px;'>T1</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        {t2_cell}<br><span style='color:#9ca3af;font-size:10px;'>T2</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        {rr_cell}<br><span style='color:#9ca3af;font-size:10px;'>R:R</span>
      </td>
      <td style="padding:10px 8px;font-size:11px;color:#6b7280;">{drift_cell}</td>
      <td style="padding:10px 8px;font-size:12px;color:#374151;">{c['action']}</td>
    </tr>"""


def _section(items: list, label: str, accent: str) -> str:
    if not items:
        return ""
    header = f"""
    <tr>
      <td colspan="12" style="padding:14px 8px 6px;font-size:11px;font-weight:700;
          letter-spacing:1.5px;text-transform:uppercase;color:{accent};
          border-bottom:2px solid {accent};">{label}</td>
    </tr>"""
    return header + "".join(_row(r) for r in items)


def _btst_row(r: dict) -> str:
    p           = r["pick"]
    c           = r["classification"]
    ticker_disp = p.get("ticker", "").replace(".NS", "")
    ltp         = f"₹{c['ltp']:,.1f}" if c["ltp"] is not None else "N/A"
    off_high    = f"{c['pct_off_high']:.1f}%" if c.get("pct_off_high") is not None else "—"
    captured    = f"{c['pct_captured']:.0f}%" if c.get("pct_captured") is not None else "—"
    tier        = f"T{p.get('tier','?')}"
    sl          = p.get("sl", 0) or 0
    t1          = p.get("t1", 0) or 0

    diag = r.get("intraday_diag")
    if diag:
        diag_disp = (f"<span style='color:#6b7280;'>{diag['label']}</span><br>"
                      f"<span style='font-size:10px;color:#9ca3af;'>Z {diag['z']:+.1f} · Eff {diag['efficiency']:.2f}</span>")
    else:
        diag_disp = "<span style='color:#d1d5db;'>—</span>"

    return f"""
    <tr style="border-bottom:1px solid #e5e7eb;">
      <td style="padding:10px 8px;">
        <span style="font-weight:700;font-size:14px;color:#111827;">{ticker_disp}</span><br>
        <span style="font-size:11px;color:#9ca3af;">{tier} · Score {p.get('score','')}</span>
      </td>
      <td style="padding:10px 8px;font-size:12px;color:#6b7280;">{p.get('sector','')}</td>
      <td style="padding:10px 8px;text-align:center;">
        <span style="background:{c['bg']};color:{c['color']};padding:4px 10px;
            border-radius:4px;font-size:11px;font-weight:700;white-space:nowrap;">
            {c['label']}</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-weight:700;color:#111827;">{ltp}</td>
      <td style="padding:10px 8px;text-align:right;">{_fmt_rvol(c['rvol'], c.get('rvol_src',''))}</td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;color:#6b7280;">{off_high}</td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;color:#6b7280;">{captured}</td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">{diag_disp}</td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        <span style='color:#dc2626;font-weight:600;'>₹{sl:,.1f}</span><br>
        <span style='color:#9ca3af;font-size:10px;'>SL</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-size:12px;">
        <span style='color:#16a34a;font-weight:600;'>₹{t1:,.1f}</span><br>
        <span style='color:#9ca3af;font-size:10px;'>T1</span>
      </td>
      <td style="padding:10px 8px;font-size:12px;color:#374151;">{c['action']}</td>
    </tr>"""


def build_btst_html(results: list, scan_date: str, run_time: str) -> str:
    order = {"ASM_GSM_BLOCKED": 0, "EARNINGS_BLOCKED": 1, "BTST_CANDIDATE": 2,
             "NEAR_T1": 3, "FADED": 4, "VOL_LIGHT": 5,
             "NOT_HELD": 6, "STOPPED_OUT": 7, "DATA_ERROR": 8}
    results = sorted(results, key=lambda r: order.get(r["classification"]["status"], 9))

    asm_gsm_n   = sum(1 for r in results if r["classification"]["status"] == "ASM_GSM_BLOCKED")
    earnings_n  = sum(1 for r in results if r["classification"]["status"] == "EARNINGS_BLOCKED")
    candidate_n = sum(1 for r in results if r["classification"]["status"] == "BTST_CANDIDATE")
    near_t1_n   = sum(1 for r in results if r["classification"]["status"] == "NEAR_T1")
    faded_n     = sum(1 for r in results if r["classification"]["status"] == "FADED")
    vol_light_n = sum(1 for r in results if r["classification"]["status"] == "VOL_LIGHT")
    not_held_n  = sum(1 for r in results if r["classification"]["status"] == "NOT_HELD")
    stopped_n   = sum(1 for r in results if r["classification"]["status"] == "STOPPED_OUT")
    error_n     = sum(1 for r in results if r["classification"]["status"] == "DATA_ERROR")

    summary_parts = []
    if asm_gsm_n:   summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{asm_gsm_n} ASM/GSM</span>")
    if earnings_n:  summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{earnings_n} EARNINGS EVE</span>")
    if candidate_n: summary_parts.append(f"<span style='color:#16a34a;font-weight:700;'>{candidate_n} BTST CANDIDATE{'S' if candidate_n != 1 else ''}</span>")
    if near_t1_n:   summary_parts.append(f"<span style='color:#ca8a04;font-weight:700;'>{near_t1_n} NEAR T1</span>")
    if faded_n:     summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{faded_n} FADED</span>")
    if vol_light_n: summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{vol_light_n} LIGHT VOL</span>")
    if not_held_n:  summary_parts.append(f"<span style='color:#7c3aed;font-weight:700;'>{not_held_n} BELOW PIVOT</span>")
    if stopped_n:   summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{stopped_n} STOPPED OUT</span>")
    if error_n:     summary_parts.append(f"<span style='color:#9ca3af;font-weight:700;'>{error_n} DATA ERROR</span>")
    summary_html = (" &nbsp;|&nbsp; ".join(summary_parts) if summary_parts
                     else "No positions were CONFIRMED today — nothing to evaluate for BTST")

    action_box = ""
    if candidate_n:
        # [2026-08-11] Informational framing, not a recommendation -- see
        # classify_btst()'s BTST_CANDIDATE branch for the backtest context
        # (1,707 historical signals, edge not statistically validated).
        action_box = f"""
        <div style="background:#fffbeb;border-left:4px solid #ca8a04;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#92400e;font-size:15px;">
            {candidate_n} position(s) meet closing-strength criteria — not a hold recommendation
          </div>
          <div style="color:#78350f;font-size:13px;margin-top:4px;">
            Near today's high, full-day RVOL confirmed, meaningful room left to T1 — these are
            the same checks a genuine setup would clear, but backtesting hasn't yet shown this
            filter predicts a profitable overnight hold (see footer). Use your own judgment.
          </div>
        </div>"""
    elif results:
        action_box = f"""
        <div style="background:#fef2f2;border-left:4px solid #dc2626;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#991b1b;font-size:15px;">
            NO BTST CANDIDATES — square off today's confirmed positions
          </div>
          <div style="color:#7f1d1d;font-size:13px;margin-top:4px;">
            None of today's confirmed setups still clear the closing-strength
            and R:R bar for holding overnight.
          </div>
        </div>"""

    rows_html = "".join(_btst_row(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f9fafb;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
<div style="max-width:1000px;margin:24px auto;background:#ffffff;
    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">

  <div style="background:#0f172a;padding:20px 28px;">
    <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;">
        NSE MOMENTUM DISCOVERY &ndash; V{VERSION}</div>
    <div style="color:#f1f5f9;font-size:22px;font-weight:700;margin-top:4px;">
        BTST Scan — Closing Strength Check (informational)</div>
    <div style="color:#64748b;font-size:13px;margin-top:2px;">
        {scan_date} &nbsp;·&nbsp; Run at {run_time} IST</div>
  </div>

  <div style="background:#f8fafc;border-bottom:1px solid #e2e8f0;
      padding:14px 28px;font-size:14px;">
    {summary_html}
  </div>

  {action_box}

  <div style="padding:8px 28px 28px;">
    <table style="width:100%;border-collapse:collapse;margin-top:8px;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Ticker</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Sector</th>
          <th style="padding:10px 8px;text-align:center;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Status</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">CMP</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;" title="Elapsed-time-matched RVOL from market open to now — a full-day figure at 15:15, not the 45min morning window">RVOL (Full-Day)</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;" title="How far CMP has faded from today's intraday high">Off High</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;" title="% of the entry-to-T1 move already captured">To T1 Used</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#9ca3af;letter-spacing:1px;text-transform:uppercase;" title="EXPERIMENTAL, paper-mode only — see footer. Is today's move (so far) unusually large for this stock AND clean/trending, or is it choppy? Does not affect status or SL/T1.">Intraday Read (exp.)</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#dc2626;letter-spacing:1px;text-transform:uppercase;">SL</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#16a34a;letter-spacing:1px;text-transform:uppercase;">T1</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Action</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px;
      font-size:11px;color:#94a3b8;">
    Not SEBI-registered investment advice. All trading involves capital risk.
    BTST = Buy Today Sell Tomorrow — holds carry overnight gap risk. SL = hard stop, do not widen.
    Only positions CONFIRMED at an earlier checkpoint today are evaluated here.<br><br>
    <strong>On "BTST CANDIDATE":</strong> this label means a position meets the closing-strength
    and R:R filter (near today's high, RVOL confirmed, room left to T1) — it does NOT mean this
    filter has been shown to predict a profitable overnight hold. A backtest against 1,707
    historical signals (2026-08-11) found the filtered subset was not statistically distinguishable
    from an unfiltered baseline once corrected for testing multiple exit definitions. Treat this
    section as informational context, not a recommendation, until re-validated with more data.<br><br>
    <strong>On "Intraday Read":</strong> EXPERIMENTAL, paper-mode only — does not affect status,
    SL, T1, or any decision this email implies. Combines how large today's move is relative to
    this stock's normal daily volatility (Z) with how much of today's total price travel was net
    progress vs. back-and-forth (Efficiency, 0-1). Backtested against 153 historical BTST-confirmed
    signals (2015-2026) and found UNDERPOWERED — no checkpoint reached statistical significance,
    and results flipped direction on the smallest sample buckets. Shown here purely to accumulate
    real observations for a future re-test, not because it's been shown to work.
  </div>
</div>
</body></html>"""


def build_stale_html(scan_date_str: str, today_iso: str, run_time: str) -> str:
    """Email sent when picks_latest.json is genuinely stale."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f9fafb;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
<div style="max-width:600px;margin:24px auto;background:#ffffff;
    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">
  <div style="background:#0f172a;padding:20px 28px;">
    <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;">
        NSE MOMENTUM DISCOVERY &ndash; V{VERSION}</div>
    <div style="color:#f1f5f9;font-size:22px;font-weight:700;margin-top:4px;">
        10am Confirmation — STALE DATA</div>
    <div style="color:#64748b;font-size:13px;margin-top:2px;">
        Run at {run_time} IST</div>
  </div>
  <div style="background:#fef2f2;border-left:4px solid #dc2626;
      margin:24px;padding:16px;border-radius:0 4px 4px 0;">
    <div style="font-weight:700;color:#991b1b;font-size:15px;">
      Evening scan did not run
    </div>
    <div style="color:#7f1d1d;font-size:13px;margin-top:8px;">
      picks_latest.json is dated <strong>{scan_date_str}</strong> which is older than
      the last trading day before <strong>{today_iso}</strong>.<br><br>
      No confirmation performed. Do not trade today until the evening scan runs
      and produces a fresh picks file.
    </div>
  </div>
  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px;
      font-size:11px;color:#94a3b8;">
    Not SEBI-registered investment advice. All trading involves capital risk.
  </div>
</div>
</body></html>"""


def build_html(results: list, scan_date: str, run_time: str, meta: dict = None) -> str:
    meta = meta or {}
    # [2026-08-18] scan_date as passed in has historically been TODAY's
    # display date, not the picks' actual Bhavcopy vintage (which is
    # normally yesterday's trading day by design). Prefer the verified
    # meta field for the header so "Picks from Bhavcopy {date}" is
    # actually true, not just today's date relabeled.
    scan_date = meta.get("bhavcopy_trading_date") or scan_date
    order = {
        "CONFIRMED": 0, "CONFIRMED_PENDING_RVOL": 1, "CONFIRMED_LOW_VOL": 2,
        "BREAKOUT_NO_VOLUME": 3, "PENDING": 4, "MISSED": 5, "BROKEN": 6,
        "DATA_ERROR": 7
    }
    results = sorted(results,
                     key=lambda r: order.get(r["classification"]["status"], 9))

    t1 = [r for r in results if r["pick"].get("tier") == 1]
    t2 = [r for r in results if r["pick"].get("tier") == 2]

    # [BUG-2] Accurate counts per status
    confirmed_n     = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED")
    pending_rvol_n  = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED_PENDING_RVOL")
    low_vol_n       = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED_LOW_VOL")
    no_vol_n        = sum(1 for r in results if r["classification"]["status"] == "BREAKOUT_NO_VOLUME")
    pending_n       = sum(1 for r in results if r["classification"]["status"] == "PENDING")
    missed_n        = sum(1 for r in results if r["classification"]["status"] == "MISSED")
    broken_n        = sum(1 for r in results if r["classification"]["status"] == "BROKEN")
    error_n         = sum(1 for r in results if r["classification"]["status"] == "DATA_ERROR")

    summary_parts = []
    if confirmed_n:    summary_parts.append(f"<span style='color:#16a34a;font-weight:700;'>{confirmed_n} CONFIRMED</span>")
    if pending_rvol_n: summary_parts.append(f"<span style='color:#2563eb;font-weight:700;'>{pending_rvol_n} VOL PENDING</span>")
    if low_vol_n:      summary_parts.append(f"<span style='color:#ca8a04;font-weight:700;'>{low_vol_n} LOW VOLUME</span>")
    if no_vol_n:       summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{no_vol_n} NO VOLUME</span>")
    if pending_n:      summary_parts.append(f"<span style='color:#1d4ed8;font-weight:700;'>{pending_n} PENDING</span>")
    if missed_n:       summary_parts.append(f"<span style='color:#7c3aed;font-weight:700;'>{missed_n} MISSED</span>")
    if broken_n:       summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{broken_n} BROKEN</span>")
    if error_n:        summary_parts.append(f"<span style='color:#9ca3af;font-weight:700;'>{error_n} DATA ERROR</span>")
    summary_html = " &nbsp;|&nbsp; ".join(summary_parts)

    # Action box
    action_box = ""
    if confirmed_n:
        pending_note = (
            f'<br>{pending_rvol_n} more setup(s) still awaiting volume data — '
            f'see VOL PENDING rows below, do not enter those yet.'
            if pending_rvol_n else ""
        )
        action_box = f"""
        <div style="background:#f0fdf4;border-left:4px solid #16a34a;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#15803d;font-size:15px;">
            ACTION REQUIRED — {confirmed_n} setup(s) confirmed for entry today
          </div>
          <div style="color:#166534;font-size:13px;margin-top:4px;">
            Place orders now. RVOL >= 1.5x confirmed. Standard position sizing.{pending_note}
          </div>
        </div>"""
    elif pending_rvol_n and not confirmed_n and not low_vol_n:
        action_box = f"""
        <div style="background:#eff6ff;border-left:4px solid #2563eb;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#1d4ed8;font-size:15px;">
            NO ENTRY YET — {pending_rvol_n} setup(s) above pivot, volume unconfirmed
          </div>
          <div style="color:#1e3a8a;font-size:13px;margin-top:4px;">
            Price cleared the pivot but RVOL data hasn't resolved. Wait for the
            next checkpoint — do not enter on price alone.
          </div>
        </div>"""
    elif no_vol_n and not confirmed_n and not low_vol_n:
        action_box = f"""
        <div style="background:#fef2f2;border-left:4px solid #dc2626;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#991b1b;font-size:15px;">
            NO ENTRY TODAY — {no_vol_n} breakout(s) without volume confirmation
          </div>
          <div style="color:#7f1d1d;font-size:13px;margin-top:4px;">
            Breakout without volume is a trap. Wait for volume or fresh setup tomorrow.
          </div>
        </div>"""
    elif missed_n and not confirmed_n:
        action_box = f"""
        <div style="background:#f5f3ff;border-left:4px solid #7c3aed;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#6d28d9;font-size:15px;">
            {missed_n} setup(s) moved too far — no entry today
          </div>
          <div style="color:#5b21b6;font-size:13px;margin-top:4px;">
            Wait for a pullback to entry zone or a fresh setup tomorrow.
          </div>
        </div>"""

    rows_html = (
        _section(t1, "Tier 1 — Top Picks", "#0f172a") +
        _section(t2, "Tier 2 — Aggressive", "#92400e")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f9fafb;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
<div style="max-width:1000px;margin:24px auto;background:#ffffff;
    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">

  <div style="background:#0f172a;padding:20px 28px;">
    <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;">
        NSE MOMENTUM DISCOVERY &ndash; V{VERSION}</div>
    <div style="color:#f1f5f9;font-size:22px;font-weight:700;margin-top:4px;">
        10am Confirmation Report</div>
    <div style="color:#64748b;font-size:13px;margin-top:2px;">
        Picks from Bhavcopy {scan_date} &nbsp;·&nbsp; Confirmed at {run_time} IST</div>
    {"" if not meta.get('code_provenance', {}).get('git_dirty') else
     '<div style="color:#f87171;font-size:11px;margin-top:4px;">'
     '&#9888; Evening scan was a LOCAL/UNCOMMITTED code run &mdash; not the verified production pipeline</div>'}
  </div>

  <div style="background:#f8fafc;border-bottom:1px solid #e2e8f0;
      padding:14px 28px;font-size:14px;">
    {summary_html}
    &nbsp;&nbsp;
    <span style="color:#94a3b8;font-size:12px;">
        {len(results)} picks checked · RVOL = first-45min vs 20-day avg</span>
  </div>

  {action_box}

  <div style="padding:8px 28px 28px;">
    <table style="width:100%;border-collapse:collapse;margin-top:8px;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Ticker</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Sector</th>
          <th style="padding:10px 8px;text-align:center;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Status</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">CMP</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;" title="Elapsed-time-matched intraday RVOL — NOT the same metric as the evening Daily Intelligence Report's RVOL (EOD)">RVOL (Live)</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">vs Pivot</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#dc2626;letter-spacing:1px;text-transform:uppercase;">SL</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#16a34a;letter-spacing:1px;text-transform:uppercase;">T1</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#15803d;letter-spacing:1px;text-transform:uppercase;">T2</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#1e40af;letter-spacing:1px;text-transform:uppercase;">R:R</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">vs Entry</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">Action</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px;
      font-size:11px;color:#94a3b8;">
    Not SEBI-registered investment advice. All trading involves capital risk.
    SL = hard stop, do not widen. BREAKOUT_NO_VOLUME = breakout trap, skip.
    VOL PENDING = price confirmed, RVOL not yet resolved — will retry next checkpoint.
  </div>
</div>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Email sender
# ─────────────────────────────────────────────────────────────────────────────

# [2026-08-18] --no-send dry-run support, mirroring scanner.py's existing
# --dry-run convention. Lets morning-pipeline changes be verified against
# a REAL trading morning's live data without risking sending a wrong/test
# email while iterating.
_NO_SEND = False


def send_email(subject: str, html_body: str):
    if _NO_SEND:
        log.info(f"  [--no-send] Would have sent: {subject!r} ({len(html_body)} chars of HTML) — not sending.")
        return
    recipients = _load_recipients()
    cc = _load_cc_recipients()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = ", ".join(recipients)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.sendmail(GMAIL_ADDRESS, recipients + cc, msg.as_string())
    log.info(f"  Email sent to {recipients}" + (f" (cc: {cc})" if cc else ""))


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-2] Multi-checkpoint state persistence
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIR = "logs"


def _state_path(today_iso: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"confirm_state_{today_iso.replace('-', '')}.json")


def _load_state(path: str) -> dict:
    """{ticker: {"pick": {...}, "classification": {...}}} for already-terminal picks."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"  Could not read state file {path}: {e} — starting fresh")
        return {}


def _save_state(path: str, state: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _persist_confirmation_outcomes(today_iso: str, checkpoint_label: str,
                                    newly_terminal: list):
    """
    Durable counterpart to _save_state()'s JSON file — writes each
    newly-terminal pick into database.schema's confirmation_outcomes
    table so trap rate (BREAKOUT_NO_VOLUME share) can be queried against
    regime over time. Previously this data only existed in
    logs/confirm_state_YYYYMMDD.json files with no retention guarantee
    (only 2 of many days' worth survived on disk as of 2026-08-17).

    Called with ONLY the picks that became terminal THIS run (not the
    full cached state) — TERMINAL_STATUSES are sticky and never
    re-classified, so re-inserting already-persisted rows on a later
    checkpoint the same day would be redundant, not wrong (PRIMARY KEY
    (confirm_date, ticker) makes it an idempotent upsert either way),
    but there's no reason to do the extra DB work.

    Best-effort: a DB write failure here must never block the email
    send or crash the checkpoint run — same fail-open convention as
    every other side-channel logging call in this module.
    """
    if not newly_terminal:
        return
    try:
        init_all_tables()
        conn = get_connection()
        now_iso = datetime.now(IST).isoformat()
        rows = []
        for entry in newly_terminal:
            pick = entry["pick"]
            c    = entry["classification"]
            rows.append((
                today_iso,
                pick.get("ticker", ""),
                pick.get("scan_date"),
                pick.get("tier"),
                pick.get("pattern"),
                pick.get("score"),
                pick.get("regime"),
                pick.get("regime_confidence"),
                c.get("status"),
                c.get("rvol"),
                c.get("rvol_src"),
                c.get("gap_pct"),
                c.get("drift_pct"),
                checkpoint_label,
                now_iso,
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO confirmation_outcomes
               (confirm_date, ticker, scan_date, tier, pattern, score,
                regime, regime_confidence, status, rvol, rvol_src,
                gap_pct, drift_pct, checkpoint, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        conn.close()
        log.info(f"  Persisted {len(rows)} confirmation outcome(s) to DB "
                 f"(confirmation_outcomes table)")
    except Exception as e:
        log.warning(f"  Could not persist confirmation outcomes to DB "
                    f"(non-fatal, JSON state file is unaffected): "
                    f"{type(e).__name__}: {e}")


def _persist_btst_outcomes(today_iso: str, checkpoint_label: str, results: list):
    """
    Durable counterpart to _persist_confirmation_outcomes(), for the
    15:15 "hold overnight?" decision (classify_btst()) rather than the
    morning "enter?" decision (classify()). Previously run_btst_scan()
    had NO persistence whatsoever -- results only ever reached an email,
    so there was no way to ask e.g. "of stocks confirmed to enter, what
    fraction were still BTST_CANDIDATE by the close, and does that vary
    by regime?" without re-reading old emails by hand.

    entry_status (CONFIRMED / CONFIRMED_LOW_VOL, stashed on each result
    dict by run_btst_scan) links each row back to the morning decision
    that made it eligible for this scan at all -- joins to
    confirmation_outcomes on (confirm_date, ticker) for that link, though
    the two rows are NOT the same day's confirm_date in general (a stock
    entered on Monday's morning checkpoint gets BTST-evaluated the SAME
    Monday afternoon, so confirm_date is actually shared here — see
    run_btst_scan's same-day-only candidate pool).

    Writes ALL results every run (not just newly-terminal, unlike the
    morning function) -- run_btst_scan doesn't have TERMINAL_STATUSES/
    multi-checkpoint caching (it's a single 15:15 run, not repeated
    through the morning), so there's no "already persisted" state to
    avoid re-writing. PRIMARY KEY (confirm_date, ticker) makes repeat
    runs on the same day idempotent regardless.

    Best-effort / fail-open, same as _persist_confirmation_outcomes().
    """
    if not results:
        return
    try:
        init_all_tables()
        conn = get_connection()
        now_iso = datetime.now(IST).isoformat()
        rows = []
        for entry in results:
            pick = entry["pick"]
            c    = entry["classification"]
            rows.append((
                today_iso,
                pick.get("ticker", ""),
                pick.get("scan_date"),
                pick.get("tier"),
                pick.get("pattern"),
                pick.get("score"),
                pick.get("regime"),
                pick.get("regime_confidence"),
                entry.get("entry_status"),
                c.get("status"),
                c.get("rvol"),
                c.get("rvol_src"),
                c.get("pct_off_high"),
                c.get("pct_captured"),
                checkpoint_label,
                now_iso,
            ))
        conn.executemany(
            """INSERT OR REPLACE INTO btst_outcomes
               (confirm_date, ticker, scan_date, tier, pattern, score,
                regime, regime_confidence, entry_status, status, rvol,
                rvol_src, pct_off_high, pct_captured, checkpoint, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        conn.close()
        log.info(f"  Persisted {len(rows)} BTST outcome(s) to DB "
                 f"(btst_outcomes table)")
    except Exception as e:
        log.warning(f"  Could not persist BTST outcomes to DB "
                    f"(non-fatal, email already sent): "
                    f"{type(e).__name__}: {e}")


def _get_tolerance(regime_code: str | None) -> dict:
    """
    [FIX-3] Looks up regime-aware drift/extension tolerance. Falls back
    to the fixed module defaults if regime_code is missing or unrecognized
    -- as of 2026-07-28, orchestrator.py always writes "regime" to every
    pick, so a fallback here now means something else went wrong (e.g.
    an old picks_latest.json from before that fix), not a naming mismatch.
    """
    if regime_code and regime_code in REGIME_TOLERANCE:
        return REGIME_TOLERANCE[regime_code]
    if regime_code:
        log.warning(f"  Regime code {regime_code!r} not in REGIME_TOLERANCE — "
                    f"using fixed defaults ({MAX_ENTRY_DRIFT_PCT}%/{MAX_PIVOT_EXTENSION_PCT}%)")
    else:
        log.warning(f"  No regime code found in picks_meta — "
                    f"using fixed defaults ({MAX_ENTRY_DRIFT_PCT}%/{MAX_PIVOT_EXTENSION_PCT}%). "
                    f"Check picks_latest.json's actual regime field name if this is unexpected.")
    return {"max_entry_drift_pct": MAX_ENTRY_DRIFT_PCT, "max_pivot_extension_pct": MAX_PIVOT_EXTENSION_PCT}


# ─────────────────────────────────────────────────────────────────────────────
# [NEW 2026-08-11] 15:15 BTST scan
# ─────────────────────────────────────────────────────────────────────────────

def run_btst_scan(today_iso: str, today_str: str, run_time: str):
    """
    3:15 PM BTST (Buy Today Sell Tomorrow) scan. Only evaluates tickers that
    were CONFIRMED/CONFIRMED_LOW_VOL at some earlier checkpoint today --
    pulled straight from confirm_state_<date>.json (the same file the
    morning checkpoints already write to, see TERMINAL_STATUSES). This is a
    hold-or-exit decision on a position you'd actually be in, not a fresh
    screen of all 20 evening picks -- a stock that was never confirmed for
    entry this morning has nothing to hold overnight regardless of how it
    looks at 15:15.

    Re-fetches live CMP + RVOL (get_rvol's elapsed-time-matched window
    naturally becomes a full-day figure when as_of is ~15:15, no separate
    "full day" RVOL function needed) + today's intraday high, then applies
    classify_btst()'s two gates.
    """
    state_path = _state_path(today_iso)
    state = _load_state(state_path)

    candidates = [
        (ticker, entry["pick"], entry["classification"]["status"])
        for ticker, entry in state.items()
        if entry["classification"]["status"] in ("CONFIRMED", "CONFIRMED_LOW_VOL")
    ]
    log.info(f"  BTST scan: {len(candidates)} position(s) were CONFIRMED today, "
             f"re-checking for closing strength...")

    # Fetched once for the whole scan, not per ticker -- one NSE call, not N.
    asm_gsm_symbols = get_asm_gsm_symbols()

    now_ist = datetime.now(IST)
    results = []
    for ticker_key, pick, entry_status in candidates:
        ticker_raw = pick.get("ticker_raw") or ticker_key
        if not ticker_raw.endswith(".NS"):
            ticker_raw += ".NS"

        log.info(f"  Checking {ticker_key} for BTST...")
        cmp, _cmp_fetched_at, _cmp_src, _cmp_suspect = get_live_price(ticker_raw)
        rvol, rvol_src, _rvol_elapsed = get_rvol(ticker_raw, as_of=now_ist)
        day_high, day_low, dh_src  = get_intraday_high_low(ticker_raw, as_of=now_ist)
        days_to_earnings           = days_to_next_earnings(ticker_raw, as_of=now_ist.date())
        c = classify_btst(pick, cmp, day_high, rvol, rvol_src,
                           asm_gsm_symbols=asm_gsm_symbols,
                           days_to_earnings=days_to_earnings)

        # [2026-08-13, PAPER-MODE] Informational only — see
        # compute_intraday_diagnostic()'s docstring. Never affects `c`
        # (the real classification) and is never allowed to raise.
        intraday_diag = compute_intraday_diagnostic(ticker_raw, as_of=now_ist)

        cmp_disp      = f"₹{cmp:,.1f}" if cmp else "N/A"
        rvol_disp     = f"{rvol:.1f}x" if rvol >= 0 else "N/A"
        off_high_disp = f"{c['pct_off_high']:.1f}%" if c['pct_off_high'] is not None else "N/A"
        log.info(f"    → {c['status']:15s}  CMP={cmp_disp:>10}  RVOL={rvol_disp:>8}  off_high={off_high_disp}")

        results.append({"pick": pick, "classification": c, "intraday_diag": intraday_diag,
                         "entry_status": entry_status})
        time.sleep(0.3)

    candidate_n = sum(1 for r in results if r["classification"]["status"] == "BTST_CANDIDATE")

    subject = (f"[NSE Momentum {run_time} BTST SCAN] "
               f"{candidate_n} BTST CANDIDATE{'S' if candidate_n != 1 else ''} | {today_str}")

    html = build_btst_html(results, today_str, run_time)
    send_email(subject, html)
    _persist_btst_outcomes(today_iso, run_time, results)
    log.info("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help="Label for this run, e.g. '09:20', '09:35', '09:55'. "
                              "Purely for logging/subject clarity -- does not affect logic.")
    parser.add_argument("--final", action="store_true",
                         help="Mark this as the last checkpoint of the morning -- always "
                              "sends a full summary email even if nothing new happened, "
                              "and includes ALL results (terminal + still-pending) rather "
                              "than just what's new since the last checkpoint.")
    parser.add_argument("--btst", action="store_true",
                         help="Run the 15:15 BTST (Buy Today Sell Tomorrow) closing-strength "
                              "scan instead of the standard entry-confirmation checkpoint. "
                              "Only evaluates tickers CONFIRMED at an earlier checkpoint today.")
    parser.add_argument("--no-send", action="store_true",
                         help="Run the full fetch/classify pipeline against REAL live data and "
                              "log what would have been emailed, without actually sending — "
                              "mirrors scanner.py's --dry-run for safe verification of morning-"
                              "pipeline changes on a real trading morning.")
    args = parser.parse_args()
    global _NO_SEND
    _NO_SEND = args.no_send

    # [BUG-3] IST timestamp — not UTC labeled as IST
    now_ist   = datetime.now(IST)
    today_str = now_ist.strftime("%d %b %Y")
    today_iso = now_ist.date().isoformat()
    run_time  = now_ist.strftime("%H:%M")
    checkpoint_label = args.checkpoint or run_time

    log.info(f"NSE Momentum v{VERSION} — Confirmation checkpoint [{checkpoint_label}] starting "
             f"({run_time} IST = {datetime.now(ZoneInfo('UTC')).strftime('%H:%M')} UTC)...")

    if not os.path.exists(PICKS_JSON_PATH):
        log.error(f"{PICKS_JSON_PATH} not found — evening scan must run first")
        raise FileNotFoundError(PICKS_JSON_PATH)

    with open(PICKS_JSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # [2026-08-18] Schema compatibility. picks_latest.json changed from a
    # bare array to {"meta": {...verified provenance...}, "picks": [...]}
    # (see orchestrator.py). A legacy flat-array file post-rollout means the
    # evening scan that produced it did NOT run the current provenance-aware
    # code (uncommitted/stale runner, exactly the class of bug that caused
    # the 2026-08-18 incident) — hard-fail rather than silently falling back
    # to the old, unverified semantics.
    if isinstance(raw, list):
        log.error(f"{PICKS_JSON_PATH} is in the legacy flat-array shape — the "
                   f"evening scan that produced it did not run the current "
                   f"provenance-aware code. Refusing to trust it.")
        stale_html = build_stale_html("unknown (legacy file format)", today_iso, run_time)
        send_email(f"[NSE Momentum Checkpoint] STALE DATA — legacy picks file format | {today_str}",
                   stale_html)
        return
    meta  = raw.get("meta", {}) if isinstance(raw, dict) else {}
    picks = raw.get("picks", []) if isinstance(raw, dict) else []
    log.info(f"  Loaded {len(picks)} picks from {PICKS_JSON_PATH}")

    # [BUG-6 FIX, extended 2026-08-18] Stale data guard — compare scan_date
    # against the LAST TRADING DAY, not literally "today". scan_date ==
    # yesterday's trading day is the CORRECT, expected state (see module
    # docstring). [4f] Previously this was the ONLY check — a stale
    # artifact with a coincidentally-correct scan_date string would sail
    # through undetected. Now also requires meta["bhavcopy_date_verified"]
    # to be True and meta["generated_at"] to be a real, parseable
    # timestamp whose date matches scan_date -- a corrupted/replayed
    # artifact would now need to fake three independent fields, not one.
    scan_date_str = meta.get("bhavcopy_trading_date") or (picks[0].get("scan_date") if picks else "")

    def _fail_stale(reason: str):
        log.error(f"STALE PICKS FILE — {reason}")
        stale_html = build_stale_html(scan_date_str or "unknown", today_iso, run_time)
        send_email(
            f"[NSE Momentum Checkpoint] STALE DATA — evening scan missing | {today_str}",
            stale_html
        )

    if scan_date_str:
        try:
            picks_date = date.fromisoformat(scan_date_str)
            check_staleness(picks_date, today=now_ist.date())
        except StaleDataError as e:
            _fail_stale(str(e))
            return

        if not meta.get("bhavcopy_date_verified", False):
            _fail_stale("meta.bhavcopy_date_verified is False — the evening scan itself "
                        "could not confirm its Bhavcopy data matched the expected trading date.")
            return

        gen_at = meta.get("generated_at")
        if not gen_at:
            _fail_stale("meta.generated_at is missing — cannot verify this file wasn't "
                        "a stale/replayed artifact with a coincidentally-correct date.")
            return
        try:
            gen_at_dt = datetime.fromisoformat(gen_at)
        except ValueError:
            _fail_stale(f"meta.generated_at {gen_at!r} is not a parseable timestamp.")
            return
        if gen_at_dt.date().isoformat() != scan_date_str:
            _fail_stale(f"meta.generated_at date ({gen_at_dt.date()}) does not match "
                        f"meta.bhavcopy_trading_date ({scan_date_str}).")
            return

        code_prov = meta.get("code_provenance", {}) or {}
        if code_prov.get("git_dirty") and code_prov.get("is_cloud_runner"):
            _fail_stale("evening scan's code_provenance shows a dirty working tree on "
                        "the cloud runner — its own code state could not be verified.")
            return
        if code_prov.get("git_dirty"):
            log.warning("  Today's picks came from a LOCAL/UNCOMMITTED evening scan run "
                         "(git_dirty=True, not the cloud runner) — proceeding, but this "
                         "will be flagged in the confirmation email too.")

    # [NEW 2026-08-11] BTST scan is a completely separate flow -- different
    # candidate pool (today's already-CONFIRMED positions, not all 20 picks),
    # different classification, different email. Branch out before the
    # regime-tolerance lookup and multi-checkpoint machinery below, neither
    # of which the BTST path uses.
    if args.btst:
        run_btst_scan(today_iso, today_str, run_time)
        return

    # [FIX-3] Regime-aware tolerance — see ASSUMPTION note in module docstring
    # re: confirming the actual key name if this doesn't seem to activate.
    regime_code = meta.get("regime") or (picks[0].get("regime") if picks else None)
    tolerance   = _get_tolerance(regime_code)
    log.info(f"  Regime: {regime_code!r} → tolerance "
             f"entry_drift={tolerance['max_entry_drift_pct']}% "
             f"pivot_ext={tolerance['max_pivot_extension_pct']}%")

    # [FIX-2] Load prior state for today. Empty state = this is the first
    # checkpoint of the morning.
    state_path   = _state_path(today_iso)
    prior_state  = _load_state(state_path)
    is_first_run = len(prior_state) == 0
    log.info(f"  Prior state: {len(prior_state)} ticker(s) already terminal "
              f"({'first checkpoint today' if is_first_run else 'resuming'})")

    results       = []
    newly_terminal = []

    for pick in picks:
        ticker_raw = pick.get("ticker_raw") or pick.get("ticker", "")
        if not ticker_raw.endswith(".NS"):
            ticker_raw += ".NS"
        ticker_key = pick.get("ticker", ticker_raw)

        # [FIX-2] Reuse cached terminal classification — don't re-fetch
        # price/RVOL or re-email something already resolved this morning.
        if ticker_key in prior_state:
            cached = prior_state[ticker_key]
            cached_status = cached["classification"]["status"]

            # [2026-08-18, 4c] BROKEN gets a lightweight re-verification
            # instead of being trusted forever — see NO_RESEND_STATUSES
            # docstring above. Cheap: one live-price call, no RVOL refetch
            # needed since BROKEN only depends on cmp <= sl.
            if cached_status == "BROKEN":
                recheck_cmp, recheck_fetched_at, _recheck_src, _recheck_suspect = get_live_price(ticker_raw)
                recheck_prov = verify_intraday_freshness(
                    recheck_fetched_at, now_ist, source_name="live_price"
                )
                sl = pick.get("sl", 0)
                if recheck_prov.ok and recheck_cmp is not None and recheck_cmp > sl:
                    log.info(f"  {ticker_key}: cached BROKEN but CMP ₹{recheck_cmp:,.1f} "
                             f"has recovered above SL ₹{sl:,.1f} — re-classifying from scratch")
                    del prior_state[ticker_key]
                    # falls through to the full re-check below
                else:
                    log.info(f"  {ticker_key}: cached [BROKEN] re-verified still <= SL "
                             f"— skipping re-email (not new information)")
                    results.append(cached)
                    continue
            else:
                log.info(f"  {ticker_key}: cached [{cached_status}] "
                         f"from earlier checkpoint — skipping re-check")
                results.append(cached)
                continue

        log.info(f"  Checking {ticker_key}...")
        cmp, cmp_fetched_at, cmp_src, cmp_suspect = get_live_price(ticker_raw)
        price_prov     = verify_intraday_freshness(cmp_fetched_at, now_ist, source_name="live_price")
        rvol, rvol_src, rvol_elapsed = get_rvol(ticker_raw, as_of=now_ist)
        orh, orl, or_src = get_opening_range(ticker_raw, as_of=now_ist)
        c              = classify(pick, cmp, rvol, rvol_src,
                                   max_entry_drift_pct=tolerance["max_entry_drift_pct"],
                                   max_pivot_extension_pct=tolerance["max_pivot_extension_pct"],
                                   opening_range=(orh, orl, or_src),
                                   price_verified=price_prov.ok, price_source=cmp_src)
        c["price_fetched_at"] = cmp_fetched_at.strftime("%H:%M:%S") if cmp_fetched_at else None
        c["price_source"]     = cmp_src
        c["price_suspect"]    = cmp_suspect
        c["rvol_elapsed_minutes"] = rvol_elapsed

        log.info(
            f"    → {c['status']:25s}  "
            f"CMP={f'₹{cmp:,.1f}' if cmp else 'N/A':>10} ({cmp_src})  "
            f"RVOL={f'{rvol:.1f}x ({rvol_src})' if rvol >= 0 else 'N/A':>20}"
        )
        entry = {"pick": pick, "classification": c}
        results.append(entry)

        if c["status"] in TERMINAL_STATUSES:
            prior_state[ticker_key] = entry
            newly_terminal.append(entry)

        time.sleep(0.3)

    _save_state(state_path, prior_state)
    _persist_confirmation_outcomes(today_iso, checkpoint_label, newly_terminal)

    # ── Decide whether/what to send ──────────────────────────────────────────
    # First checkpoint of the day: always send (baseline picture).
    # Final checkpoint: always send, full results (terminal + still-pending).
    # Middle checkpoints: only send if something NEW became terminal this
    # run, and only show those newly-terminal picks -- avoids duplicate
    # near-identical emails across 3+ checkpoints per morning.
    if args.final or is_first_run:
        email_results = results
        scope_label   = "FULL SUMMARY" if args.final else "FIRST CHECK"
    elif newly_terminal:
        email_results = newly_terminal
        scope_label   = "NEW THIS CHECK"
    else:
        log.info(f"  No new terminal picks since last checkpoint — skipping email "
                 f"(not --final, not first run of the day).")
        log.info("  Done.")
        return

    confirmed_n     = sum(1 for r in email_results if r["classification"]["status"] == "CONFIRMED")
    pending_rvol_n  = sum(1 for r in email_results if r["classification"]["status"] == "CONFIRMED_PENDING_RVOL")
    low_vol_n       = sum(1 for r in email_results if r["classification"]["status"] == "CONFIRMED_LOW_VOL")
    no_vol_n        = sum(1 for r in email_results if r["classification"]["status"] == "BREAKOUT_NO_VOLUME")
    missed_n        = sum(1 for r in email_results if r["classification"]["status"] == "MISSED")
    broken_n        = sum(1 for r in email_results if r["classification"]["status"] == "BROKEN")
    pending_n       = sum(1 for r in email_results if r["classification"]["status"] == "PENDING")

    parts = []
    if confirmed_n:    parts.append(f"{confirmed_n} CONFIRMED")
    if pending_rvol_n: parts.append(f"{pending_rvol_n} VOL PENDING")
    if low_vol_n:      parts.append(f"{low_vol_n} LOW VOL")
    if no_vol_n:       parts.append(f"{no_vol_n} NO VOL BREAKOUT")
    if missed_n:       parts.append(f"{missed_n} MISSED")
    if broken_n:       parts.append(f"{broken_n} BROKEN")
    if pending_n:      parts.append(f"{pending_n} PENDING")
    if not parts:      parts.append("No actionable setups")

    subject = f"[NSE Momentum {checkpoint_label} {scope_label}] {' | '.join(parts)} | {today_str}"

    html = build_html(email_results, today_str, run_time, meta=meta)
    send_email(subject, html)
    log.info("  Done.")


if __name__ == "__main__":
    main()
