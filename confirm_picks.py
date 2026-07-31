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
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from market_calendar.staleness_check import check_staleness, StaleDataError
import dhan_rvol

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
TERMINAL_STATUSES = {"CONFIRMED", "CONFIRMED_LOW_VOL", "BREAKOUT_NO_VOLUME", "MISSED", "BROKEN"}
# ─────────────────────────────────────────────────────────────────────────────


def _load_recipients() -> list:
    path = "recipients.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return [GMAIL_ADDRESS]


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
            except Exception:
                pass
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


def _rvol_dhan(ticker_raw: str, as_of: datetime = None) -> float:
    """
    [NEW] Primary RVOL source — Dhan's paid Data API, elapsed-time-matched
    (see dhan_rvol.py). Returns rvol as a float, or -1.0 on any failure
    (too early in session, security_id not mapped, network/API error, no
    historical data) so it slots into the same fallback chain as the
    tvDatafeed/yfinance functions below.
    """
    symbol = ticker_raw.replace(".NS", "")
    try:
        result = dhan_rvol.compute_rvol(symbol, as_of=as_of)
    except Exception as e:
        log.warning(f"    [Dhan] {symbol}: RVOL call raised {type(e).__name__}: {e}")
        return -1.0

    if "error" in result:
        log.info(f"    [Dhan] {symbol}: {result['error']}")
        return -1.0

    log.info(f"    [Dhan] {symbol}: RVOL = {result['rvol']:.2f}x "
             f"({result.get('elapsed_minutes', '?')}min elapsed)")
    return result["rvol"]


def get_rvol(ticker: str, as_of: datetime = None) -> tuple:
    """
    Returns (rvol_float, source_label).
    Tries Dhan first (paid, elapsed-time-matched, most reliable), falls
    back to tvDatafeed, then yfinance, then N/A.
    """
    ticker_raw = ticker if ticker.endswith(".NS") else ticker + ".NS"

    rvol = _rvol_dhan(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "Dhan"

    log.info(f"    Dhan failed for {ticker_raw} — trying tvDatafeed fallback")
    rvol = _rvol_tvdatafeed(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "tvDatafeed"

    log.info(f"    tvDatafeed failed for {ticker_raw} — trying yfinance fallback")
    rvol = _rvol_yfinance_fallback(ticker_raw, as_of=as_of)
    if rvol >= 0:
        return rvol, "yfinance"

    log.warning(f"    All RVOL sources failed for {ticker_raw} — returning N/A")
    return -1.0, "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Live price
# ─────────────────────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> float | None:
    try:
        import yfinance as yf
        t    = ticker if ticker.endswith(".NS") else ticker + ".NS"
        fast = yf.Ticker(t).fast_info
        ltp  = float(fast.last_price)
        return ltp if ltp > 0 else None
    except Exception as e:
        log.warning(f"  [WARN] live price failed for {ticker}: {e}")
        return None


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
                except Exception:
                    pass
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


def classify(pick: dict, cmp: float | None, rvol: float, rvol_src: str,
             max_entry_drift_pct: float = MAX_ENTRY_DRIFT_PCT,
             max_pivot_extension_pct: float = MAX_PIVOT_EXTENSION_PCT,
             opening_range: tuple = (None, None, None)) -> dict:
    """
    Status set (v5.3):
      CONFIRMED           — above pivot, RVOL >= 1.5x
      CONFIRMED_LOW_VOL   — above pivot, RVOL 1.0-1.5x
      BREAKOUT_NO_VOLUME  — above pivot, RVOL < 1.0x  ← NEW
      MISSED              — entry drift > threshold, OR pivot extension > threshold with low RVOL
      PENDING             — below pivot
      BROKEN              — at or below stop loss
      DATA_ERROR          — price feed failed

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
# Email HTML
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_rvol(rvol: float, src: str) -> str:
    if rvol < 0:
        return "<span style='color:#9ca3af;'>N/A</span>"
    color = "#16a34a" if rvol >= 1.5 else "#ca8a04" if rvol >= 1.0 else "#dc2626"
    icon  = "✅" if rvol >= 1.5 else "⚠️" if rvol >= 1.0 else "❌"
    return (
        f"<span style='color:{color};font-weight:700;'>{icon} {rvol:.1f}x (Live)</span>"
        f"<br><span style='font-size:10px;color:#9ca3af;'>{src}</span>"
    )


def _row(r: dict) -> str:
    p           = r["pick"]
    c           = r["classification"]
    ticker_disp = p.get("ticker", "").replace(".NS", "")
    ltp         = f"₹{c['ltp']:,.1f}" if c["ltp"] is not None else "N/A"
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
      <td style="padding:10px 8px;text-align:right;">{_fmt_rvol(c['rvol'], c.get('rvol_src',''))}</td>
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


def build_html(results: list, scan_date: str, run_time: str) -> str:
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
        {scan_date} &nbsp;·&nbsp; Run at {run_time} IST</div>
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

def send_email(subject: str, html_body: str):
    recipients = _load_recipients()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.sendmail(GMAIL_ADDRESS, recipients, msg.as_string())
    log.info(f"  Email sent to {recipients}")


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
    args = parser.parse_args()

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
        picks = json.load(f)
    log.info(f"  Loaded {len(picks)} picks from {PICKS_JSON_PATH}")

    # [BUG-6 FIX] Stale data guard — compare scan_date against the LAST
    # TRADING DAY, not literally "today". scan_date == yesterday's
    # trading day is the CORRECT, expected state (see module docstring).
    picks_meta    = picks[0] if picks else {}
    scan_date_str = picks_meta.get("scan_date", "")

    if scan_date_str:
        try:
            picks_date = date.fromisoformat(scan_date_str)
            check_staleness(picks_date)
        except StaleDataError as e:
            log.error(f"STALE PICKS FILE — {e}")
            stale_html = build_stale_html(scan_date_str, today_iso, run_time)
            send_email(
                f"[NSE Momentum Checkpoint] STALE DATA — evening scan missing | {today_str}",
                stale_html
            )
            return

    # [FIX-3] Regime-aware tolerance — see ASSUMPTION note in module docstring
    # re: confirming the actual key name if this doesn't seem to activate.
    regime_code = picks_meta.get("regime")
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
            log.info(f"  {ticker_key}: cached [{cached['classification']['status']}] "
                     f"from earlier checkpoint — skipping re-check")
            results.append(cached)
            continue

        log.info(f"  Checking {ticker_key}...")
        cmp            = get_live_price(ticker_raw)
        rvol, rvol_src = get_rvol(ticker_raw, as_of=now_ist)
        orh, orl, or_src = get_opening_range(ticker_raw, as_of=now_ist)
        c              = classify(pick, cmp, rvol, rvol_src,
                                   max_entry_drift_pct=tolerance["max_entry_drift_pct"],
                                   max_pivot_extension_pct=tolerance["max_pivot_extension_pct"],
                                   opening_range=(orh, orl, or_src))

        log.info(
            f"    → {c['status']:25s}  "
            f"CMP={f'₹{cmp:,.1f}' if cmp else 'N/A':>10}  "
            f"RVOL={f'{rvol:.1f}x ({rvol_src})' if rvol >= 0 else 'N/A':>20}"
        )
        entry = {"pick": pick, "classification": c}
        results.append(entry)

        if c["status"] in TERMINAL_STATUSES:
            prior_state[ticker_key] = entry
            newly_terminal.append(entry)

        time.sleep(0.3)

    _save_state(state_path, prior_state)

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

    html = build_html(email_results, today_str, run_time)
    send_email(subject, html)
    log.info("  Done.")


if __name__ == "__main__":
    main()
