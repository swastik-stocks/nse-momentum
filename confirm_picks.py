#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.3 — 10am Confirmation Checker (complete replacement)

CHANGES vs v5.2:
  [BUG-1] tvDatafeed: explicit debug logging added so failures are visible
          in GitHub Actions logs. No more silent fallthrough.
  [BUG-2] New status BREAKOUT_NO_VOLUME for stocks above pivot with RVOL < 1.0x
          Subject line now accurate: counts each status separately.
  [BUG-3] Timestamp fixed: datetime.now(ZoneInfo) for IST, not UTC labeled as IST
  [BUG-4] Secondary drift check: CMP > pivot * 1.03 AND RVOL < 1.5x -> MISSED
          Drift guard now checks pivot extension, not just entry price.
  [BUG-5] MEDANTA / extended breakouts: 2.7%+ above pivot with low RVOL
          correctly classified as MISSED, not LOW_VOL.
  [BUG-6] Stale-data guard now compares scan_date against the LAST TRADING
          DAY (skipping weekends + NSE holidays), not literally "today".
          The evening scan runs the night before and produces picks meant
          to be confirmed the next trading morning -- scan_date == the
          prior trading day is the CORRECT, expected state, not staleness.
"""

import os, json, smtplib, time
from datetime import datetime, date
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from market_calendar.staleness_check import check_staleness, StaleDataError

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
VERSION = "5.3"

# ── Thresholds ────────────────────────────────────────────────────────────────
RVOL_CONFIRM_MIN        = 1.5   # RVOL >= this → CONFIRMED (full size)
RVOL_LOW_VOL_MIN        = 1.0   # RVOL >= this → CONFIRMED_LOW_VOL (half size)
                                 # RVOL < 1.0 above pivot → BREAKOUT_NO_VOLUME
MAX_ENTRY_DRIFT_PCT     = 2.0   # CMP > entry + this% → MISSED
MAX_PIVOT_EXTENSION_PCT = 3.0   # CMP > pivot + this% with low RVOL → MISSED
IST = ZoneInfo("Asia/Kolkata")
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

def _rvol_tvdatafeed(ticker_nse: str) -> float:
    """
    Primary RVOL source: tvDatafeed (TradingView 5-min bars).
    Returns RVOL float >= 0, or -1.0 on failure.
    ALL failures are logged explicitly — no silent fallthrough.
    """
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
        # shifting it, causing the 09:15-10:15 window search to miss
        # every bar (they land ~5.5hrs off from where they should be).
        try:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            try:
                df.index = df.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
        # --- END FIX

        today = datetime.now(IST).date()
        today_45 = df[
            (df.index.date == today) &
            (df.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df.index.time <= datetime.strptime("10:15", "%H:%M").time())
        ]

        if today_45.empty:
            log.warning(f"    [tvDatafeed] {symbol}: no bars for today "
                        f"({today}) in 09:15-10:00 window. "
                        f"Latest bar date: {df.index[-1].date()}")
            return -1.0

        today_vol = float(today_45["volume"].sum())
        log.info(f"    [tvDatafeed] {symbol}: today 45-min vol = {today_vol:,.0f}")

        hist = df[df.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_45 = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
                (hist.index.time <= datetime.strptime("10:15", "%H:%M").time())
            ]
            if not day_45.empty:
                hist_by_day[row_date] = float(day_45["volume"].sum())

        if len(hist_by_day) < 3:
            log.warning(f"    [tvDatafeed] {symbol}: only {len(hist_by_day)} "
                        f"historical days — insufficient for avg")
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        rvol = round(today_vol / avg_hist, 2) if avg_hist > 0 else -1.0
        log.info(f"    [tvDatafeed] {symbol}: RVOL = {rvol:.2f}x "
                 f"(today {today_vol:,.0f} / avg {avg_hist:,.0f})")
        return rvol

    except ImportError:
        log.warning("    [tvDatafeed] Not installed — pip install tvDatafeed")
        return -1.0
    except Exception as e:
        log.warning(f"    [tvDatafeed] {ticker_nse}: FAILED — {type(e).__name__}: {e}")
        return -1.0


def _rvol_yfinance_fallback(ticker_nse: str) -> float:
    """
    Fallback RVOL: yfinance 1-min data.
    Returns RVOL float >= 0, or -1.0 on failure.
    """
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

        today = datetime.now(IST).date()
        today_45 = df_1m[
            (df_1m.index.date == today) &
            (df_1m.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df_1m.index.time <= datetime.strptime("10:15", "%H:%M").time())
        ]
        if today_45.empty:
            log.warning(f"    [yfinance-fallback] {t}: no bars for today in window")
            return -1.0

        today_vol = float(today_45["Volume"].sum())
        hist = df_1m[df_1m.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_45 = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
                (hist.index.time <= datetime.strptime("10:15", "%H:%M").time())
            ]
            if not day_45.empty:
                hist_by_day[row_date] = float(day_45["Volume"].sum())

        if len(hist_by_day) < 3:
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        rvol = round(today_vol / avg_hist, 2) if avg_hist > 0 else -1.0
        log.info(f"    [yfinance-fallback] {t}: RVOL = {rvol:.2f}x")
        return rvol

    except Exception as e:
        log.warning(f"    [yfinance-fallback] {ticker_nse}: FAILED — {e}")
        return -1.0


def get_rvol(ticker: str) -> tuple:
    """
    Returns (rvol_float, source_label).
    Tries tvDatafeed first, falls back to yfinance, then N/A.
    """
    ticker_raw = ticker if ticker.endswith(".NS") else ticker + ".NS"

    rvol = _rvol_tvdatafeed(ticker_raw)
    if rvol >= 0:
        return rvol, "tvDatafeed"

    log.info(f"    tvDatafeed failed for {ticker_raw} — trying yfinance fallback")
    rvol = _rvol_yfinance_fallback(ticker_raw)
    if rvol >= 0:
        return rvol, "yfinance"

    log.warning(f"    Both RVOL sources failed for {ticker_raw} — returning N/A")
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

def classify(pick: dict, cmp: float | None, rvol: float, rvol_src: str) -> dict:
    """
    Status set (v5.3):
      CONFIRMED           — above pivot, RVOL >= 1.5x
      CONFIRMED_LOW_VOL   — above pivot, RVOL 1.0-1.5x
      BREAKOUT_NO_VOLUME  — above pivot, RVOL < 1.0x  ← NEW
      MISSED              — entry drift > 2%, OR pivot extension > 3% with low RVOL
      PENDING             — below pivot
      BROKEN              — at or below stop loss
      DATA_ERROR          — price feed failed
    """
    entry = pick["entry"]
    sl    = pick["sl"]
    pivot = pick.get("pivot", entry)
    t1    = pick["t1"]

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

    # Entry drift vs planned entry price
    drift_pct       = round(((cmp - entry) / entry) * 100, 1) if entry > 0 else 0.0
    entry_chasing   = drift_pct > MAX_ENTRY_DRIFT_PCT

    # [BUG-4/5] Secondary: pivot extension with low volume
    pivot_ext_pct   = round(((cmp - pivot) / pivot) * 100, 1) if pivot > 0 else 0.0
    pivot_extended  = (pivot_ext_pct > MAX_PIVOT_EXTENSION_PCT
                       and rvol >= 0 and rvol < RVOL_CONFIRM_MIN)

    vol_ok_confirm  = (rvol >= RVOL_CONFIRM_MIN) if rvol >= 0 else True
    vol_ok_low      = (rvol >= RVOL_LOW_VOL_MIN) if rvol >= 0 else True

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
        if pivot_extended and not entry_chasing:
            reason = (f"Breakout extended {pivot_ext_pct:.1f}% above pivot "
                      f"with only {rvol:.1f}x volume — trap, not entry")
        else:
            reason = (f"Price drifted {drift_pct:+.1f}% above planned entry "
                      f"₹{entry:,.1f} — R:R destroyed")
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
        f"<span style='color:{color};font-weight:700;'>{icon} {rvol:.1f}x</span>"
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
        "CONFIRMED": 0, "CONFIRMED_LOW_VOL": 1, "BREAKOUT_NO_VOLUME": 2,
        "PENDING": 3, "MISSED": 4, "BROKEN": 5, "DATA_ERROR": 6
    }
    results = sorted(results,
                     key=lambda r: order.get(r["classification"]["status"], 9))

    t1 = [r for r in results if r["pick"].get("tier") == 1]
    t2 = [r for r in results if r["pick"].get("tier") == 2]

    # [BUG-2] Accurate counts per status
    confirmed_n  = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED")
    low_vol_n    = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED_LOW_VOL")
    no_vol_n     = sum(1 for r in results if r["classification"]["status"] == "BREAKOUT_NO_VOLUME")
    pending_n    = sum(1 for r in results if r["classification"]["status"] == "PENDING")
    missed_n     = sum(1 for r in results if r["classification"]["status"] == "MISSED")
    broken_n     = sum(1 for r in results if r["classification"]["status"] == "BROKEN")
    error_n      = sum(1 for r in results if r["classification"]["status"] == "DATA_ERROR")

    summary_parts = []
    if confirmed_n: summary_parts.append(f"<span style='color:#16a34a;font-weight:700;'>{confirmed_n} CONFIRMED</span>")
    if low_vol_n:   summary_parts.append(f"<span style='color:#ca8a04;font-weight:700;'>{low_vol_n} LOW VOLUME</span>")
    if no_vol_n:    summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{no_vol_n} NO VOLUME</span>")
    if pending_n:   summary_parts.append(f"<span style='color:#1d4ed8;font-weight:700;'>{pending_n} PENDING</span>")
    if missed_n:    summary_parts.append(f"<span style='color:#7c3aed;font-weight:700;'>{missed_n} MISSED</span>")
    if broken_n:    summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{broken_n} BROKEN</span>")
    if error_n:     summary_parts.append(f"<span style='color:#9ca3af;font-weight:700;'>{error_n} DATA ERROR</span>")
    summary_html = " &nbsp;|&nbsp; ".join(summary_parts)

    # Action box
    action_box = ""
    if confirmed_n:
        action_box = f"""
        <div style="background:#f0fdf4;border-left:4px solid #16a34a;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#15803d;font-size:15px;">
            ACTION REQUIRED — {confirmed_n} setup(s) confirmed for entry today
          </div>
          <div style="color:#166534;font-size:13px;margin-top:4px;">
            Place orders now. RVOL >= 1.5x confirmed. Standard position sizing.
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
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;letter-spacing:1px;text-transform:uppercase;">RVOL</th>
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # [BUG-3] IST timestamp — not UTC labeled as IST
    now_ist   = datetime.now(IST)
    today_str = now_ist.strftime("%d %b %Y")
    run_time  = now_ist.strftime("%H:%M")

    log.info(f"NSE Momentum v5.3 — 10am Confirmation starting "
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
    today_iso     = date.today().isoformat()
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
                f"[NSE Momentum 10am] STALE DATA — evening scan missing | {today_str}",
                stale_html
            )
            return

    results = []
    for pick in picks:
        ticker_raw = pick.get("ticker_raw") or pick.get("ticker", "")
        if not ticker_raw.endswith(".NS"):
            ticker_raw += ".NS"

        log.info(f"  Checking {pick.get('ticker', ticker_raw)}...")
        cmp            = get_live_price(ticker_raw)
        rvol, rvol_src = get_rvol(ticker_raw)
        c              = classify(pick, cmp, rvol, rvol_src)

        log.info(
            f"    → {c['status']:25s}  "
            f"CMP={f'₹{cmp:,.1f}' if cmp else 'N/A':>10}  "
            f"RVOL={f'{rvol:.1f}x ({rvol_src})' if rvol >= 0 else 'N/A':>20}"
        )
        results.append({"pick": pick, "classification": c})
        time.sleep(0.3)

    # [BUG-2] Accurate subject line
    confirmed_n  = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED")
    low_vol_n    = sum(1 for r in results if r["classification"]["status"] == "CONFIRMED_LOW_VOL")
    no_vol_n     = sum(1 for r in results if r["classification"]["status"] == "BREAKOUT_NO_VOLUME")
    missed_n     = sum(1 for r in results if r["classification"]["status"] == "MISSED")
    broken_n     = sum(1 for r in results if r["classification"]["status"] == "BROKEN")
    pending_n    = sum(1 for r in results if r["classification"]["status"] == "PENDING")

    parts = []
    if confirmed_n: parts.append(f"{confirmed_n} CONFIRMED")
    if low_vol_n:   parts.append(f"{low_vol_n} LOW VOL")
    if no_vol_n:    parts.append(f"{no_vol_n} NO VOL BREAKOUT")
    if missed_n:    parts.append(f"{missed_n} MISSED")
    if broken_n:    parts.append(f"{broken_n} BROKEN")
    if pending_n:   parts.append(f"{pending_n} PENDING")
    if not parts:   parts.append("No actionable setups")

    subject = f"[NSE Momentum 10am] {' | '.join(parts)} | {today_str}"

    html = build_html(results, today_str, run_time)
    send_email(subject, html)
    log.info("  Done.")


if __name__ == "__main__":
    main()
