#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.2 - 10am Confirmation Checker (complete replacement)

CHANGES vs v5.0:
  [P1-RVOL]   Real RVOL from tvDatafeed first-45min window.
              Falls back to yfinance 1-min, then labels RVOL: N/A.
              No more hardcoded 1.0x.
  [P1-DRIFT]  Entry drift guard: stocks >2% above planned entry
              are flagged MISSED — prevents R:R-destroying chases.
  [CAPS]      Only T1 + T2 (max 8) are checked — matches orchestrator caps.
  [COSMETIC]  .NS suffix stripped from displayed ticker.
              picks_latest.json now has both ticker (clean) and ticker_raw.
  [LIB]       loguru replaces print() for structured logs.
"""

import os, json, smtplib, time
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

try:
    from loguru import logger as log
except ImportError:
    import logging
    log = logging.getLogger(__name__)

load_dotenv(override=True)

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
PICKS_JSON_PATH    = os.getenv("PICKS_JSON_PATH", "picks_latest.json")

# ── Thresholds ────────────────────────────────────────────────────────────────
RVOL_CONFIRM_MIN   = 1.5   # RVOL >= this → CONFIRMED (full size)
RVOL_LOW_VOL_MIN   = 1.0   # RVOL >= this → CONFIRMED_LOW_VOL (half size)
MAX_ENTRY_DRIFT_PCT = 2.0  # CMP more than this % above entry_price → MISSED
RVOL_45MIN_FRAC    = 0.115  # 45 min / 390 min trading day ≈ 11.5%
# ─────────────────────────────────────────────────────────────────────────────


def _load_recipients() -> list:
    path = "recipients.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return [GMAIL_ADDRESS]


# ─────────────────────────────────────────────────────────────────────────────
# RVOL  —  real first-45min volume ratio
# ─────────────────────────────────────────────────────────────────────────────

def _rvol_tvdatafeed(ticker_nse: str) -> float:
    """
    Primary RVOL source: tvDatafeed (TradingView unofficial API).
    Install: pip install --upgrade tvDatafeed
    Returns RVOL float, or -1.0 on failure / insufficient data.
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
        import pandas as pd

        symbol = ticker_nse.replace(".NS", "")
        tv     = TvDatafeed()   # anonymous — no login required for NSE
        df = tv.get_hist(
            symbol=symbol, exchange="NSE",
            interval=Interval.in_5_minute, n_bars=60
        )
        if df is None or len(df) < 9:
            return -1.0

        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_localize("Asia/Kolkata")
        except Exception:
            pass

        today = datetime.now().date()
        today_45 = df[
            (df.index.date == today) &
            (df.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df.index.time <= datetime.strptime("10:00", "%H:%M").time())
        ]
        if today_45.empty:
            return -1.0

        today_vol = float(today_45["volume"].sum())

        hist = df[df.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_45 = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
                (hist.index.time <= datetime.strptime("10:00", "%H:%M").time())
            ]
            if not day_45.empty:
                hist_by_day[row_date] = float(day_45["volume"].sum())

        if len(hist_by_day) < 5:
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        if avg_hist <= 0:
            return -1.0

        return round(today_vol / avg_hist, 2)

    except ImportError:
        return -1.0
    except Exception as e:
        log.debug(f"tvDatafeed RVOL failed for {ticker_nse}: {e}")
        return -1.0


def _rvol_yfinance_fallback(ticker_nse: str) -> float:
    """
    Fallback RVOL source: yfinance 1-min data (less reliable, but no extra dep).
    Returns RVOL float, or -1.0 on failure.
    """
    try:
        import yfinance as yf
        import pandas as pd

        t = ticker_nse if ticker_nse.endswith(".NS") else ticker_nse + ".NS"
        df_1m = yf.Ticker(t).history(period="5d", interval="1m", prepost=False)
        if df_1m.empty:
            return -1.0

        try:
            df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")
        except Exception:
            pass

        today = datetime.now().date()
        today_45 = df_1m[
            (df_1m.index.date == today) &
            (df_1m.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
            (df_1m.index.time <= datetime.strptime("10:00", "%H:%M").time())
        ]
        if today_45.empty:
            return -1.0

        today_vol = float(today_45["Volume"].sum())

        hist = df_1m[df_1m.index.date < today]
        hist_by_day = {}
        for row_date in set(hist.index.date):
            day_45 = hist[
                (hist.index.date == row_date) &
                (hist.index.time >= datetime.strptime("09:15", "%H:%M").time()) &
                (hist.index.time <= datetime.strptime("10:00", "%H:%M").time())
            ]
            if not day_45.empty:
                hist_by_day[row_date] = float(day_45["Volume"].sum())

        if len(hist_by_day) < 3:
            return -1.0

        avg_hist = sum(hist_by_day.values()) / len(hist_by_day)
        return round(today_vol / avg_hist, 2) if avg_hist > 0 else -1.0

    except Exception as e:
        log.debug(f"yfinance RVOL fallback failed for {ticker_nse}: {e}")
        return -1.0


def get_rvol(ticker: str) -> tuple:
    """
    Returns (rvol_float, source_label).
    source_label: 'tvDatafeed' | 'yfinance' | 'N/A'
    rvol = -1.0 means insufficient data (displayed as N/A).
    """
    ticker_raw = ticker if ticker.endswith(".NS") else ticker + ".NS"

    rvol = _rvol_tvdatafeed(ticker_raw)
    if rvol >= 0:
        return rvol, "tvDatafeed"

    rvol = _rvol_yfinance_fallback(ticker_raw)
    if rvol >= 0:
        return rvol, "yfinance"

    return -1.0, "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Live price (CMP)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_price(ticker: str) -> float | None:
    """Returns CMP as float, or None on failure."""
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
# Entry drift guard
# ─────────────────────────────────────────────────────────────────────────────

def check_entry_drift(cmp: float, entry_price: float) -> tuple:
    """
    Returns (is_chasing: bool, drift_pct: float).
    is_chasing = True means CMP has drifted > MAX_ENTRY_DRIFT_PCT above entry.
    """
    if entry_price <= 0:
        return False, 0.0
    drift_pct = ((cmp - entry_price) / entry_price) * 100.0
    return drift_pct > MAX_ENTRY_DRIFT_PCT, round(drift_pct, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(pick: dict, cmp: float | None, rvol: float, rvol_src: str) -> dict:
    entry  = pick["entry"]
    sl     = pick["sl"]
    pivot  = pick.get("pivot", entry)
    t1     = pick["t1"]

    if cmp is None:
        return {
            "status": "DATA_ERROR",
            "ltp": None, "gap_pct": None, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": None, "is_chasing": False,
            "action": "Could not fetch live price - check manually",
            "label":  "DATA ERROR",
            "color":  "#6b7280",
            "bg":     "#f3f4f6",
        }

    gap_pct = round(((cmp - pivot) / pivot) * 100, 1) if pivot > 0 else 0.0
    is_chasing, drift_pct = check_entry_drift(cmp, entry)
    vol_ok = (rvol >= RVOL_CONFIRM_MIN) if rvol >= 0 else True  # if no data, don't penalise

    # ── SL breached ──
    if cmp <= sl:
        return {
            "status": "BROKEN",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": False,
            "action": f"SL breached — DO NOT enter. CMP ₹{cmp:,.1f} ≤ SL ₹{sl:,.1f}",
            "label":  "SKIP — BROKEN",
            "color":  "#ffffff",
            "bg":     "#dc2626",
        }

    # ── Chasing — drifted too far above entry ──
    if is_chasing:
        return {
            "status": "MISSED",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": True,
            "action": (
                f"Price drifted {drift_pct:+.1f}% above planned entry ₹{entry:,.1f} "
                f"— R:R destroyed, DO NOT chase"
            ),
            "label":  f"MISSED +{drift_pct:.1f}%",
            "color":  "#ffffff",
            "bg":     "#7c3aed",
        }

    # ── Above pivot, vol ok ──
    if cmp >= pivot and vol_ok:
        return {
            "status": "CONFIRMED",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": False,
            "action": (
                f"Enter — CMP ₹{cmp:,.1f} | Entry ₹{entry:,.1f} | "
                f"SL ₹{sl:,.1f} | T1 ₹{t1:,.1f}"
            ),
            "label":  "CONFIRMED — ENTER",
            "color":  "#ffffff",
            "bg":     "#16a34a",
        }

    # ── Above pivot, low volume ──
    if cmp >= pivot and not vol_ok:
        rvol_disp = f"{rvol:.1f}x" if rvol >= 0 else "N/A"
        return {
            "status": "CONFIRMED_LOW_VOL",
            "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
            "drift_pct": drift_pct, "is_chasing": False,
            "action": (
                f"Above pivot but low volume (RVOL {rvol_disp}) — "
                f"enter half size or wait for volume confirmation"
            ),
            "label":  f"LOW VOLUME ({rvol_disp})",
            "color":  "#ffffff",
            "bg":     "#ca8a04",
        }

    # ── Below pivot ──
    dist     = pivot - cmp
    dist_pct = round((dist / pivot) * 100, 1) if pivot > 0 else 0.0
    return {
        "status": "PENDING",
        "ltp": cmp, "gap_pct": gap_pct, "rvol": rvol, "rvol_src": rvol_src,
        "drift_pct": drift_pct, "is_chasing": False,
        "action": (
            f"Below pivot by ₹{dist:,.1f} ({dist_pct:.1f}%) — "
            f"set alert at ₹{pivot:,.1f}"
        ),
        "label":  "PENDING",
        "color":  "#1e40af",
        "bg":     "#dbeafe",
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
        f"<span style='color:{color};font-weight:700;'>"
        f"{icon} {rvol:.1f}x</span>"
        f"<br><span style='font-size:10px;color:#9ca3af;'>{src}</span>"
    )


def _row(r: dict) -> str:
    p   = r["pick"]
    c   = r["classification"]
    # [COSMETIC] use clean ticker (already stripped in picks_latest.json)
    ticker_disp = p.get("ticker", "").replace(".NS", "")
    ltp   = f"₹{c['ltp']:,.1f}"    if c["ltp"]       is not None else "N/A"
    gap   = f"{c['gap_pct']:+.1f}%" if c.get("gap_pct") is not None else "N/A"
    drift_cell = ""
    if c.get("is_chasing"):
        drift_cell = (
            f"<span style='color:#7c3aed;font-weight:700;font-size:10px;'>"
            f"+{c['drift_pct']:.1f}% above entry</span>"
        )
    elif c.get("drift_pct") is not None:
        drift_cell = (
            f"<span style='color:#9ca3af;font-size:10px;'>"
            f"{c['drift_pct']:+.1f}% vs entry</span>"
        )
    tier = f"T{p.get('tier','?')}"

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
      <td style="padding:10px 8px;font-size:11px;color:#6b7280;">{drift_cell}</td>
      <td style="padding:10px 8px;font-size:12px;color:#374151;">{c['action']}</td>
    </tr>"""


def _section(items: list, label: str, accent: str) -> str:
    if not items:
        return ""
    header = f"""
    <tr>
      <td colspan="8" style="padding:14px 8px 6px;font-size:11px;font-weight:700;
          letter-spacing:1.5px;text-transform:uppercase;color:{accent};
          border-bottom:2px solid {accent};">{label}</td>
    </tr>"""
    return header + "".join(_row(r) for r in items)


def build_html(results: list, scan_date: str, run_time: str) -> str:
    order = {
        "CONFIRMED": 0, "CONFIRMED_LOW_VOL": 1,
        "PENDING": 2, "MISSED": 3, "BROKEN": 4, "DATA_ERROR": 5
    }
    results = sorted(results,
                     key=lambda r: order.get(r["classification"]["status"], 9))

    t1 = [r for r in results if r["pick"].get("tier") == 1]
    t2 = [r for r in results if r["pick"].get("tier") == 2]

    confirmed_n = sum(1 for r in results
                      if r["classification"]["status"].startswith("CONFIRMED"))
    pending_n   = sum(1 for r in results
                      if r["classification"]["status"] == "PENDING")
    broken_n    = sum(1 for r in results
                      if r["classification"]["status"] == "BROKEN")
    missed_n    = sum(1 for r in results
                      if r["classification"]["status"] == "MISSED")
    error_n     = sum(1 for r in results
                      if r["classification"]["status"] == "DATA_ERROR")

    summary_parts = []
    if confirmed_n:
        summary_parts.append(
            f"<span style='color:#16a34a;font-weight:700;'>{confirmed_n} CONFIRMED</span>")
    if pending_n:
        summary_parts.append(
            f"<span style='color:#1d4ed8;font-weight:700;'>{pending_n} PENDING</span>")
    if missed_n:
        summary_parts.append(
            f"<span style='color:#7c3aed;font-weight:700;'>{missed_n} MISSED (chased)</span>")
    if broken_n:
        summary_parts.append(
            f"<span style='color:#dc2626;font-weight:700;'>{broken_n} BROKEN</span>")
    if error_n:
        summary_parts.append(
            f"<span style='color:#9ca3af;font-weight:700;'>{error_n} DATA ERROR</span>")
    summary_html = " &nbsp;|&nbsp; ".join(summary_parts)

    action_box = ""
    if confirmed_n:
        action_box = f"""
        <div style="background:#f0fdf4;border-left:4px solid #16a34a;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#15803d;font-size:15px;">
            ACTION REQUIRED — {confirmed_n} setup(s) confirmed for entry today
          </div>
          <div style="color:#166534;font-size:13px;margin-top:4px;">
            Place orders now. Standard Regime B position sizing.
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
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
<body style="margin:0;padding:0;background:#f9fafb;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
<div style="max-width:980px;margin:24px auto;background:#ffffff;
    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">

  <div style="background:#0f172a;padding:20px 28px;">
    <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;">
        NSE MOMENTUM DISCOVERY &ndash; V5.2</div>
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
        {len(results)} picks checked (RVOL via tvDatafeed/yfinance first-45min)</span>
  </div>

  {action_box}

  <div style="padding:8px 28px 28px;">
    <table style="width:100%;border-collapse:collapse;margin-top:8px;">
      <thead>
        <tr style="background:#f1f5f9;">
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">Ticker</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">Sector</th>
          <th style="padding:10px 8px;text-align:center;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">Status</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">CMP</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">RVOL</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">vs Pivot</th>
          <th style="padding:10px 8px;text-align:right;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">vs Entry</th>
          <th style="padding:10px 8px;text-align:left;font-size:11px;color:#6b7280;
              letter-spacing:1px;text-transform:uppercase;">Action</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px;
      font-size:11px;color:#94a3b8;">
    Not SEBI-registered investment advice. All trading involves capital risk.
    SL = hard stop, do not widen. RVOL = first-45min volume vs 20-day avg.
    MISSED = price drifted &gt;2% above planned entry — R:R compromised.
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
    today_str = date.today().strftime("%d %b %Y")
    run_time  = datetime.now().strftime("%H:%M")
    log.info(f"NSE Momentum v5.2 — 10am Confirmation starting ({run_time} IST)...")

    if not os.path.exists(PICKS_JSON_PATH):
        log.error(f"{PICKS_JSON_PATH} not found — evening scan must run first")
        raise FileNotFoundError(PICKS_JSON_PATH)

    with open(PICKS_JSON_PATH, encoding="utf-8") as f:
        picks = json.load(f)
    log.info(f"  Loaded {len(picks)} picks from {PICKS_JSON_PATH}")

    results = []
    for pick in picks:
        # Use ticker_raw (.NS suffix) for price/RVOL lookups
        ticker_raw = pick.get("ticker_raw") or pick.get("ticker", "")
        if not ticker_raw.endswith(".NS"):
            ticker_raw += ".NS"

        log.info(f"  Checking {pick.get('ticker', ticker_raw)}...")

        cmp          = get_live_price(ticker_raw)
        rvol, rvol_src = get_rvol(ticker_raw)

        c = classify(pick, cmp, rvol, rvol_src)

        log.info(
            f"    → {c['status']:22s}  "
            f"CMP={f'₹{cmp:,.1f}' if cmp else 'N/A':>10}  "
            f"RVOL={f'{rvol:.1f}x ({rvol_src})' if rvol >= 0 else 'N/A':>18}  "
            f"{c['action'][:60]}"
        )
        results.append({"pick": pick, "classification": c})

        # Small delay to avoid rate limiting on yfinance
        time.sleep(0.3)

    confirmed_n = sum(1 for r in results
                      if r["classification"]["status"].startswith("CONFIRMED"))
    broken_n    = sum(1 for r in results
                      if r["classification"]["status"] == "BROKEN")
    missed_n    = sum(1 for r in results
                      if r["classification"]["status"] == "MISSED")

    if confirmed_n > 0:
        subject = (
            f"[NSE Momentum 10am] {confirmed_n} CONFIRMED for entry | {today_str}"
        )
    elif missed_n > 0 and not confirmed_n:
        subject = (
            f"[NSE Momentum 10am] {missed_n} MISSED (chased) — no entry | {today_str}"
        )
    elif broken_n == len(results):
        subject = (
            f"[NSE Momentum 10am] All setups broken — skip today | {today_str}"
        )
    else:
        subject = (
            f"[NSE Momentum 10am] Setups pending — no action yet | {today_str}"
        )

    html = build_html(results, today_str, run_time)
    send_email(subject, html)
    log.info("  Done.")


if __name__ == "__main__":
    main()
