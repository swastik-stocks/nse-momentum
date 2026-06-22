#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.0 - 10am Confirmation Checker
Loads picks_latest.json saved by evening scan,
fetches live prices from Dhan, classifies each as
CONFIRMED / CONFIRMED_LOW_VOL / PENDING / BROKEN,
then sends a crisp HTML email.
"""

import os, json, smtplib, urllib.request
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv(override=True)

# ENV VARS - matches your project exactly
DHAN_ACCESS_TOKEN  = os.getenv("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID     = os.getenv("DHAN_CLIENT_ID", "")
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
PICKS_JSON_PATH    = os.getenv("PICKS_JSON_PATH", "picks_latest.json")

def _load_recipients() -> list:
    path = "recipients.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return [GMAIL_ADDRESS]

# DHAN LIVE QUOTE
DHAN_QUOTE_URL = "https://api.dhan.co/v2/marketfeed/quote"

def get_live_quote(ticker: str) -> dict | None:
    try:
        import yfinance as yf
        t = ticker if ticker.endswith(".NS") else ticker + ".NS"
        fast = yf.Ticker(t).fast_info
        ltp  = float(fast.last_price)
        vol  = float(fast.three_month_average_volume or 1)
        return {"ltp": ltp, "volume": vol, "avg_volume": vol}
    except Exception as e:
        print(f"  [WARN] yfinance quote failed for {ticker}: {e}")
        return None

# CLASSIFICATION
def classify(pick: dict, quote: dict | None) -> dict:
    entry = pick["entry"]
    sl    = pick["sl"]
    pivot = pick.get("pivot", entry)
    t1    = pick["t1"]

    if quote is None:
        return {
            "status": "DATA_ERROR",
            "ltp": None, "gap_pct": None, "rvol": None,
            "action": "Could not fetch live price - check manually",
            "label":  "DATA ERROR",
            "color":  "#6b7280",
            "bg":     "#f3f4f6",
        }

    ltp     = quote["ltp"]
    gap_pct = ((ltp - pivot) / pivot) * 100
    rvol    = None
    vol_ok  = True

    if quote.get("avg_volume", 0) > 0:
        rvol   = round(quote["volume"] / quote["avg_volume"], 1)
        vol_ok = rvol >= 1.0

    if ltp <= sl:
        return {
            "status": "BROKEN",
            "ltp": ltp, "gap_pct": gap_pct, "rvol": rvol,
            "action": f"SL breached - DO NOT enter. CMP Rs.{ltp:,.1f} is at or below SL Rs.{sl:,.1f}",
            "label":  "SKIP - BROKEN",
            "color":  "#ffffff",
            "bg":     "#dc2626",
        }
    elif ltp >= pivot and vol_ok:
        return {
            "status": "CONFIRMED",
            "ltp": ltp, "gap_pct": gap_pct, "rvol": rvol,
            "action": f"Enter now - CMP Rs.{ltp:,.1f} | Entry Rs.{entry:,.1f} | SL Rs.{sl:,.1f} | T1 Rs.{t1:,.1f}",
            "label":  "CONFIRMED - ENTER",
            "color":  "#ffffff",
            "bg":     "#16a34a",
        }
    elif ltp >= pivot and not vol_ok:
        return {
            "status": "CONFIRMED_LOW_VOL",
            "ltp": ltp, "gap_pct": gap_pct, "rvol": rvol,
            "action": f"Above pivot but low volume (RVOL {rvol}x) - enter half size or wait for volume",
            "label":  "LOW VOLUME",
            "color":  "#ffffff",
            "bg":     "#ca8a04",
        }
    else:
        dist     = pivot - ltp
        dist_pct = (dist / pivot) * 100
        return {
            "status": "PENDING",
            "ltp": ltp, "gap_pct": gap_pct, "rvol": rvol,
            "action": f"Below pivot by Rs.{dist:,.1f} ({dist_pct:.1f}%) - set alert at Rs.{pivot:,.1f}",
            "label":  "PENDING",
            "color":  "#1e40af",
            "bg":     "#dbeafe",
        }

# EMAIL HTML
def _row(r: dict) -> str:
    p   = r["pick"]
    c   = r["classification"]
    ltp  = f"Rs.{c['ltp']:,.1f}"   if c["ltp"]  is not None else "N/A"
    rvol = f"{c['rvol']}x"         if c.get("rvol") is not None else "N/A"
    gap  = f"{c['gap_pct']:+.1f}%" if c.get("gap_pct") is not None else "N/A"
    tier = f"T{p.get('tier','?')}"
    return f"""
    <tr style="border-bottom:1px solid #e5e7eb;">
      <td style="padding:10px 8px;">
        <span style="font-weight:700;font-size:14px;color:#111827;">{p['ticker']}</span><br>
        <span style="font-size:11px;color:#9ca3af;">{tier} &middot; Score {p.get('score','')}</span>
      </td>
      <td style="padding:10px 8px;font-size:12px;color:#6b7280;">{p.get('sector','')}</td>
      <td style="padding:10px 8px;text-align:center;">
        <span style="background:{c['bg']};color:{c['color']};padding:4px 10px;border-radius:4px;font-size:11px;font-weight:700;white-space:nowrap;">{c['label']}</span>
      </td>
      <td style="padding:10px 8px;text-align:right;font-weight:700;color:#111827;">{ltp}</td>
      <td style="padding:10px 8px;text-align:right;color:#6b7280;">{rvol}</td>
      <td style="padding:10px 8px;text-align:right;color:#6b7280;">{gap}</td>
      <td style="padding:10px 8px;font-size:12px;color:#374151;">{c['action']}</td>
    </tr>"""

def _section(items: list, label: str, accent: str) -> str:
    if not items:
        return ""
    header = f"""
    <tr>
      <td colspan="7" style="padding:14px 8px 6px;font-size:11px;font-weight:700;
          letter-spacing:1.5px;text-transform:uppercase;color:{accent};
          border-bottom:2px solid {accent};">{label}</td>
    </tr>"""
    return header + "".join(_row(r) for r in items)

def build_html(results: list, scan_date: str, run_time: str) -> str:
    order = {"CONFIRMED": 0, "CONFIRMED_LOW_VOL": 1, "PENDING": 2, "BROKEN": 3, "DATA_ERROR": 4}
    results = sorted(results, key=lambda r: order.get(r["classification"]["status"], 5))

    t1 = [r for r in results if r["pick"].get("tier") == 1]
    t2 = [r for r in results if r["pick"].get("tier") == 2]

    confirmed_n = sum(1 for r in results if r["classification"]["status"].startswith("CONFIRMED"))
    pending_n   = sum(1 for r in results if r["classification"]["status"] == "PENDING")
    broken_n    = sum(1 for r in results if r["classification"]["status"] == "BROKEN")
    error_n     = sum(1 for r in results if r["classification"]["status"] == "DATA_ERROR")

    summary_parts = []
    if confirmed_n: summary_parts.append(f"<span style='color:#16a34a;font-weight:700;'>{confirmed_n} CONFIRMED</span>")
    if pending_n:   summary_parts.append(f"<span style='color:#1d4ed8;font-weight:700;'>{pending_n} PENDING</span>")
    if broken_n:    summary_parts.append(f"<span style='color:#dc2626;font-weight:700;'>{broken_n} BROKEN</span>")
    if error_n:     summary_parts.append(f"<span style='color:#9ca3af;font-weight:700;'>{error_n} DATA ERROR</span>")
    summary_html = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(summary_parts)

    action_box = ""
    if confirmed_n:
        action_box = f"""
        <div style="background:#f0fdf4;border-left:4px solid #16a34a;
            margin:16px 28px 0;padding:14px 16px;border-radius:0 4px 4px 0;">
          <div style="font-weight:700;color:#15803d;font-size:15px;">
            ACTION REQUIRED &mdash; {confirmed_n} setup(s) confirmed for entry today
          </div>
          <div style="color:#166534;font-size:13px;margin-top:4px;">
            Place orders now. Standard Regime B position sizing.
          </div>
        </div>"""

    rows_html = _section(t1, "Tier 1 &mdash; Top Picks", "#0f172a") + \
                _section(t2, "Tier 2 &mdash; Aggressive", "#92400e")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
<body style="margin:0;padding:0;background:#f9fafb;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
<div style="max-width:920px;margin:24px auto;background:#ffffff;
    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden;">

  <div style="background:#0f172a;padding:20px 28px;">
    <div style="color:#94a3b8;font-size:11px;letter-spacing:2px;">
        NSE MOMENTUM DISCOVERY &ndash; V5.0</div>
    <div style="color:#f1f5f9;font-size:22px;font-weight:700;margin-top:4px;">
        10am Confirmation Report</div>
    <div style="color:#64748b;font-size:13px;margin-top:2px;">
        {scan_date} &nbsp;&middot;&nbsp; Run at {run_time} IST</div>
  </div>

  <div style="background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:14px 28px;font-size:14px;">
    {summary_html}
    &nbsp;&nbsp;
    <span style="color:#94a3b8;font-size:12px;">Checking {len(results)} picks from yesterday</span>
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
    SL = hard stop, do not widen.
  </div>
</div>
</body></html>"""

# EMAIL SENDER
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
    print(f"  [OK] Email sent to {recipients}")

# MAIN
def main():
    today_str = date.today().strftime("%d %b %Y")
    run_time  = datetime.now().strftime("%H:%M")
    print(f"[{run_time}] NSE Momentum 10am Confirmation starting...")

    if not os.path.exists(PICKS_JSON_PATH):
        print(f"  [ERROR] {PICKS_JSON_PATH} not found")
        raise FileNotFoundError(PICKS_JSON_PATH)

    with open(PICKS_JSON_PATH, encoding="utf-8") as f:
        picks = json.load(f)
    print(f"  Loaded {len(picks)} picks")

    results = []
    for pick in picks:
        ticker = pick["ticker"]
        sec_id = pick.get("security_id", "")
        seg    = pick.get("segment", "NSE_EQ")
        print(f"  Checking {ticker}...")
        quote  = get_live_quote(ticker)
        c      = classify(pick, quote)
        print(f"    -> {c['status']:20s}  CMP={c['ltp']}  {c['action'][:60]}")
        results.append({"pick": pick, "classification": c})

    confirmed_n = sum(1 for r in results if r["classification"]["status"].startswith("CONFIRMED"))
    broken_n    = sum(1 for r in results if r["classification"]["status"] == "BROKEN")

    if confirmed_n > 0:
        subject = f"[NSE Momentum 10am] {confirmed_n} CONFIRMED for entry | {today_str}"
    elif broken_n == len(results):
        subject = f"[NSE Momentum 10am] All setups broken - skip today | {today_str}"
    else:
        subject = f"[NSE Momentum 10am] Setups pending - no action yet | {today_str}"

    html = build_html(results, today_str, run_time)
    send_email(subject, html)
    print("  Done.")

if __name__ == "__main__":
    main()
