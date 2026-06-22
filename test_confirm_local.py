#!/usr/bin/env python3
"""
LOCAL TEST — runs confirm_picks.py logic with MOCK prices (no Dhan API needed).
Simulates: 2 confirmed, 3 pending, 1 broken, 1 confirmed low-vol, 1 data error.

Usage:
    python test_confirm_local.py

Set EMAIL_FROM / EMAIL_PASSWORD / EMAIL_TO to actually send the email,
or leave unset to just print the HTML to console.
"""

import os
import sys
import json

# ── Inject mock prices before importing confirm_picks ────────────────────────
MOCK_PRICES = {
    "CGPOWER":   {"ltp": 963.5,   "volume": 120000, "avg_volume": 95000},   # ✅ CONFIRMED
    "KPIL":      {"ltp": 1355.0,  "volume": 45000,  "avg_volume": 50000},   # ⏳ PENDING (below pivot 1391)
    "TIINDIA":   {"ltp": 3090.0,  "volume": 30000,  "avg_volume": 40000},   # ❌ BROKEN (below SL 3107)
    "NAM-INDIA": {"ltp": 1195.0,  "volume": 28000,  "avg_volume": 55000},   # ✅ CONFIRMED but low vol
    "ANGELONE":  {"ltp": 345.0,   "volume": 0,      "avg_volume": 0},       # ⏳ PENDING (below pivot 352.8)
    "ADANIENT":  {"ltp": 3095.0,  "volume": 180000, "avg_volume": 110000},  # ✅ CONFIRMED
    "SONACOMS":  {"ltp": 610.0,   "volume": 0,      "avg_volume": 0},       # ⏳ PENDING
    "HONAUT":    None,                                                        # ⚠️ DATA ERROR
}

# Patch get_dhan_quote before importing
import confirm_picks as cp

def mock_get_dhan_quote(security_id: str, exchange_segment: str = "NSE_EQ"):
    # Map by position since we don't have security_id→ticker mapping in mock
    return None   # overridden per-pick below

cp.get_dhan_quote = mock_get_dhan_quote   # not used; we override in main below

# ── Load example picks ───────────────────────────────────────────────────────
example_path = os.path.join(os.path.dirname(__file__), "picks_latest.example.json")
with open(example_path) as f:
    picks = json.load(f)

# ── Classify with mock prices ────────────────────────────────────────────────
results = []
for pick in picks:
    ticker = pick["ticker"]
    quote  = MOCK_PRICES.get(ticker)
    classification = cp.classify(pick, quote)
    print(f"  {ticker:12s}  CMP={classification['ltp']}  → {classification['status']:20s}  {classification['action']}")
    results.append({"pick": pick, "classification": classification})

# ── Build HTML ───────────────────────────────────────────────────────────────
html = cp.build_html(results, "22 Jun 2026 (TEST)", "10:00")

# Write to file for inspection
out_path = "confirm_test_output.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n  HTML written to {out_path}  — open in browser to preview")

# ── Optionally send email ────────────────────────────────────────────────────
if os.environ.get("EMAIL_FROM") and os.environ.get("EMAIL_PASSWORD"):
    confirmed_count = sum(1 for r in results if r["classification"]["status"].startswith("CONFIRMED"))
    subject = f"[TEST] ✅ {confirmed_count} CONFIRMED | NSE Momentum 10am | 22 Jun 2026"
    cp.send_email(subject, html)
    print(f"  Test email sent to {os.environ.get('EMAIL_TO', os.environ.get('EMAIL_FROM'))}")
else:
    print("  (Set EMAIL_FROM + EMAIL_PASSWORD env vars to also send a test email)")
