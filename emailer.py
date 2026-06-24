"""
NSE Momentum v5.3 - Email Reporter
4-section HTML email:
  Section 1: T1/T2/T3 evidence-based trade cards
  Section 2: Top 20 watchlist table (no T1 duplicates)
  Section 3: Market intelligence (regime, breadth, macro, event)
  Section 4: Near-breakout watchlist (set alerts, do not buy yet)
"""

import os, smtplib, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PW  = os.getenv("GMAIL_APP_PASSWORD")
BASE_DIR      = Path(__file__).parent

REGIME_META = {
    "A": ("STRONG BULL",  "#00E676", "rgba(0,230,118,0.12)", "rgba(0,230,118,0.35)",
          "All conditions optimal. Highest probability entry window."),
    "B": ("BULL",         "#00D4AA", "rgba(0,212,170,0.10)", "rgba(0,212,170,0.3)",
          "Good conditions. Standard position sizing appropriate."),
    "C": ("RANGE BOUND",  "#FFB300", "rgba(255,179,0,0.10)", "rgba(255,179,0,0.3)",
          "Choppy market. Tighter stops, smaller sizes, wait for clear breakout."),
    "D": ("CORRECTION",   "#FF8C00", "rgba(255,140,0,0.10)", "rgba(255,140,0,0.3)",
          "Market pulling back. Higher failure rate on breakouts. Watchlist mode."),
    "E": ("BEAR MARKET",  "#FF5252", "rgba(255,82,82,0.10)", "rgba(255,82,82,0.3)",
          "Avoid new longs. Capital preservation is the priority."),
}

MACRO_COLOR = {"SUPPORTIVE": "#00E676", "MIXED": "#FFB300", "HOSTILE": "#FF5252"}
EVENT_COLOR = {"NORMAL": "#5E7A96",     "WATCH": "#FFB300",  "HIGH_RISK": "#FF5252"}
BQ_COLOR    = {"MAJOR": "#00E676",      "MINOR": "#FFB300",  "RECOVERY": "#4D9EFF"}


def _load_recipients() -> list:
    path = BASE_DIR / "recipients.txt"
    if not path.exists():
        return [GMAIL_ADDRESS] if GMAIL_ADDRESS else []
    emails = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "@" in line:
            emails.append(line)
    return emails or ([GMAIL_ADDRESS] if GMAIL_ADDRESS else [])


def send_email_report(tiers: dict):
    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        log.warning("Gmail credentials missing.")
        return

    recipients = _load_recipients()
    if not recipients:
        log.warning("No recipients in recipients.txt")
        return

    t1           = tiers.get("tier1", [])
    t2           = tiers.get("tier2", [])
    t3           = tiers.get("tier3", [])
    all_r        = tiers.get("all_results", [])
    near_bo      = tiers.get("near_breakout", [])
    regime       = tiers.get("regime", "C")
    brdth        = tiers.get("breadth", 5)
    bd           = tiers.get("breadth_detail", {})
    macro_state  = tiers.get("macro_state", "MIXED")
    event_risk   = tiers.get("event_risk", "NORMAL")
    t1_cap       = tiers.get("t1_cap", 15)

    rlbl, rcol, rbg, rborder, rnote = REGIME_META.get(regime, REGIME_META["C"])
    date_str = datetime.today().strftime("%d %b %Y")
    penalty  = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -25}.get(regime, 0)

    html = _build_html(t1, t2, t3, all_r, near_bo,
                       regime, rlbl, rcol, rbg, rborder, rnote,
                       brdth, bd, date_str, penalty,
                       macro_state, event_risk, t1_cap)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = (f"NSE Momentum v5.3 - {date_str} - "
                      f"Regime {regime} ({rlbl}) - {len(t1)} picks")
    msg["From"] = GMAIL_ADDRESS
    msg["To"]   = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            smtp.sendmail(GMAIL_ADDRESS, recipients, msg.as_string())
        log.info(f"Email sent to {len(recipients)} recipient(s)")
    except Exception as e:
        log.error(f"Email send failed: {e}")
        raise


def _tier_card(r, tier_label: str, tier_color: str) -> str:
    # MAJOR/MINOR/RECOVERY label
    bq      = getattr(r, "breakout_quality", "") or "MINOR"
    bq_col  = BQ_COLOR.get(bq, "#FFB300")
    bq_html = (f'<span style="font-size:9px;color:{bq_col};'
               f'border:1px solid {bq_col};border-radius:3px;'
               f'padding:1px 5px;margin-left:6px">{bq}</span>')

    # Confirmation state
    conf       = getattr(r, "confirmation_state", "SETUP_READY")
    conf_col   = "#00E676" if conf == "BREAKOUT_CONFIRMED" else "#FFB300"
    conf_label = "CONFIRMED" if conf == "BREAKOUT_CONFIRMED" else "SETUP READY"

    # Asymmetry
    risk_pct   = getattr(r, "asymmetry_risk_pct",   r.stop_pct)
    reward_pct = getattr(r, "asymmetry_reward_pct", r.gain_pct_t1)
    rr_actual  = r.rrr   # consistent with table display

    working_html = "".join(
        f'<li style="margin:3px 0;color:#9AAFC4">OK {w}</li>'
        for w in (r.what_is_working or [])[:3]
    )
    missing_html = "".join(
        f'<li style="margin:3px 0;color:#FFB300">! {m}</li>'
        for m in (r.what_is_missing or [])[:2]
    )
    trigger_html = "".join(
        f'<li style="margin:3px 0;color:#4D9EFF">- {t}</li>'
        for t in (r.trigger_conditions or [])[:2]
    )
    risk_html = "".join(
        f'<li style="margin:3px 0;color:#FF8C00">! {rk}</li>'
        for rk in (r.risk_factors or [])[:2]
    )

    return f"""
<div style="background:#0E1622;border:1px solid #1F3046;border-left:3px solid {tier_color};
            border-radius:8px;padding:16px 18px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <span style="font-family:monospace;font-size:10px;color:{tier_color};
                   letter-spacing:0.15em;text-transform:uppercase">{tier_label}</span>
      <div style="font-size:17px;font-weight:700;color:#FFFFFF;margin:3px 0">
        {r.ticker.replace('.NS','')} {bq_html}
      </div>
      <div style="font-size:11px;color:#5E7A96">{r.name} - {r.sector} - {r.universe}</div>
      <div style="margin-top:4px">
        <span style="font-size:9px;color:{conf_col};border:1px solid {conf_col};
                     border-radius:3px;padding:1px 5px">{conf_label}</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-family:monospace;font-size:22px;font-weight:700;color:{tier_color}">{r.total_score}</div>
      <div style="font-family:monospace;font-size:9px;color:#5E7A96">/ 100 pts</div>
      <div style="font-size:10px;color:#9AAFC4;margin-top:2px">Conf: {r.confidence_pct:.0f}%</div>
    </div>
  </div>

  <div style="background:#070B11;border-radius:6px;padding:10px 14px;margin-bottom:10px;
              font-family:monospace;font-size:11px;">
    <div style="display:flex;gap:16px;flex-wrap:wrap;color:#9AAFC4">
      <span>{r.pattern}</span>
      <span>RS {r.rs_percentile:.0f}th%</span>
      <span>RVOL {r.rvol:.1f}x</span>
      <span>RSI {r.rsi_val:.0f}</span>
      <span>Del {r.del_pct:.0f}%</span>
    </div>
    <div style="margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px">
      <span style="color:#00D4AA">Entry Rs.{r.entry:.1f}</span>
      <span style="color:#FF5252">SL Rs.{r.stop_loss:.1f} ({risk_pct:.1f}% risk)</span>
      <span style="color:#F0B429">T1 Rs.{r.target1:.1f} (+{reward_pct:.1f}%)</span>
      <span style="color:#BB86FC">T2 Rs.{r.target2:.1f}</span>
      <span style="color:#00E676">R:R {rr_actual:.1f}x</span>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px">
    <div>
      <div style="color:#5E7A96;font-size:9px;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:4px">What is working</div>
      <ul style="margin:0;padding:0 0 0 14px">{working_html}</ul>
    </div>
    <div>
      {"" if not missing_html else f'<div style="color:#5E7A96;font-size:9px;text-transform:uppercase;margin-bottom:4px">Missing</div><ul style="margin:0;padding:0 0 0 14px">{missing_html}</ul>'}
      {"" if not trigger_html else f'<div style="color:#5E7A96;font-size:9px;text-transform:uppercase;margin-bottom:4px;margin-top:6px">Trigger to act</div><ul style="margin:0;padding:0 0 0 14px">{trigger_html}</ul>'}
      {"" if not risk_html else f'<div style="color:#5E7A96;font-size:9px;text-transform:uppercase;margin-bottom:4px;margin-top:6px">Risk factors</div><ul style="margin:0;padding:0 0 0 14px">{risk_html}</ul>'}
    </div>
  </div>
</div>"""


def _near_breakout_section(near_bo: list) -> str:
    if not near_bo:
        return ""
    rows = ""
    for nb in near_bo:
        rows += f"""
<tr style="border-bottom:1px solid #1F3046">
  <td style="padding:7px 10px;font-weight:600;color:#E8F0F8;font-family:monospace">
    {nb['ticker'].replace('.NS','')}
  </td>
  <td style="padding:7px 10px;color:#9AAFC4;font-size:11px">{nb['name']}</td>
  <td style="padding:7px 10px;color:#9AAFC4;font-size:11px">{nb['pattern']}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#00D4AA">Rs.{nb['price']:.1f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#F0B429">Rs.{nb['breakout']:.1f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#FFB300">{nb['gap_pct']:.1f}% away</td>
  <td style="padding:7px 10px;font-family:monospace;color:#9AAFC4">{nb.get('rsi',0):.0f}</td>
  <td style="padding:7px 10px;font-size:10px;color:#5E7A96">{nb['universe']}</td>
</tr>"""

    return f"""
  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#5E7A96;
              text-transform:uppercase;margin:24px 0 10px">
    Section 4 - Near-Breakout Watchlist (set alerts, do not buy yet)
  </div>
  <div style="background:#0A1018;border:1px solid #1F3046;border-radius:8px;
              padding:10px 14px;margin-bottom:12px;font-size:11px;color:#9AAFC4">
    These stocks are within 3% of their breakout level with valid patterns forming.
    They have NOT triggered yet. Set an alert at the breakout level. Buy only on a
    confirmed close above breakout on volume >= 1.5x average.
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead>
      <tr style="background:#0A1018;border-bottom:1px solid #1F3046">
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">TICKER</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">NAME</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">PATTERN</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">CMP</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">BREAKOUT</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">GAP</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">RSI</th>
        <th style="padding:7px 10px;text-align:left;color:#5E7A96;font-size:9px">UNI</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>"""


def _build_html(t1, t2, t3, all_r, near_bo,
                regime, rlbl, rcol, rbg, rborder, rnote,
                breadth, bd, date_str, penalty,
                macro_state, event_risk, t1_cap) -> str:

    mcol = MACRO_COLOR.get(macro_state, "#FFB300")
    ecol = EVENT_COLOR.get(event_risk,  "#5E7A96")

    # Section 1: trade cards
    tier_cards = ""
    if t1:
        for r in t1[:8]:
            tier_cards += _tier_card(r, "TIER 1 - TOP PICK", "#00E676")
    else:
        tier_cards += f"""
<div style="background:#070B11;border:1px solid #1F3046;border-radius:8px;
            padding:20px;text-align:center;color:#5E7A96;margin-bottom:12px">
  No stocks cleared the Tier 1 gate today. Regime {regime} penalty ({penalty} pts).<br>
  <span style="color:#FFB300">Stay in cash. Capital preservation is the priority.</span>
</div>"""

    if t2:
        for r in t2[:3]:
            tier_cards += _tier_card(r, "TIER 2 - AGGRESSIVE", "#F0B429")
    if t3:
        for r in t3[:2]:
            tier_cards += _tier_card(r, "TIER 3 - WATCHLIST", "#4D9EFF")

    # Section 2: watchlist table — EXCLUDE T1 tickers to avoid duplicates
    t1_tickers = {r.ticker for r in t1}
    table_stocks = [r for r in all_r if r.ticker not in t1_tickers][:20]
    top20_rows = ""
    for i, r in enumerate(table_stocks, 1):
        bg      = "#0A1018" if i % 2 == 0 else "transparent"
        tcol    = {"1": "#00E676", "2": "#F0B429", "3": "#4D9EFF"}.get(str(r.tier), "#5E7A96")
        bq      = getattr(r, "breakout_quality", "MINOR") or "MINOR"
        bq_col  = BQ_COLOR.get(bq, "#FFB300")
        top20_rows += f"""
<tr style="background:{bg}">
  <td style="padding:6px 10px;font-family:monospace;font-size:10px;color:{tcol}">T{r.tier}</td>
  <td style="padding:6px 10px;font-weight:600;color:#E8F0F8">{r.ticker.replace('.NS','')}</td>
  <td style="padding:6px 10px;color:#9AAFC4;font-size:11px">
    {r.pattern}
    <span style="color:{bq_col};font-size:9px;margin-left:4px">[{bq}]</span>
  </td>
  <td style="padding:6px 10px;font-family:monospace;color:{tcol}">{r.total_score}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#9AAFC4">{r.rs_percentile:.0f}%</td>
  <td style="padding:6px 10px;font-family:monospace;color:#FFB300">{r.rvol:.1f}x</td>
  <td style="padding:6px 10px;font-family:monospace;color:#00D4AA">Rs.{r.entry:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#FF5252">Rs.{r.stop_loss:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#F0B429">Rs.{r.target1:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#00E676">{r.rrr:.1f}x</td>
  <td style="padding:6px 10px;font-size:10px;color:#5E7A96">{r.universe}</td>
</tr>"""

    ad   = bd.get("ad_ratio", "-")
    ab50 = bd.get("above_50_pct", "-")
    nh   = bd.get("new_highs", "-")
    nl   = bd.get("new_lows", "-")
    bbar = int(breadth / 10 * 100)
    bcol = "#00E676" if breadth >= 7 else "#FFB300" if breadth >= 4 else "#FF5252"

    near_section = _near_breakout_section(near_bo)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:20px;background:#070B11;font-family:'Segoe UI',Arial,sans-serif">
<div style="max-width:700px;margin:0 auto">

<div style="background:#0A1422;padding:22px 28px 18px;border-radius:12px 12px 0 0;
            border:1px solid #1F3046;border-bottom:none">
  <div style="font-family:monospace;font-size:10px;letter-spacing:0.2em;color:#00D4AA;
              text-transform:uppercase;margin-bottom:6px">
    NSE Momentum Discovery - v5.3 - {date_str}
  </div>
  <div style="font-size:22px;font-weight:800;color:#FFFFFF;margin-bottom:4px">Daily Intelligence Report</div>
  <div style="font-size:11px;color:#5E7A96;margin-bottom:12px">
    404 stocks - 3 universes - 14 agents - 19 patterns - All free data
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <div style="background:{rbg};border:1px solid {rborder};border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{rcol};letter-spacing:0.1em">
        REGIME {regime} - {rlbl}
      </span>
    </div>
    <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:#9AAFC4">Breadth {breadth}/10</span>
    </div>
    <div style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.1);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{mcol}">Macro: {macro_state}</span>
    </div>
    <div style="background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.1);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{ecol}">Event: {event_risk}</span>
    </div>
    <div style="background:rgba(240,180,41,0.06);border:1px solid rgba(240,180,41,0.2);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:#9AAFC4">
        {len(t1)} Picks (cap {t1_cap}) - {len(t2)} Watchlist - {len(near_bo)} Near-breakout
      </span>
    </div>
  </div>
</div>

<div style="background:#0D1520;padding:20px 28px;border:1px solid #1F3046;
            border-top:none;border-radius:0 0 12px 12px">

  <div style="background:{rbg};border:1px solid {rborder};border-radius:8px;
              padding:10px 14px;margin-bottom:20px;font-size:12px;color:{rcol}">
    <strong>Regime {regime}:</strong> {rnote}
    {"" if penalty == 0 else f'<span style="color:#9AAFC4"> - Score penalty: {penalty} pts applied.</span>'}
  </div>

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#5E7A96;
              text-transform:uppercase;margin-bottom:10px">
    Section 1 - Evidence-Based Trade Cards
  </div>
  {tier_cards}

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#5E7A96;
              text-transform:uppercase;margin:24px 0 10px">
    Section 2 - Top 20 Watchlist (T2/T3 only - T1 picks shown above)
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead>
      <tr style="background:#0A1018;border-bottom:1px solid #1F3046">
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">TIER</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">TICKER</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">PATTERN</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">SCORE</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">RS</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">RVOL</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">ENTRY</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">SL</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">T1</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">R:R</th>
        <th style="padding:8px 10px;text-align:left;color:#5E7A96;font-size:9px">UNI</th>
      </tr>
    </thead>
    <tbody>{top20_rows}</tbody>
  </table>
  </div>

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#5E7A96;
              text-transform:uppercase;margin:24px 0 10px">
    Section 3 - Market Intelligence
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
    <div style="background:#0E1622;border:1px solid #1F3046;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#5E7A96;text-transform:uppercase;margin-bottom:8px">Market Breadth</div>
      <div style="height:4px;background:#1F3046;border-radius:2px;margin-bottom:8px">
        <div style="height:4px;width:{bbar}%;background:{bcol};border-radius:2px"></div>
      </div>
      <div style="font-family:monospace;font-size:12px;color:{bcol}">{breadth}/10</div>
      <div style="font-size:11px;color:#9AAFC4;margin-top:6px">
        A/D Ratio: {ad}<br>Above 50-EMA: {ab50}%<br>52w Highs/Lows: {nh}/{nl}
      </div>
    </div>
    <div style="background:#0E1622;border:1px solid #1F3046;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#5E7A96;text-transform:uppercase;margin-bottom:8px">Regime Signal</div>
      <div style="font-family:monospace;font-size:18px;font-weight:700;color:{rcol}">{regime}</div>
      <div style="font-size:12px;color:{rcol};margin-top:2px">{rlbl}</div>
      <div style="font-size:11px;color:#9AAFC4;margin-top:8px">{rnote}</div>
    </div>
    <div style="background:#0E1622;border:1px solid #1F3046;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#5E7A96;text-transform:uppercase;margin-bottom:8px">Macro / Event</div>
      <div style="font-family:monospace;font-size:14px;font-weight:700;color:{mcol}">{macro_state}</div>
      <div style="font-size:11px;color:#9AAFC4;margin-top:4px">T1 cap: {t1_cap} stocks</div>
      <div style="margin-top:8px;font-size:11px;color:{ecol}">Event: {event_risk}</div>
    </div>
  </div>

  {near_section}

  <div style="margin-top:24px;padding-top:16px;border-top:1px solid #1F3046;
              font-size:10px;color:#2D4055;text-align:center;line-height:1.8">
    NSE Momentum Scanner v5.3 - 404 stocks - All free data - Evidence-based<br>
    T1 = Gate cleared. T2 = One condition missing. T3 = Setup forming.<br>
    Near-breakout = Set alert only, do not buy until breakout confirmed.<br>
    Not SEBI-registered investment advice. All trading involves capital risk.
  </div>

</div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    print("Emailer v5.3 loaded OK")
