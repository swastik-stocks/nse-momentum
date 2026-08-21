"""
NSE Momentum v6.2 - Email Reporter
6-section HTML email:
  Section 1: T1/T2/T3 evidence-based trade cards
  Section 2: Top 20 watchlist table (no T1 duplicates)
  Section 3: Market intelligence (regime, breadth, macro, event)
  Section 4: Near-breakout watchlist (set alerts, do not buy yet)
  Section 5: Defensive / Relative-Strength watchlist (capital preservation
             triage Î“Ã‡Ã¶ only populated when the scan actually produced one;
             see agents/defensive_agent.py for the trigger logic). Styled
             deliberately differently from Sections 1/2/4 Î“Ã‡Ã¶ muted/neutral
             palette, no "buy" language anywhere Î“Ã‡Ã¶ because these are NOT
             new entry signals, just relatively-less-damaged names.
  Section 6: Position Alerts Î“Ã‡Ã¶ EXIT / TRIM / ADD-ON on stocks you already
             HOLD (Phase 2, P2-01/02/05/06/07). Distinct from Sections 1-5,
             which are all about the 504-stock scan universe; this section
             is about your actual Portfolio Dashboard holdings, read via
             the Turso bridge (P1-04). Only renders subsections that
             actually have entries -- an empty scan day shows nothing here,
             same "silently disappear" pattern as Sections 4/5.
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

# Change this if you ever want the greeting to say something else Î“Ã‡Ã¶
# kept as one constant rather than hardcoded inline so it's a single edit.
RECIPIENT_NAME = "Swastik"

REGIME_META = {
    "A": ("STRONG BULL",  "#0E8F63", "rgba(14,143,99,0.12)", "rgba(14,143,99,0.35)",
          "All conditions optimal. Highest probability entry window."),
    "B": ("BULL",         "#0E7A6B", "rgba(14,122,107,0.10)", "rgba(14,122,107,0.3)",
          "Good conditions. Standard position sizing appropriate."),
    "C": ("RANGE BOUND",  "#96690A", "rgba(150,105,10,0.10)", "rgba(150,105,10,0.3)",
          "Choppy market. Tighter stops, smaller sizes, wait for clear breakout."),
    "D": ("CORRECTION",   "#B84D0A", "rgba(184,77,10,0.10)", "rgba(184,77,10,0.3)",
          "Market pulling back. Higher failure rate on breakouts. Watchlist mode."),
    "E": ("BEAR MARKET",  "#C6403D", "rgba(198,64,61,0.10)", "rgba(198,64,61,0.3)",
          "Avoid new longs. Capital preservation is the priority."),
}

MACRO_COLOR = {"SUPPORTIVE": "#0E8F63", "MIXED": "#96690A", "HOSTILE": "#C6403D"}
EVENT_COLOR = {"NORMAL": "#8895AA",     "WATCH": "#96690A",  "HIGH_RISK": "#C6403D"}
BQ_COLOR    = {"MAJOR": "#0E8F63",      "MINOR": "#96690A",  "RECOVERY": "#2A5FB0"}

# Deliberately muted/neutral Î“Ã‡Ã¶ NOT the green/orange/red buy-signal palette
# used elsewhere in this file. A defensive pick should never visually read
# like a Tier 1/2/3 trade card at a glance.
DEFENSIVE_COLOR  = "#5D6E85"
DEFENSIVE_BG     = "rgba(93,110,133,0.08)"
DEFENSIVE_BORDER = "rgba(93,110,133,0.25)"

# Section 6 (Position Alerts) colors Î“Ã‡Ã¶ deliberately reuse existing meanings
# from elsewhere in this file rather than invent a new palette: EXIT reuses
# the same red as the SL (stop-loss) figure in every tier card; TRIM reuses
# Regime D's "correction, pulling back" orange, not Tier 2's gold (which
# means "aggressive opportunity" Î“Ã‡Ã¶ the opposite of what TRIM signals); ADD-ON
# reuses Tier 1's green since it genuinely is a fresh breakout signal, just
# on a stock you already own instead of a new one.
EXIT_COLOR   = "#C6403D"
EXIT_BG      = "rgba(198,64,61,0.08)"
EXIT_BORDER  = "rgba(198,64,61,0.3)"
TRIM_COLOR   = "#B84D0A"
TRIM_BG      = "rgba(184,77,10,0.08)"
TRIM_BORDER  = "rgba(184,77,10,0.3)"
ADDON_COLOR  = "#0E8F63"
ADDON_BG     = "rgba(14,143,99,0.08)"
ADDON_BORDER = "rgba(14,143,99,0.3)"

# Section 6 (P4-04) Î“Ã‡Ã¶ Sector Concentration colors. â‰¡Æ’Ã¶â”¤-equivalent (concentrated
# + weak breadth) reuses EXIT's red Î“Ã‡Ã¶ same severity as a stop-loss breach,
# since both mean "this needs a decision now." â‰¡Æ’Æ’Ã­-equivalent (concentrated,
# breadth not yet weak) reuses Regime C's amber, same "watch, don't act yet"
# meaning it has everywhere else in this file, not TRIM's orange (TRIM means
# a specific stock is deteriorating; this means a sector-level bet exists,
# which may be fine if the sector stays strong).
CONC_WEAK_COLOR  = "#C6403D"
CONC_WEAK_BG     = "rgba(198,64,61,0.08)"
CONC_WEAK_BORDER = "rgba(198,64,61,0.3)"
CONC_COLOR       = "#96690A"
CONC_BG          = "rgba(150,105,10,0.08)"
CONC_BORDER      = "rgba(150,105,10,0.3)"

# Kept identical to PortfolioDashboard/app.py's CONCENTRATION_THRESHOLD_PCT /
# WEAK_SECTOR_SMA50_PCT (P3-08). Duplicated, not imported -- nse_momentum and
# PortfolioDashboard are separate repos/deployments with no shared import
# path today, same reasoning as every other per-file constant in this file
# (see EXIT/TRIM/ADDON colors above). If you change these, change the
# matching pair in app.py too -- meant to stay identical, not mechanically
# linked.
CONCENTRATION_THRESHOLD_PCT = 25.0
WEAK_SECTOR_SMA50_PCT = 50.0

# Section 1a Î“Ã‡Ã¶ Tier 1/2 at-a-glance heatmap. Colors reuse pass/fail cutoffs
# that already exist elsewhere in the codebase Î“Ã‡Ã¶ deliberately NOT new
# thresholds invented for this table, so "green" here always means the same
# thing it means in the rest of the pipeline:
#   RS_OUTPERFORM_MIN   -- agents/rs_agent.py's own scoring breakpoint
#                           (elif p >= 70: base = 15)
#   RVOL_CONFIRM_MIN /
#   RVOL_LOW_VOL_MIN    -- confirm_picks.py's CONFIRMED / CONFIRMED_LOW_VOL
#                           cutoffs (RVOL_CONFIRM_MIN, RVOL_LOW_VOL_MIN)
#   HEAT_DELIVERY_MIN   -- sector_breadth.py's HIGH_DELIVERY_THRESHOLD /
#                           institutional_proxy_agent.py's own threshold
#   HEAT_MIN_RRR        -- agents/risk_agent.py's MIN_RRR per universe (the
#                           real R:R gate). Duplicated, not imported, same
#                           reasoning as CONCENTRATION_THRESHOLD_PCT above.
#                           A stock reads GREEN on R:R only if it clears this
#                           by HEAT_RRR_MARGIN -- comfortably past the gate,
#                           not just past it.
# Score, Confidence, Entry and Risk-to-stop are deliberately left uncolored:
# Score/Confidence are the agents' own composite of everything else in the
# row (coloring them too would double-count the same evidence), and
# risk_agent.py's per-universe stop cap doesn't hold cleanly enough against
# real picks yet to trust as a green/red signal -- see emailer.py commit
# notes. Entry is just a price to act on, not a signal.
RS_OUTPERFORM_MIN = 70.0
RVOL_CONFIRM_MIN  = 1.5
RVOL_LOW_VOL_MIN  = 1.0
HEAT_DELIVERY_MIN = 50.0
HEAT_MIN_RRR      = {"LARGE": 1.5, "MID": 1.8, "SMALL": 2.0}
HEAT_RRR_MARGIN    = 1.5

HEAT_PASS_BG, HEAT_PASS_FG = "#DDF3E8", "#0B7A4F"
HEAT_MID_BG,  HEAT_MID_FG  = "#FBEED4", "#8A5A0C"
HEAT_FAIL_BG, HEAT_FAIL_FG = "#F1F3F7", "#8895AA"


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


def _load_cc_recipients() -> list:
    """Optional CC list -- recipients_cc.txt, one address per line."""
    path = BASE_DIR / "recipients_cc.txt"
    if not path.exists():
        return []
    emails = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "@" in line:
            emails.append(line)
    return emails


def send_email_report(tiers: dict):
    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        log.warning("Gmail credentials missing.")
        return

    recipients = _load_recipients()
    if not recipients:
        log.warning("No recipients in recipients.txt")
        return
    cc = _load_cc_recipients()

    t1           = tiers.get("tier1", [])
    t2           = tiers.get("tier2", [])
    t3           = tiers.get("tier3", [])
    all_r        = tiers.get("all_results", [])
    near_bo      = tiers.get("near_breakout", [])
    defensive    = tiers.get("defensive_watchlist", [])   # NEW
    regime       = tiers.get("regime", "C")
    brdth        = tiers.get("breadth", 5)
    bd           = tiers.get("breadth_detail", {})
    macro_state  = tiers.get("macro_state", "MIXED")
    event_risk   = tiers.get("event_risk", "NORMAL")
    t1_cap       = tiers.get("t1_cap", 15)
    dhan_status  = tiers.get("dhan_status", {})   # [NEW] see data_fetcher.get_dhan_status()
    data_prov    = tiers.get("data_provenance", {})   # [2026-08-18] verified freshness/provenance, see market_calendar/staleness_check.py
    # P2-07: position alerts on held stocks, from Phase 2 (P2-01/02/05/06)
    exit_alerts  = tiers.get("exit_alerts", [])
    trim_signals = tiers.get("trim_signals", [])
    add_on       = tiers.get("add_on_candidates", [])
    # P4-04: sector concentration check, from sector_concentration_alert.py
    # (orchestrator.py calls compute_sector_concentration() and passes the
    # result through here Î“Ã‡Ã¶ same "compute elsewhere, this file just
    # renders" pattern as everything else in tiers).
    sector_concentration = tiers.get("sector_concentration", [])
    held_status  = tiers.get("held_status", [])
    holding_heat = _compute_portfolio_heat(held_status)   # P4-04 heat alert
    # P4-02: signal→outcome attribution, computed by signal_attribution.py
    # and passed through tiers by orchestrator.py — same pattern as everything
    # else. Returns {} if no matched trades yet (e.g. trades_v4 empty after a
    # dry-run scan) — _signal_attribution_section renders nothing in that case.
    signal_attribution = tiers.get("signal_attribution", {})
    # P2-08 gate: ADD-ON premise not yet validated (staged research plan --
    # see orchestrator.py's ADDON_LIVE_EXECUTION). While False, recommendations
    # are computed and logged (the paper stream itself) but must never reach
    # the email -- "observe, don't pay." Read from tiers, not imported from
    # orchestrator.py, to avoid coupling this file to that one's heavy agent
    # imports; orchestrator.py is the single source of truth for the flag's
    # actual value.
    if not tiers.get("addon_live_execution", False):
        add_on = []

    rlbl, rcol, rbg, rborder, rnote = REGIME_META.get(regime, REGIME_META["C"])
    # [2026-08-18] date_str used to be purely datetime.today() — the wall-clock
    # date the process ran, with no connection to the data's actual vintage.
    # Now prefers the VERIFIED Bhavcopy trading date from data_provenance
    # (set by scanner.py's freshness gate); wall-clock is only a fallback for
    # ad-hoc scripts that don't populate data_provenance at all.
    bhav_date = data_prov.get("bhavcopy_trading_date")
    date_str = (datetime.fromisoformat(bhav_date).strftime("%d %b %Y")
                if bhav_date else datetime.today().strftime("%d %b %Y"))
    penalty  = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -25}.get(regime, 0)

    html = _build_html(t1, t2, t3, all_r, near_bo, defensive,
                       regime, rlbl, rcol, rbg, rborder, rnote,
                       brdth, bd, date_str, penalty,
                       macro_state, event_risk, t1_cap, dhan_status,
                       exit_alerts, trim_signals, add_on,
                       sector_concentration, holding_heat,
                       signal_attribution, data_prov)

    msg = MIMEMultipart("alternative")
    exit_subject_flag = f" - Î“ÃœÃ¡{len(exit_alerts)} EXIT ALERT" + ("S" if len(exit_alerts) != 1 else "") if exit_alerts else ""
    weak_concentration = [r for r in sector_concentration if r.get("flag") == "concentrated_weak"]
    conc_subject_flag = (f" - Î“ÃœÃ»{len(weak_concentration)} SECTOR CONCENTRATION"
                         if weak_concentration else "")
    msg["Subject"] = (f"NSE Momentum v6.2 - {date_str} - "
                      f"Regime {regime} ({rlbl}) - {len(t1)} picks{exit_subject_flag}{conc_subject_flag}")
    msg["From"] = GMAIL_ADDRESS
    msg["To"]   = ", ".join(recipients)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            smtp.sendmail(GMAIL_ADDRESS, recipients + cc, msg.as_string())
        log.info(f"Email sent to {len(recipients)} recipient(s)" + (f" (cc: {len(cc)})" if cc else ""))
    except Exception as e:
        log.error(f"Email send failed: {e}")
        raise


def _heat_cell(label: str, kind: str) -> str:
    """kind: 'pass' | 'mid' | 'fail' -- see threshold constants above."""
    bg, fg = {
        "pass": (HEAT_PASS_BG, HEAT_PASS_FG),
        "mid":  (HEAT_MID_BG,  HEAT_MID_FG),
        "fail": (HEAT_FAIL_BG, HEAT_FAIL_FG),
    }[kind]
    weight = "700" if kind == "pass" else "600" if kind == "mid" else "400"
    return (f'<span style="background:{bg};color:{fg};font-weight:{weight};'
            f'border-radius:6px;padding:3px 8px;display:inline-block;'
            f'min-width:42px;text-align:center">{label}</span>')


def _heatmap_table(stocks: list, tier_label: str, accent_color: str) -> str:
    """
    Section 1a -- Tier 1/2 at-a-glance heatmap. Stacks a tier's picks as
    rows so they can be scanned/compared in one view instead of reading
    each trade card in full, per the reasoning in the constants block
    above: colors are read from real pass/fail thresholds elsewhere in
    the codebase, not invented for this table, and Score/Confidence/Entry/
    Risk-to-stop are shown plain rather than colored (composite-metric
    double-counting risk for the first two, an unverified stop-cap for
    the last -- see the comment above HEAT_MIN_RRR).
    """
    if not stocks:
        return ""

    rows_html = ""
    for r in stocks:
        universe = getattr(r, "universe", "MID") or "MID"
        min_rrr  = HEAT_MIN_RRR.get(universe, HEAT_MIN_RRR["MID"])

        rs = r.rs_percentile
        rs_kind = "pass" if rs >= RS_OUTPERFORM_MIN else "fail"

        rvol = r.rvol
        rvol_kind = ("pass" if rvol >= RVOL_CONFIRM_MIN else
                     "mid"  if rvol >= RVOL_LOW_VOL_MIN else "fail")

        deliv = r.del_pct
        deliv_kind = "pass" if deliv >= HEAT_DELIVERY_MIN else "fail"

        rrr = r.rrr
        rrr_kind = ("pass" if rrr >= min_rrr * HEAT_RRR_MARGIN else
                    "mid"  if rrr >= min_rrr else "fail")

        n_green = sum(k == "pass" for k in (rs_kind, rvol_kind, deliv_kind, rrr_kind))

        rows_html += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:8px 10px;font-weight:700;color:#101826;font-family:monospace">{r.ticker.replace('.NS','')}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace;color:#55627A">Rs.{r.entry:,.1f}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace">{_heat_cell(f"{rs:.0f}th", rs_kind)}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace">{_heat_cell(f"{rvol:.1f}x", rvol_kind)}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace">{_heat_cell(f"{deliv:.0f}%", deliv_kind)}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace">{_heat_cell(f"{rrr:.1f}x", rrr_kind)}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace;color:#55627A">{r.stop_pct:.1f}%</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace;color:#55627A">{r.total_score}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace;color:#55627A">{r.confidence_pct:.0f}%</td>
  <td style="padding:8px 10px;font-size:11px;color:#8895AA">
    <b style="color:{accent_color};font-family:monospace">{n_green}</b>/4 green
  </td>
</tr>"""

    return f"""
<div style="margin-bottom:14px">
  <div style="font-family:monospace;font-size:10px;letter-spacing:0.1em;color:{accent_color};
              text-transform:uppercase;margin-bottom:8px">{tier_label} -- At A Glance</div>
  <div style="overflow-x:auto;border:1px solid #DFE5EE;border-radius:8px">
  <table style="width:100%;border-collapse:collapse;font-size:12px;min-width:600px">
    <thead>
      <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">ENTRY</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">RS %ILE</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">RVOL</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">DELIVERY</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">R:R</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">RISK TO STOP</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">SCORE</th>
        <th style="padding:8px 10px;text-align:right;color:#8895AA;font-size:9px">CONF.</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">READS</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>"""


def _tier_card(r, tier_label: str, tier_color: str) -> str:
    # MAJOR/MINOR/RECOVERY label
    bq      = getattr(r, "breakout_quality", "") or "MINOR"
    bq_col  = BQ_COLOR.get(bq, "#96690A")
    bq_html = (f'<span style="font-size:9px;color:{bq_col};'
               f'border:1px solid {bq_col};border-radius:3px;'
               f'padding:1px 5px;margin-left:6px">{bq}</span>')

    # NEW Î“Ã‡Ã¶ low historical edge badge. Distinct from breakout_quality above:
    # that measures breakout SIZE (major vs minor move), this measures the
    # PATTERN's own backtested expectancy (e.g. High Base +0.09% Î“Ã‡Ã¶ barely
    # above zero across 20,916 signals). A pick can be a MAJOR breakout on
    # a pattern with near-zero historical edge; both facts matter separately.
    low_edge_html = ""
    if getattr(r, "low_edge_pattern", False):
        low_edge_html = (
            '<span style="font-size:9px;color:#BA4A1E;'
            'border:1px solid #BA4A1E;border-radius:3px;'
            'padding:1px 5px;margin-left:6px" '
            'title="This pattern\'s prior expectancy estimate is disputed Î“Ã‡Ã¶ '
            'score cleared on other factors, not pattern strength">'
            'LOW HISTORICAL EDGE</span>'
        )

    # Confirmation state
    conf       = getattr(r, "confirmation_state", "SETUP_READY")
    conf_col   = "#0E8F63" if conf == "BREAKOUT_CONFIRMED" else "#96690A"
    conf_label = "CONFIRMED" if conf == "BREAKOUT_CONFIRMED" else "SETUP READY"

    # Asymmetry
    # [FIX] Previously preferred r.asymmetry_risk_pct / r.asymmetry_reward_pct
    # when present. Those are computed by the AsymmetryGate against a
    # DIFFERENT reference point (appears to be the pattern-detection zone's
    # bound, not the stated Entry price) -- legitimate for that gate's own
    # risk/reward decision, but wrong to show unlabeled next to "T1 Rs.X"
    # and "SL Rs.X", which read as "gain/risk from Entry." Confirmed via
    # cross-check against real report data: r.gain_pct_t1 / r.stop_pct
    # matched (target1-entry)/entry and (entry-sl)/entry exactly; the
    # asymmetry_* values were ~4-6 points higher, matching a calculation
    # anchored to the pattern zone low instead of Entry. Always use the
    # Entry-anchored values here now.
    risk_pct   = r.stop_pct
    reward_pct = r.gain_pct_t1
    rr_actual  = r.rrr   # consistent with table display

    working_html = "".join(
        f'<li style="margin:3px 0;color:#55627A">OK {w}</li>'
        for w in (r.what_is_working or [])[:3]
    )
    missing_html = "".join(
        f'<li style="margin:3px 0;color:#96690A">! {m}</li>'
        for m in (r.what_is_missing or [])[:2]
    )
    trigger_html = "".join(
        f'<li style="margin:3px 0;color:#2A5FB0">- {t}</li>'
        for t in (r.trigger_conditions or [])[:2]
    )
    risk_html = "".join(
        f'<li style="margin:3px 0;color:#B84D0A">! {rk}</li>'
        for rk in (r.risk_factors or [])[:2]
    )

    return f"""
<div style="background:#FFFFFF;border:1px solid #DFE5EE;border-left:3px solid {tier_color};
            border-radius:8px;padding:16px 18px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <span style="font-family:monospace;font-size:10px;color:{tier_color};
                   letter-spacing:0.15em;text-transform:uppercase">{tier_label}</span>
      <div style="font-size:17px;font-weight:700;color:#101826;margin:3px 0">
        {r.ticker.replace('.NS','')} {bq_html}{low_edge_html}
      </div>
      <div style="font-size:11px;color:#8895AA">{r.name} - {r.sector} - {r.universe}</div>
      <div style="margin-top:4px">
        <span style="font-size:9px;color:{conf_col};border:1px solid {conf_col};
                     border-radius:3px;padding:1px 5px">{conf_label}</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-family:monospace;font-size:22px;font-weight:700;color:{tier_color}">{r.total_score}</div>
      <div style="font-family:monospace;font-size:9px;color:#8895AA">/ 100 pts</div>
      <div style="font-size:10px;color:#55627A;margin-top:2px">Conf: {r.confidence_pct:.0f}%</div>
    </div>
  </div>

  <div style="background:#F7F9FC;border-radius:6px;padding:10px 14px;margin-bottom:10px;
              font-family:monospace;font-size:11px;">
    <div style="display:flex;gap:16px;flex-wrap:wrap;color:#55627A">
      <span>{r.pattern}</span>
      <span>RS {r.rs_percentile:.0f}th%</span>
      <span title="End-of-day RVOL: today's closed daily volume vs 20-day average. Not the same metric as the 10am confirmation email's intraday RVOL.">RVOL {r.rvol:.1f}x (EOD)</span>
      <span title="Daily-close RSI(14)">RSI {r.rsi_val:.0f} (D)</span>
      <span>Del {r.del_pct:.0f}%</span>
    </div>
    <div style="margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;font-size:12px">
      <span style="color:#0E7A6B">Entry Rs.{r.entry:.1f}</span>
      <span style="color:#C6403D">SL Rs.{r.stop_loss:.1f} ({risk_pct:.1f}% risk)</span>
      <span style="color:#A8680E">T1 Rs.{r.target1:.1f} (+{reward_pct:.1f}%)</span>
      <span style="color:#7C4FC4">T2 Rs.{r.target2:.1f}</span>
      <span style="color:#0E8F63">R:R {rr_actual:.1f}x</span>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px">
    <div>
      <div style="color:#8895AA;font-size:9px;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:4px">What is working</div>
      <ul style="margin:0;padding:0 0 0 14px">{working_html}</ul>
    </div>
    <div>
      {"" if not missing_html else f'<div style="color:#8895AA;font-size:9px;text-transform:uppercase;margin-bottom:4px">Missing</div><ul style="margin:0;padding:0 0 0 14px">{missing_html}</ul>'}
      {"" if not trigger_html else f'<div style="color:#8895AA;font-size:9px;text-transform:uppercase;margin-bottom:4px;margin-top:6px">Trigger to act</div><ul style="margin:0;padding:0 0 0 14px">{trigger_html}</ul>'}
      {"" if not risk_html else f'<div style="color:#8895AA;font-size:9px;text-transform:uppercase;margin-bottom:4px;margin-top:6px">Risk factors</div><ul style="margin:0;padding:0 0 0 14px">{risk_html}</ul>'}
    </div>
  </div>
</div>"""


def _near_breakout_section(near_bo: list) -> str:
    if not near_bo:
        return ""
    rows = ""
    for nb in near_bo:
        rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:600;color:#101826;font-family:monospace">
    {nb['ticker'].replace('.NS','')}
  </td>
  <td style="padding:7px 10px;color:#55627A;font-size:11px">{nb['name']}</td>
  <td style="padding:7px 10px;color:#55627A;font-size:11px">{nb['pattern']}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#0E7A6B">Rs.{nb['price']:.1f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#A8680E">Rs.{nb['breakout']:.1f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#96690A">{nb['gap_pct']:.1f}% away</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">{nb.get('rsi',0):.0f}</td>
  <td style="padding:7px 10px;font-size:10px;color:#8895AA">{nb['universe']}</td>
</tr>"""

    return f"""
  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin:24px 0 10px">
    Section 4 - Near-Breakout Watchlist (set alerts, do not buy yet)
  </div>
  <div style="background:#F7F9FC;border:1px solid #DFE5EE;border-radius:8px;
              padding:10px 14px;margin-bottom:12px;font-size:11px;color:#55627A">
    These stocks are within 3% of their breakout level with valid patterns forming.
    They have NOT triggered yet. Set an alert at the breakout level. Buy only on a
    confirmed close above breakout on volume >= 1.5x average.
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead>
      <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">NAME</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">PATTERN</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">CMP</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">BREAKOUT</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">GAP</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RSI</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">UNI</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>"""


def _defensive_section(defensive: list, regime: str) -> str:
    """
    NEW Î“Ã‡Ã¶ Section 5. Only renders anything when defensive_watchlist is
    non-empty (i.e. agents/defensive_agent.py actually triggered and found
    qualifying candidates that scan). Deliberately styled with the muted
    DEFENSIVE_COLOR palette, not the green/gold/blue used for T1/T2/T3 Î“Ã‡Ã¶
    a reader should never mistake this table for a buy-signal list.
    """
    if not defensive:
        return ""

    rows = ""
    for d in defensive:
        note_html = ""
        if d.get("note"):
            note_html = (f'<div style="font-size:10px;color:#55627A;margin-top:2px">'
                         f'{d["note"]}</div>')
        rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:600;color:#101826;font-family:monospace">
    {d['ticker']}
    {note_html}
  </td>
  <td style="padding:7px 10px;color:#55627A;font-size:11px">{d['sector']}</td>
  <td style="padding:7px 10px;font-size:10px;color:#8895AA">{d['tier']}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{DEFENSIVE_COLOR}">{d['rs_universe_pct']:.0f}th</td>
  <td style="padding:7px 10px;font-family:monospace;color:{DEFENSIVE_COLOR}">{d['rs_sector_pct']:.0f}th</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">{d['stock_dd_pct']:.1f}%</td>
  <td style="padding:7px 10px;font-family:monospace;color:#8895AA">{d['nifty_dd_pct']:.1f}%</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">Rs.{d['adt_cr']:.0f}Cr</td>
</tr>"""

    return f"""
  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin:24px 0 10px">
    Section 5 - Defensive / Relative-Strength Watchlist
  </div>
  <div style="background:{DEFENSIVE_BG};border:1px solid {DEFENSIVE_BORDER};border-radius:8px;
              padding:10px 14px;margin-bottom:12px;font-size:11px;color:#55627A">
    Regime {regime} triggered a capital-preservation scan Î“Ã‡Ã¶ no new Tier 1/2 entries
    are being generated today. These names are relatively less damaged than NIFTY
    (higher relative strength, shallower drawdown over the same window) Î“Ã‡Ã¶ useful
    as a "what's holding up" reference or a hold/rotate-into view for existing
    positions. <strong style="color:{DEFENSIVE_COLOR}">This is not a new-entry buy
    signal</strong> Î“Ã‡Ã¶ treat it with the same caution as the rest of a Regime
    {regime} scan.
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead>
      <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">SECTOR</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TIER</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RS (UNIVERSE)</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RS (SECTOR)</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">STOCK DD</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">NIFTY DD</th>
        <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">ADT</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>"""


def _position_alerts_section(exit_alerts: list, trim_signals: list, add_on: list) -> str:
    """
    Section 6 (P2-07) Î“Ã‡Ã¶ EXIT / TRIM / ADD-ON on stocks you already HOLD, per
    Portfolio Dashboard (read via the Turso bridge, P1-04). Distinct data
    source from every other section in this file, which is all about the
    504-stock scan universe.

    Three independent subsections, each following the same "render nothing
    if empty" rule as _near_breakout_section/_defensive_section above Î“Ã‡Ã¶ an
    empty scan day should not show empty tables. exit_alerts items are
    held_status dicts (ticker/exit_check/technical_stop); trim_signals items
    are {ticker, rs_percentile}; add_on items are held_status dicts
    (ticker/result) with result.tier/total_score/rvol/rs_percentile/pattern.
    """
    if not exit_alerts and not trim_signals and not add_on:
        return ""

    section_header = f"""
  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin:24px 0 10px">
    Section 6 - Position Alerts (Your Holdings)
  </div>"""

    exit_html = ""
    if exit_alerts:
        rows = ""
        for h in exit_alerts:
            ec = h["exit_check"]
            ts = h["technical_stop"]
            rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:700;color:#101826;font-family:monospace">
    {h['ticker'].replace('.NS','')}
  </td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">Rs.{ec['avg_price']:.2f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{EXIT_COLOR}">Rs.{ts['current_price']:.2f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{EXIT_COLOR}">Rs.{ec['effective_stop']:.2f}</td>
  <td style="padding:7px 10px;font-size:10px;color:#8895AA">{ec['source']}</td>
</tr>"""
        exit_html = f"""
  <div style="background:{EXIT_BG};border:1px solid {EXIT_BORDER};border-radius:8px;
              padding:12px 14px;margin-bottom:14px">
    <div style="font-size:12px;font-weight:700;color:{EXIT_COLOR};margin-bottom:6px">
      Î“ÃœÃ¡ {len(exit_alerts)} EXIT ALERT{"S" if len(exit_alerts) != 1 else ""} Î“Ã‡Ã¶ price below effective stop
    </div>
    <div style="font-size:11px;color:#55627A;margin-bottom:10px">
      Effective stop = max(hard stop from your real avg cost, today's technical stop) Î“Ã‡Ã¶
      ratchet-only, same rule day5_stop_ratchet.py applies to scanner-originated trades.
      <strong style="color:{EXIT_COLOR}">Not an automatic sell order</strong> Î“Ã‡Ã¶ a data
      point for your own decision, same caution as everywhere else in this email.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">AVG COST</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">CURRENT</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">STOP</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">STOP SOURCE</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""

    trim_html = ""
    if trim_signals:
        rows = ""
        for t in sorted(trim_signals, key=lambda x: x["rs_percentile"]):
            rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:600;color:#101826;font-family:monospace">
    {t['ticker'].replace('.NS','')}
  </td>
  <td style="padding:7px 10px;font-family:monospace;color:{TRIM_COLOR}">{t['rs_percentile']:.0f}th pct</td>
</tr>"""
        trim_html = f"""
  <div style="background:{TRIM_BG};border:1px solid {TRIM_BORDER};border-radius:8px;
              padding:12px 14px;margin-bottom:14px">
    <div style="font-size:12px;font-weight:700;color:{TRIM_COLOR};margin-bottom:6px">
      {len(trim_signals)} TRIM signal{"s" if len(trim_signals) != 1 else ""} Î“Ã‡Ã¶ RS deteriorating, not yet stopped out
    </div>
    <div style="font-size:11px;color:#55627A;margin-bottom:10px">
      Relative strength below the 40th percentile Î“Ã‡Ã¶ below where "outperforming" starts
      (70th pct elsewhere in this email). Not an EXIT alert; a graduated warning worth
      watching, not yet a stop breach.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RS PERCENTILE</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""

    addon_html = ""
    if add_on:
        rows = ""
        for h in add_on:
            r = h["result"]
            rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:600;color:#101826;font-family:monospace">
    {h['ticker'].replace('.NS','')}
  </td>
  <td style="padding:7px 10px;color:#55627A;font-size:11px">{r.pattern}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{ADDON_COLOR}">{r.total_score}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{ADDON_COLOR}">{r.rvol:.1f}x</td>
  <td style="padding:7px 10px;font-family:monospace;color:{ADDON_COLOR}">{r.rs_percentile:.0f}th</td>
  <td style="padding:7px 10px;font-family:monospace;color:#0E7A6B">Rs.{r.entry:.1f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:#C6403D">Rs.{r.stop_loss:.1f}</td>
</tr>"""
        addon_html = f"""
  <div style="background:{ADDON_BG};border:1px solid {ADDON_BORDER};border-radius:8px;
              padding:12px 14px;margin-bottom:14px">
    <div style="font-size:12px;font-weight:700;color:{ADDON_COLOR};margin-bottom:6px">
      {len(add_on)} ADD-ON candidate{"s" if len(add_on) != 1 else ""} Î“Ã‡Ã¶ held stock(s) with a fresh breakout
    </div>
    <div style="font-size:11px;color:#55627A;margin-bottom:10px">
      Already in Tier 1/2 above under its own ticker Î“Ã‡Ã¶ flagged here separately because
      you already own it. Sizing/blended-cost-stop rules for adding to an existing
      position are not yet built (P2-03/P2-04) Î“Ã‡Ã¶ treat this as information, not a sizing
      recommendation.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">PATTERN</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">SCORE</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RVOL</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">RS</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">ENTRY</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">SL</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""

    return section_header + exit_html + trim_html + addon_html


def _compute_portfolio_heat(held_status: list) -> dict:
    """
    P4-04: compute per-holding heat from held_status without a second Turso
    round-trip. Returns {ticker: {current_price, stop, distance_pct, method}}
    or {} if no technical stops are available yet.
    heat per position = (current_price - effective_stop) / current_price * 100
    """
    if not held_status:
        return {}
    result = {}
    for h in held_status:
        ts = h.get("technical_stop")
        ec = h.get("exit_check")
        if not ts or not ec:
            continue
        price = ts.get("current_price", 0)
        stop  = ec.get("effective_stop", 0)
        if price <= 0 or stop <= 0:
            continue
        result[h["ticker"]] = {
            "current_price": price,
            "stop":          stop,
            "distance_pct":  round((price - stop) / price * 100, 1),
            "method":        ts.get("method", ""),
        }
    return result


def _portfolio_heat_alert(holding_heat: dict, heat_warning_pct: float = 5.0) -> str:
    """
    P4-04: email subsection for positions within heat_warning_pct% of their
    stop. Returns "" when nothing is within the warning threshold â€” same
    "silently disappear" rule as every other conditional subsection in
    this file.
    """
    if not holding_heat:
        return ""
    close_to_stop = sorted(
        [(t, d) for t, d in holding_heat.items() if d["distance_pct"] <= heat_warning_pct],
        key=lambda x: x[1]["distance_pct"]
    )
    if not close_to_stop:
        return ""

    rows = ""
    for ticker, d in close_to_stop:
        rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:700;color:#101826;font-family:monospace">
    {ticker.replace('.NS','')}
  </td>
  <td style="padding:7px 10px;font-family:monospace;color:#B84D0A">{d['distance_pct']:.1f}% to stop</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">Stop â‚¹{d['stop']:.2f} ({d['method']})</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">CMP â‚¹{d['current_price']:.2f}</td>
</tr>"""

    return f"""
  <div style="background:rgba(184,77,10,0.08);border:1px solid rgba(184,77,10,0.3);
              border-radius:8px;padding:12px 14px;margin-bottom:14px">
    <div style="font-size:12px;font-weight:700;color:#B84D0A;margin-bottom:6px">
      âš  {len(close_to_stop)} position(s) within {heat_warning_pct:.0f}% of their stop
    </div>
    <div style="font-size:11px;color:#55627A;margin-bottom:10px">
      These holdings are trading close to their technical stop-loss level.
      Not an automatic sell â€” a visibility flag for your own review.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">DISTANCE</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">STOP</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">CMP</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""


def _sector_concentration_section(sector_concentration: list) -> str:
    """
    Section 6, fourth subsection (P4-04) Î“Ã‡Ã¶ "are you unknowingly making one
    big sector bet?" Email-alert version of Portfolio Dashboard's Sector
    Concentration view (P3-08, app.py). sector_concentration items come
    from sector_concentration_alert.compute_sector_concentration():
    {sector, invested_value, pct_of_portfolio, sector_sma50, flag} where
    flag is "" / "concentrated" / "concentrated_weak".

    Same "render nothing if empty" rule as exit/trim/add-on above -- but
    here "empty" specifically means no FLAGGED sector, not zero rows.
    Showing every held sector's weight every day (most of which are well
    under the threshold) would be noise; this only surfaces when there's
    something worth a decision, same principle as EXIT/TRIM only firing
    on an actual breach/deterioration rather than listing every holding's
    status daily.
    """
    flagged = [r for r in sector_concentration if r["flag"]]
    if not flagged:
        return ""

    weak = [r for r in flagged if r["flag"] == "concentrated_weak"]
    mild = [r for r in flagged if r["flag"] == "concentrated"]

    rows = ""
    for r in sorted(flagged, key=lambda x: -x["pct_of_portfolio"]):
        is_weak = r["flag"] == "concentrated_weak"
        row_color = CONC_WEAK_COLOR if is_weak else CONC_COLOR
        sma50_str = f"{r['sector_sma50']:.1f}%" if r["sector_sma50"] is not None else "Î“Ã‡Ã¶"
        rows += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:7px 10px;font-weight:600;color:#101826">{r['sector']}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{row_color}">Rs.{r['invested_value']:,.0f}</td>
  <td style="padding:7px 10px;font-family:monospace;color:{row_color}">{r['pct_of_portfolio']:.1f}%</td>
  <td style="padding:7px 10px;font-family:monospace;color:#55627A">{sma50_str}</td>
</tr>"""

    header_color = CONC_WEAK_COLOR if weak else CONC_COLOR
    header_bg = CONC_WEAK_BG if weak else CONC_BG
    header_border = CONC_WEAK_BORDER if weak else CONC_BORDER
    label_bits = []
    if weak:
        label_bits.append(f"{len(weak)} concentrated + weak sector{'s' if len(weak) != 1 else ''}")
    if mild:
        label_bits.append(f"{len(mild)} concentrated sector{'s' if len(mild) != 1 else ''}")

    return f"""
  <div style="background:{header_bg};border:1px solid {header_border};border-radius:8px;
              padding:12px 14px;margin-bottom:14px">
    <div style="font-size:12px;font-weight:700;color:{header_color};margin-bottom:6px">
      Î“ÃœÃ» {" + ".join(label_bits)} Î“Ã‡Ã¶ Î“Ã«Ã‘{CONCENTRATION_THRESHOLD_PCT:.0f}% of portfolio in one sector
    </div>
    <div style="font-size:11px;color:#55627A;margin-bottom:10px">
      â‰¡Æ’Ã¶â”¤ rows are also below {WEAK_SECTOR_SMA50_PCT:.0f}% of that sector's stocks above SMA50 Î“Ã‡Ã¶
      concentrated AND the sector itself is currently weak. â‰¡Æ’Æ’Ã­ rows are concentrated but the
      sector's breadth hasn't turned weak yet. Not an instruction to sell or rebalance Î“Ã‡Ã¶
      just visibility into a bet you may not have noticed, same caution as everywhere else
      in this email.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">SECTOR</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">INVESTED</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">% OF PORTFOLIO</th>
          <th style="padding:7px 10px;text-align:left;color:#8895AA;font-size:9px">SECTOR SMA50 BREADTH</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""


def _signal_attribution_section(attribution: dict) -> str:
    """
    Section 7 (P4-02) — Signal→Outcome Attribution. Answers "are the scanner's
    live signals working with real money?" by joining trades_v4 scanner picks
    against realized_trades broker data (see signal_attribution.py). Renders
    only when there are matched trades — returns "" if attribution is empty or
    has no matched data, same empty-renders-nothing rule as every other section.

    This is the real-money version of validation/pipeline_replay_deep.py:
    not historical simulation, but actual capital deployed following live signals.
    The "signal worked" definition is net_pnl_pct > 0 (cost-inclusive, since
    realized_trades already stores net_pnl_pct after broker charges from
    import_broker_trades.py).

    Shows three breakdowns: by pattern, by score bucket, by regime — same
    three dimensions signal_attribution.py aggregates to, so the email is a
    direct reflection of the Turso signal_attribution_summary table.
    """
    if not attribution or not attribution.get("n_matched"):
        return ""

    agg     = attribution.get("aggregates", {})
    overall = agg.get("overall", {})
    if not overall:
        return ""

    n_matched  = attribution.get("n_matched", 0)
    n_signals  = attribution.get("n_signals", 0)
    n_realized = attribution.get("n_realized", 0)

    def _rows_html(data: dict) -> str:
        if not data:
            return "<tr><td colspan='5' style='padding:7px 10px;color:#8895AA'>No data</td></tr>"
        html = ""
        for k, s in sorted(data.items(), key=lambda x: -(x[1].get("avg_net_pnl_pct") or 0)):
            if not s:
                continue
            wr    = s.get("win_rate", 0)
            avg   = s.get("avg_net_pnl_pct", 0)
            total = s.get("total_net_pnl", 0)
            n     = s.get("n", 0)
            val_color = "#0E7A6B" if avg >= 0 else "#C6403D"
            html += f"""
<tr style="border-bottom:1px solid #DFE5EE">
  <td style="padding:6px 10px;color:#101826;font-size:11px">{k}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#55627A;font-size:11px">{n}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#55627A;font-size:11px">{wr:.0f}%</td>
  <td style="padding:6px 10px;font-family:monospace;color:{val_color};font-size:11px">{avg:+.2f}%</td>
  <td style="padding:6px 10px;font-family:monospace;color:{val_color};font-size:11px">₹{total:,.0f}</td>
</tr>"""
        return html

    def _subsection(title: str, data: dict) -> str:
        return f"""
<div style="margin-bottom:10px">
  <div style="font-size:10px;font-weight:700;color:#8895AA;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:6px">{title}</div>
  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead>
      <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
        <th style="padding:6px 10px;text-align:left;color:#8895AA;font-size:9px"></th>
        <th style="padding:6px 10px;text-align:left;color:#8895AA;font-size:9px">N</th>
        <th style="padding:6px 10px;text-align:left;color:#8895AA;font-size:9px">WIN%</th>
        <th style="padding:6px 10px;text-align:left;color:#8895AA;font-size:9px">AVG P&L%</th>
        <th style="padding:6px 10px;text-align:left;color:#8895AA;font-size:9px">TOTAL P&L</th>
      </tr>
    </thead>
    <tbody>{_rows_html(data)}</tbody>
  </table>
</div>"""

    overall_color = "#0E7A6B" if overall.get("avg_net_pnl_pct", 0) >= 0 else "#C6403D"
    overall_avg   = overall.get("avg_net_pnl_pct", 0)
    overall_wr    = overall.get("win_rate", 0)
    overall_total = overall.get("total_net_pnl", 0)

    pattern_html = _subsection("By Pattern",      agg.get("by_pattern", {}))
    bucket_html  = _subsection("By Score Bucket", agg.get("by_score_bucket", {}))
    regime_html  = _subsection("By Regime",       agg.get("by_regime", {}))

    return f"""
  <div style="margin-top:20px">
    <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
                text-transform:uppercase;margin-bottom:10px">
      Section 7 - Signal→Outcome Attribution (P4-02)
    </div>
    <div style="background:rgba(14,122,107,0.05);border:1px solid rgba(14,122,107,0.2);
                border-radius:8px;padding:12px 14px;margin-bottom:14px">
      <div style="font-size:12px;font-weight:700;color:#0E7A6B;margin-bottom:4px">
        Live Signal Performance — Real Money vs Scanner Picks
      </div>
      <div style="font-size:11px;color:#55627A;margin-bottom:10px">
        {n_matched} matched trades out of {n_signals} scanner signals and {n_realized} broker
        trades (±5 day entry window). Win rate {overall_wr:.0f}% |
        Avg net P&L <span style="color:{overall_color}">{overall_avg:+.2f}%</span> |
        Total ₹<span style="color:{overall_color}">{overall_total:,.0f}</span>.
        Not a performance guarantee — small sample, grows as more broker trades are imported.
      </div>
      {pattern_html}
      {bucket_html}
      {regime_html}
    </div>
  </div>"""


def _build_html(t1, t2, t3, all_r, near_bo, defensive,
                regime, rlbl, rcol, rbg, rborder, rnote,
                breadth, bd, date_str, penalty,
                macro_state, event_risk, t1_cap, dhan_status=None,
                exit_alerts=None, trim_signals=None, add_on=None,
                sector_concentration=None, holding_heat=None,
                signal_attribution=None, data_prov=None) -> str:

    mcol = MACRO_COLOR.get(macro_state, "#96690A")
    ecol = EVENT_COLOR.get(event_risk,  "#8895AA")
    exit_alerts        = exit_alerts or []
    trim_signals       = trim_signals or []
    add_on             = add_on or []
    sector_concentration = sector_concentration or []
    holding_heat       = holding_heat or {}
    signal_attribution = signal_attribution or {}

    # [NEW] Dhan status banner Î“Ã‡Ã¶ only rendered when Dhan was unavailable
    # this run (expired/missing token, network error). See
    # data_fetcher.get_dhan_status() and its module docstring for why this
    # is deliberately an ACTIVE daily reminder rather than a silent
    # fallback: Dhan tokens expire every 24h and this system does not
    # auto-refresh them by design.
    dhan_status = dhan_status or {}
    dhan_banner = ""
    if dhan_status.get("checked") and not dhan_status.get("available"):
        dhan_banner = f"""
  <div style="background:rgba(198,64,61,0.08);border:1px solid rgba(198,64,61,0.3);
              border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:12px;color:#C6403D">
    <strong>Î“ÃœÃ¡ Dhan unavailable this run:</strong> {dhan_status.get("message", "reason unknown")}
    <span style="color:#55627A"> Î“Ã‡Ã¶ running on tvDatafeed/Yahoo fallback. Refresh at web.dhan.co.</span>
  </div>"""

    # [2026-08-18] Data-provenance banner -- makes every key number's data
    # date/coverage/code-version explicit rather than implicit, per the
    # non-negotiable "state the exact data date used" requirement. Renders
    # a compact info line always (not just on failure, unlike dhan_banner),
    # and escalates to a red warning line specifically for the two
    # conditions that mean "trust this less": a dirty/local code run, or a
    # fallback VIX.
    data_prov = data_prov or {}
    cov       = data_prov.get("ohlcv_universe_coverage", {}) or {}
    code_prov = data_prov.get("code_provenance", {}) or {}
    gen_at    = data_prov.get("generated_at", "")
    gen_at_disp = gen_at[11:16] if gen_at and len(gen_at) >= 16 else "?"
    commit    = (code_prov.get("git_commit") or "?")[:8]
    warn_bits = []
    if code_prov.get("git_dirty"):
        warn_bits.append("LOCAL/UNCOMMITTED CODE RUN — not the verified production pipeline")
    if data_prov.get("vix_is_fallback"):
        warn_bits.append("VIX is a FALLBACK DEFAULT, not a live reading")
    if not data_prov.get("bhavcopy_date_verified", True) and data_prov:
        warn_bits.append("Bhavcopy date could not be verified")

    prov_banner = f"""
  <div style="background:#F7F9FC;border:1px solid #DFE5EE;border-radius:8px;
              padding:8px 14px;margin-bottom:16px;font-size:11px;color:#55627A;font-family:monospace">
    Data as of: Bhavcopy {data_prov.get('bhavcopy_trading_date', '?')} Â· Scan run {gen_at_disp} IST Â·
    OHLCV coverage {cov.get('fetched_fresh', '?')}/{cov.get('requested', '?')} ({cov.get('pct', '?')}%) Â·
    commit {commit}
  </div>""" if data_prov else ""
    if warn_bits:
        prov_banner += f"""
  <div style="background:rgba(198,64,61,0.08);border:1px solid rgba(198,64,61,0.3);
              border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:11px;color:#C6403D">
    <strong>Data quality warning:</strong> {' Â· '.join(warn_bits)}
  </div>"""

    # Section 1: trade cards
    # [2026-08-21] Was: t1[:8]/t2[:3]/t3[:2] -- a hardcoded render-time cap
    # far smaller than and independent of the real upstream caps
    # (orchestrator.py's t1_cap, up to 15, and T2_CAP=8), silently dropping
    # picks that fully cleared the T1 gate, sector-concentration check, and
    # sanity gate. Confirmed live 2026-08-20: 11 real T1 picks (t1_cap=15,
    # sanity gate excluded 0/14), only 8 rendered here -- AUBANK, APLAPOLLO,
    # JINDALSAW were completely absent from the email despite being fully
    # qualified, yet still sat in picks_latest.json and got auto-confirmed
    # by the next morning's checkpoint (JINDALSAW: CONFIRMED -- ENTER) with
    # the user never having seen it in the evening report at all. T1/T2 are
    # now shown in full -- both are already bounded by real upstream caps,
    # so this can't make the email unbounded. T3 keeps its own cap: it's
    # explicitly "setup forming", not gate-cleared, and has no upstream cap
    # of its own, so an unbounded watchlist really could blow up email size.
    tier_cards = ""
    if t1:
        for r in t1:
            tier_cards += _tier_card(r, "TIER 1 - TOP PICK", "#0E8F63")
    else:
        tier_cards += f"""
<div style="background:#F7F9FC;border:1px solid #DFE5EE;border-radius:8px;
            padding:20px;text-align:center;color:#8895AA;margin-bottom:12px">
  No stocks cleared the Tier 1 gate today. Regime {regime} penalty ({penalty} pts).<br>
  <span style="color:#96690A">Stay in cash. Capital preservation is the priority.</span>
</div>"""

    if t2:
        for r in t2:
            tier_cards += _tier_card(r, "TIER 2 - AGGRESSIVE", "#A8680E")
    if t3:
        for r in t3[:2]:
            tier_cards += _tier_card(r, "TIER 3 - WATCHLIST", "#2A5FB0")

    # Section 1a: Tier 1/2 at-a-glance heatmap -- sits above the existing
    # Section 1 cards, doesn't replace them. Tier 3 excluded: those are
    # setup-forming watchlist names, not gate-cleared/one-condition-missing
    # picks, so ranking them on the same pass/fail columns wouldn't mean
    # the same thing.
    heatmap_html = (_heatmap_table(t1, "TIER 1 - TOP PICK",   "#0E8F63")
                     + _heatmap_table(t2, "TIER 2 - AGGRESSIVE", "#A8680E"))

    # Section 2: watchlist table Î“Ã‡Ã¶ EXCLUDE T1 tickers to avoid duplicates
    t1_tickers = {r.ticker for r in t1}
    table_stocks = [r for r in all_r if r.ticker not in t1_tickers][:20]
    top20_rows = ""
    for i, r in enumerate(table_stocks, 1):
        bg      = "#F7F9FC" if i % 2 == 0 else "transparent"
        tcol    = {"1": "#0E8F63", "2": "#A8680E", "3": "#2A5FB0"}.get(str(r.tier), "#8895AA")
        bq      = getattr(r, "breakout_quality", "MINOR") or "MINOR"
        bq_col  = BQ_COLOR.get(bq, "#96690A")
        top20_rows += f"""
<tr style="background:{bg}">
  <td style="padding:6px 10px;font-family:monospace;font-size:10px;color:{tcol}">T{r.tier}</td>
  <td style="padding:6px 10px;font-weight:600;color:#101826">{r.ticker.replace('.NS','')}</td>
  <td style="padding:6px 10px;color:#55627A;font-size:11px">
    {r.pattern}
    <span style="color:{bq_col};font-size:9px;margin-left:4px">[{bq}]</span>
  </td>
  <td style="padding:6px 10px;font-family:monospace;color:{tcol}">{r.total_score}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#55627A">{r.rs_percentile:.0f}%</td>
  <td style="padding:6px 10px;font-family:monospace;color:#96690A">{r.rvol:.1f}x</td>
  <td style="padding:6px 10px;font-family:monospace;color:#0E7A6B">Rs.{r.entry:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#C6403D">Rs.{r.stop_loss:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#A8680E">Rs.{r.target1:.1f}</td>
  <td style="padding:6px 10px;font-family:monospace;color:#0E8F63">{r.rrr:.1f}x</td>
  <td style="padding:6px 10px;font-size:10px;color:#8895AA">{r.universe}</td>
</tr>"""

    ad   = bd.get("ad_ratio", "-")
    ab50 = bd.get("above_50_pct", "-")
    nh   = bd.get("new_highs", "-")
    nl   = bd.get("new_lows", "-")
    bbar = int(breadth / 10 * 100)
    bcol = "#0E8F63" if breadth >= 7 else "#96690A" if breadth >= 4 else "#C6403D"

    near_section       = _near_breakout_section(near_bo)
    defensive_section  = _defensive_section(defensive, regime)   # NEW
    position_section   = (_position_alerts_section(exit_alerts, trim_signals, add_on)   # NEW P2-07
                           + _sector_concentration_section(sector_concentration)         # NEW P4-04
                           + _portfolio_heat_alert(holding_heat)                         # NEW P4-04 heat
                           + _signal_attribution_section(signal_attribution))            # NEW P4-02

    exit_badge = ""
    if exit_alerts:
        exit_badge = f"""
    <div style="background:{EXIT_BG};border:1px solid {EXIT_BORDER};
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{EXIT_COLOR}">
        Î“ÃœÃ¡ {len(exit_alerts)} EXIT ALERT{"S" if len(exit_alerts) != 1 else ""}
      </span>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:20px;background:#EEF1F6;font-family:'Segoe UI',Arial,sans-serif">
<div style="max-width:700px;margin:0 auto">

<div style="background:#FFFFFF;padding:22px 28px 18px;border-radius:12px 12px 0 0;
            border:1px solid #DFE5EE;border-bottom:none">
  <div style="font-family:monospace;font-size:10px;letter-spacing:0.2em;color:#0E7A6B;
              text-transform:uppercase;margin-bottom:6px">
    NSE Momentum Discovery - v6.2 - {date_str}
  </div>
  <div style="font-size:13px;color:#55627A;margin-bottom:6px">Hi {RECIPIENT_NAME},</div>
  <div style="font-size:22px;font-weight:800;color:#101826;margin-bottom:4px">Daily Intelligence Report</div>
  <div style="font-size:11px;color:#8895AA;margin-bottom:12px">
    500 stocks - 3 universes - 3 validated patterns (16 pruned, evidence-based) - All free data
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <div style="background:{rbg};border:1px solid {rborder};border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{rcol};letter-spacing:0.1em">
        REGIME {regime} - {rlbl}
      </span>
    </div>
    <div style="background:rgba(14,122,107,0.08);border:1px solid rgba(14,122,107,0.2);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:#55627A">Breadth {breadth}/10</span>
    </div>
    <div style="background:rgba(16,24,38,0.2);border:1px solid rgba(16,24,38,0.1);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{mcol}">Macro: {macro_state}</span>
    </div>
    <div style="background:rgba(16,24,38,0.2);border:1px solid rgba(16,24,38,0.1);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:{ecol}">Event: {event_risk}</span>
    </div>
    <div style="background:rgba(168,104,14,0.06);border:1px solid rgba(168,104,14,0.2);
                border-radius:20px;padding:5px 14px">
      <span style="font-family:monospace;font-size:10px;color:#55627A">
        {len(t1)} Picks (cap {t1_cap}) - {len(t2)} Watchlist - {len(near_bo)} Near-breakout
        {"" if not defensive else f" - {len(defensive)} Defensive"}
      </span>
    </div>
    {exit_badge}
  </div>
</div>

<div style="background:#FFFFFF;padding:20px 28px;border:1px solid #DFE5EE;
            border-top:none;border-radius:0 0 12px 12px">

  <div style="background:{rbg};border:1px solid {rborder};border-radius:8px;
              padding:10px 14px;margin-bottom:20px;font-size:12px;color:{rcol}">
    <strong>Regime {regime}:</strong> {rnote}
    {"" if penalty == 0 else f'<span style="color:#55627A"> - Score penalty: {penalty} pts applied.</span>'}
  </div>
  {dhan_banner}
  {prov_banner}

  {heatmap_html}

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin-bottom:10px">
    Section 1 - Evidence-Based Trade Cards
  </div>
  {tier_cards}

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin:24px 0 10px">
    Section 2 - Top 20 Watchlist (T2/T3 only - T1 picks shown above)
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead>
      <tr style="background:#F7F9FC;border-bottom:1px solid #DFE5EE">
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">TIER</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">TICKER</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">PATTERN</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">SCORE</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">RS</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px" title="End-of-day RVOL Î“Ã‡Ã¶ see tier cards above">RVOL (EOD)</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">ENTRY</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">SL</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">T1</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">R:R</th>
        <th style="padding:8px 10px;text-align:left;color:#8895AA;font-size:9px">UNI</th>
      </tr>
    </thead>
    <tbody>{top20_rows}</tbody>
  </table>
  </div>

  <div style="font-family:monospace;font-size:9px;letter-spacing:0.2em;color:#8895AA;
              text-transform:uppercase;margin:24px 0 10px">
    Section 3 - Market Intelligence
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
    <div style="background:#FFFFFF;border:1px solid #DFE5EE;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#8895AA;text-transform:uppercase;margin-bottom:8px">Market Breadth</div>
      <div style="height:4px;background:#DFE5EE;border-radius:2px;margin-bottom:8px">
        <div style="height:4px;width:{bbar}%;background:{bcol};border-radius:2px"></div>
      </div>
      <div style="font-family:monospace;font-size:12px;color:{bcol}">{breadth}/10</div>
      <div style="font-size:11px;color:#55627A;margin-top:6px">
        A/D Ratio: {ad}<br>Above 50-EMA: {ab50}%<br>52w Highs/Lows: {nh}/{nl}
      </div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DFE5EE;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#8895AA;text-transform:uppercase;margin-bottom:8px">Regime Signal</div>
      <div style="font-family:monospace;font-size:18px;font-weight:700;color:{rcol}">{regime}</div>
      <div style="font-size:12px;color:{rcol};margin-top:2px">{rlbl}</div>
      <div style="font-size:11px;color:#55627A;margin-top:8px">{rnote}</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DFE5EE;border-radius:8px;padding:14px">
      <div style="font-size:9px;color:#8895AA;text-transform:uppercase;margin-bottom:8px">Macro / Event</div>
      <div style="font-family:monospace;font-size:14px;font-weight:700;color:{mcol}">{macro_state}</div>
      <div style="font-size:11px;color:#55627A;margin-top:4px">T1 cap: {t1_cap} stocks</div>
      <div style="margin-top:8px;font-size:11px;color:{ecol}">Event: {event_risk}</div>
    </div>
  </div>

  {near_section}

  {defensive_section}

  {position_section}

  <div style="margin-top:24px;padding-top:16px;border-top:1px solid #DFE5EE;
              font-size:10px;color:#8895AA;text-align:center;line-height:1.8">
    NSE Momentum Scanner v6.2 - 500 stocks - All free data - Evidence-based<br>
    T1 = Gate cleared. T2 = One condition missing. T3 = Setup forming.<br>
    Near-breakout = Set alert only, do not buy until breakout confirmed.<br>
    Defensive watchlist = Capital-preservation reference, not a buy signal.<br>
    Position alerts (Section 6) = EXIT/TRIM/ADD-ON on your actual Portfolio Dashboard
    holdings, not the scan universe. Not automatic orders.<br>
    Not SEBI-registered investment advice. All trading involves capital risk.
  </div>

</div>
</div>
</body>
</html>"""


if __name__ == "__main__":
    print("Emailer v6.2 loaded OK")

