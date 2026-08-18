# -*- coding: utf-8 -*-
"""
NSE Momentum — Pre-send sanity gate

Runs immediately before every evening report send. Exists because every
existing safeguard in this codebase (RVOL confirmation, entry-drift check,
pivot-extension check in confirm_picks.py) lives in the NEXT MORNING's
checkpoint -- one day too late to stop a wrong number from ever reaching an
inbox. Built 2026-08-17 after the Cup & Handle pattern staleness bug
(TVSMOTOR entry Rs.4082.7 vs actual CMP Rs.4385.9 -- a 3-week-old pivot
presented as a fresh "SETUP READY" breakout, 13 of that evening's 20 top
picks affected the same way) reached two live emails before a human caught
it by eye. This gate is the automated version of that same check, run
before every send instead of relying on manual review.

Two layers:
  1. PER-PICK checks -- drop any individual pick that fails an invariant
     that should always hold for a legitimate live setup. Failing picks
     are excluded from what actually gets emailed, not silently kept.
  2. SYSTEMIC check -- if too large a fraction of scored candidates fail
     the per-pick checks, that is not "a few bad picks", it's a sign
     something upstream is broken (stale data, a scoring regression). In
     that case the whole report is held and a short alert is sent instead
     of a degraded-but-plausible-looking report.
"""

import logging
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────
# Entry price should be close to CMP -- a legitimate "SETUP READY, not yet
# confirmed" card describes a pivot the stock is AT or NEAR right now, not
# one it blew through weeks ago. 5% is deliberately looser than the ~3%
# extension cap added directly to pattern_agent.py's Cup & Handle detection
# (this gate is the last line of defense for ANY pattern, present or
# future, not a duplicate of that one fix) but tight enough that it would
# have caught every one of the 13 stale picks from 2026-08-17 (smallest gap
# 3.0%, worst 8.4%).
MAX_ENTRY_CMP_DRIFT_PCT = 5.0

# If more than this fraction of scored (tier 1-3) candidates fail a
# per-pick check, treat it as systemic rather than a few bad picks -- see
# module docstring. 2026-08-17's actual incident hit ~65%; 25% is
# comfortably below that while still well above normal day-to-day noise.
SYSTEMIC_FAILURE_RATE = 0.25
MIN_CANDIDATES_FOR_SYSTEMIC_CHECK = 5   # don't fire on a thin/quiet-regime day

# [2026-08-18] Universe OHLCV coverage floor — below this fraction of the
# universe successfully fetched (any source), the scan is running on too
# incomplete a picture to trust, and should abort before scoring rather than
# silently email whatever fraction it managed to score. See
# send_data_freshness_alert_email() below and scanner.py's use of it.
MIN_OHLCV_COVERAGE_PCT = 90.0


@dataclass
class SanityReport:
    checked: int = 0
    excluded: List[dict] = field(default_factory=list)
    systemic_alert: bool = False

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    @property
    def failure_rate(self) -> float:
        return self.excluded_count / self.checked if self.checked else 0.0


def _check_pick(r) -> list:
    """Returns a list of violation strings for one StockResult; empty = OK."""
    violations = []

    price = getattr(r, "price", 0.0)
    entry = getattr(r, "entry", 0.0)
    sl    = getattr(r, "stop_loss", 0.0)
    t1    = getattr(r, "target1", 0.0)
    t2    = getattr(r, "target2", 0.0)

    if price <= 0 or entry <= 0:
        violations.append(f"missing price/entry (price={price}, entry={entry})")
        return violations   # nothing else below is checkable without these

    drift_pct = abs(price - entry) / entry * 100
    if drift_pct > MAX_ENTRY_CMP_DRIFT_PCT:
        violations.append(
            f"entry Rs.{entry:,.1f} is {drift_pct:.1f}% away from CMP Rs.{price:,.1f} "
            f"(cap {MAX_ENTRY_CMP_DRIFT_PCT}%) -- stale/extended pivot, not a live setup"
        )

    if not (sl < entry):
        violations.append(f"stop_loss Rs.{sl:,.1f} is not below entry Rs.{entry:,.1f}")
    if t1 and not (t1 > entry):
        violations.append(f"target1 Rs.{t1:,.1f} is not above entry Rs.{entry:,.1f}")
    if t2 and t1 and not (t2 >= t1):
        violations.append(f"target2 Rs.{t2:,.1f} is below target1 Rs.{t1:,.1f}")

    return violations


def run_sanity_gate(tiers: dict) -> tuple:
    """
    Checks every pick in tiers['tier1'|'tier2'|'tier3'] against _check_pick()
    and returns (clean_tiers, report).

    clean_tiers is a shallow copy of `tiers` with tier1/tier2/tier3 replaced
    by their filtered versions -- all_results and every other key pass
    through unchanged (all_results deliberately keeps failing entries, since
    it feeds validation/backtest replay, not something a user acts on
    directly -- filtering it would corrupt that pattern-performance record).
    """
    report = SanityReport()
    clean_tiers = dict(tiers)

    for tier_key in ("tier1", "tier2", "tier3"):
        original = tiers.get(tier_key, [])
        kept = []
        for r in original:
            violations = _check_pick(r)
            report.checked += 1
            if violations:
                report.excluded.append({
                    "ticker": getattr(r, "ticker", "?"),
                    "tier": tier_key,
                    "pattern": getattr(r, "pattern", "?"),
                    "violations": violations,
                })
                log.warning(
                    f"  SANITY GATE: excluded {getattr(r, 'ticker', '?')} "
                    f"({tier_key}, {getattr(r, 'pattern', '?')}) -- "
                    + "; ".join(violations)
                )
            else:
                kept.append(r)
        clean_tiers[tier_key] = kept

    if (report.checked >= MIN_CANDIDATES_FOR_SYSTEMIC_CHECK
            and report.failure_rate > SYSTEMIC_FAILURE_RATE):
        report.systemic_alert = True
        log.error(
            f"  SANITY GATE: SYSTEMIC ALERT -- {report.excluded_count}/{report.checked} "
            f"({report.failure_rate*100:.0f}%) of scored picks failed sanity checks. "
            f"Report will be HELD, not sent normally -- see send_sanity_alert_email()."
        )

    if report.excluded_count:
        log.warning(
            f"  SANITY GATE: {report.excluded_count}/{report.checked} pick(s) excluded "
            f"({report.failure_rate*100:.0f}%)."
        )
    else:
        log.info(f"  SANITY GATE: {report.checked} pick(s) checked, none excluded.")

    return clean_tiers, report


def build_sanity_alert_html(report: "SanityReport", regime: str, regime_name: str) -> str:
    rows = "".join(
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #e2e8f0'>{e['ticker']}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #e2e8f0'>{e['tier']}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #e2e8f0'>{e['pattern']}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #e2e8f0;color:#dc2626'>"
        f"{'; '.join(e['violations'])}</td></tr>"
        for e in report.excluded[:30]
    )
    return f"""<html><body style="font-family:Arial,sans-serif;background:#f8fafc;padding:20px">
<div style="max-width:800px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
  <div style="background:#dc2626;color:#ffffff;padding:20px 28px">
    <h2 style="margin:0">NSE Momentum — Report HELD: Sanity Gate Failure</h2>
    <div style="opacity:0.9;margin-top:6px">Regime {regime} ({regime_name})</div>
  </div>
  <div style="padding:20px 28px">
    <p style="color:#334155">
      <strong>{report.excluded_count} of {report.checked}</strong> scored picks
      ({report.failure_rate * 100:.0f}%) failed a basic sanity check (entry price too
      far from CMP, or SL/target ordering broken). That is above the
      {SYSTEMIC_FAILURE_RATE * 100:.0f}% threshold treated as a systemic issue rather
      than a few bad picks, so tonight's report was held instead of being sent with a
      possibly-broken majority of its content.
    </p>
    <p style="color:#334155">Check the scan logs for the root cause before re-running.</p>
    <table style="width:100%;border-collapse:collapse;margin-top:12px;font-size:13px">
      <thead><tr style="background:#f1f5f9;text-align:left">
        <th style="padding:6px 10px">Ticker</th><th style="padding:6px 10px">Tier</th>
        <th style="padding:6px 10px">Pattern</th><th style="padding:6px 10px">Why excluded</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
</body></html>"""


def send_sanity_alert_email(report: "SanityReport", regime: str, regime_name: str):
    from confirm_picks import send_email   # reuse existing SMTP sender, not duplicated
    html = build_sanity_alert_html(report, regime, regime_name)
    subject = (f"[NSE Momentum] REPORT HELD — sanity gate failure "
               f"({report.excluded_count}/{report.checked} picks)")
    send_email(subject, html)


# ─────────────────────────────────────────────────────────────────────────────
# [2026-08-18] Data-freshness abort — sibling to the sanity-gate alert above,
# but fires EARLIER and for a different reason. The sanity gate (above) only
# catches SYMPTOMS after scoring (entry drifted from CMP) — it is blind to
# uniformly stale data, since a stale entry compared against an equally
# stale CMP shows ~0% drift. This function is the abort path for the ROOT
# CAUSE checks instead: a Bhavcopy that isn't dated the expected trading
# day, or universe OHLCV coverage too low to trust. Called from scanner.py
# BEFORE the OHLCV fetch / orchestrator run / normal email — skips the rest
# of the scan entirely rather than running it on data that would fail
# validation later anyway.
# ─────────────────────────────────────────────────────────────────────────────

def build_data_freshness_alert_html(reasons: list) -> str:
    items = "".join(f"<li style='margin:6px 0;color:#dc2626'>{r}</li>" for r in reasons)
    return f"""<html><body style="font-family:Arial,sans-serif;background:#f8fafc;padding:20px">
<div style="max-width:800px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
  <div style="background:#dc2626;color:#ffffff;padding:20px 28px">
    <h2 style="margin:0">NSE Momentum — Evening Scan HELD: Data Freshness Failure</h2>
  </div>
  <div style="padding:20px 28px">
    <p style="color:#334155">
      Tonight's scan could not verify that its source data is fresh and
      complete, so <strong>no evening report was generated or sent</strong>.
      Per the non-negotiable rule this system now follows: if the pipeline
      cannot prove the data is current and valid, it must not send a
      trading email that looks correct but might not be.
    </p>
    <ul>{items}</ul>
    <p style="color:#334155">Re-run the scan once the underlying issue
      (Bhavcopy not yet published, data provider outage, etc.) is resolved.</p>
  </div>
</div>
</body></html>"""


def send_data_freshness_alert_email(reasons: list):
    from confirm_picks import send_email
    html = build_data_freshness_alert_html(reasons)
    subject = f"[NSE Momentum] EVENING SCAN HELD — data freshness verification failed ({len(reasons)} issue(s))"
    send_email(subject, html)
