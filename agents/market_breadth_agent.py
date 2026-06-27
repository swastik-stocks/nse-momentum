"""
NSE Momentum v5.1 — MarketBreadthAgent
======================================
BUG FIXES (Jun 2026):
  1. A/D ratio now sourced from NSE-wide market stats (all ~2000 traded
     symbols), NOT from our 401/500 stock universe. Old code counted
     advances/declines only in our portfolio universe which skews heavily
     on days where smallcap/microcap broad market moves differ from
     large/mid indices.

  2. Sanity-check: only flags EXTREME data errors (A/D < 0.15 in bull
     structure, or A/D > 4.0 in bear structure). Normal pullback days
     (e.g. 63% above 50EMA with A/D=0.34) are NOT flagged — that is
     just a down day in an uptrend, which is completely normal.

  3. Calibration log: every run dumps raw inputs to
     logs/breadth_calibration.log so we can audit mismatches.

  4. VIX check removed from this agent (moved to macro_agent.py only
     to avoid double-counting).

Sources (all free, no login required):
  - A/D, New Highs/Lows: NSE market stats endpoint
    https://www.nseindia.com/market-data/live-market-statistics
    (scraped via requests + JSON parse — same session as existing NSE calls)
  - Above-50-EMA: computed internally from our 401-stock universe price DB
    (this is an internal metric, not NSE-reported — universe-bounded is OK)
"""

import logging
import os
import json
import requests
from datetime import date, datetime

log = logging.getLogger(__name__)

# ── calibration log path ──────────────────────────────────────────────────────
_CAL_LOG = os.path.join(os.path.dirname(__file__), '..', 'logs', 'breadth_calibration.log')

# NSE headers — same as existing NSE calls in the project
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Session with cookies (NSE requires cookie from homepage first)
_nse_session = None


def _get_nse_session() -> requests.Session:
    global _nse_session
    if _nse_session is None:
        s = requests.Session()
        s.headers.update(_NSE_HEADERS)
        try:
            # Warm up the session — NSE needs the cookie from the homepage
            s.get("https://www.nseindia.com/", timeout=10)
        except Exception as e:
            log.warning(f"NSE session warmup failed: {e}")
        _nse_session = s
    return _nse_session


def fetch_nse_wide_breadth() -> dict:
    """
    Fetch NSE-WIDE advances/declines/unchanged and 52-week highs/lows.

    Returns dict with keys:
        advances, declines, unchanged, new_highs, new_lows,
        total_traded, source, fetch_time

    On any failure, returns None so caller can fall back to Bhavcopy method.
    """
    s = _get_nse_session()

    # Primary endpoint: NSE market statistics (JSON)
    try:
        resp = s.get(
            "https://www.nseindia.com/api/market-data-pre-open?key=NIFTY",
            timeout=10
        )
        # This endpoint gives advance/decline for the full NSE market
        # Field structure: marketStatus.marketSummary -> advances/declines
        data = resp.json()
        summary = data.get("marketStatus", {}).get("marketSummary", {})

        advances  = int(summary.get("advances",  0))
        declines  = int(summary.get("declines",  0))
        unchanged = int(summary.get("unchanged", 0))

        if advances + declines > 500:  # sanity: must be market-wide
            log.info(
                f"NSE-wide breadth (live): ADV={advances} DEC={declines} "
                f"UNC={unchanged} (total={advances+declines+unchanged})"
            )
            # Highs/lows from a separate endpoint
            new_highs, new_lows = _fetch_nse_highs_lows(s)
            return {
                "advances":    advances,
                "declines":    declines,
                "unchanged":   unchanged,
                "new_highs":   new_highs,
                "new_lows":    new_lows,
                "total_traded": advances + declines + unchanged,
                "source":      "NSE_LIVE_API",
                "fetch_time":  datetime.now().isoformat(),
            }
        else:
            log.warning(
                f"NSE live API returned suspiciously small breadth totals "
                f"(ADV={advances}, DEC={declines}). Falling back to Bhavcopy."
            )
            return None

    except Exception as e:
        log.warning(f"NSE live breadth fetch failed: {e}. Falling back.")
        return None


def _fetch_nse_highs_lows(session: requests.Session) -> tuple:
    """Fetch 52-week new highs and lows from NSE."""
    try:
        resp = session.get(
            "https://www.nseindia.com/api/liveanalysis/high-low",
            timeout=8
        )
        data = resp.json()
        highs = len(data.get("data", {}).get("high52", []))
        lows  = len(data.get("data", {}).get("low52",  []))
        return highs, lows
    except Exception as e:
        log.warning(f"52w high/low fetch failed: {e}. Returning 0,0.")
        return 0, 0


def fetch_breadth_from_bhavcopy(bhavcopy_df) -> dict:
    """
    Fallback: compute advances/declines from Bhavcopy DataFrame.
    Bhavcopy covers ~3200 symbols (full NSE universe) — use all of them,
    NOT filtered to our 401-stock universe.

    bhavcopy_df must have columns: SYMBOL, CLOSE, PREVCLOSE (or PCLOSE).
    """
    if bhavcopy_df is None or bhavcopy_df.empty:
        log.error("Bhavcopy DataFrame is empty — cannot compute breadth.")
        return None

    df = bhavcopy_df.copy()

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    # NSE Bhavcopy uses CLOSE_PRICE not CLOSE — search both variants
    close_col = next(
        (c for c in df.columns if c in ("CLOSE_PRICE", "CLOSE", "CLOSING_PRICE")),
        None
    )
    prev_col = next(
        (c for c in df.columns if c in ("PREV_CLOSE", "PREVCLOSE", "PCLOSE",
                                         "PREV_CLOSING_PRICE")),
        None
    )

    if not close_col or not prev_col:
        log.error(f"Bhavcopy missing CLOSE_PRICE/PREV_CLOSE. Got: {list(df.columns)}")
        return None

    df[close_col] = df[close_col].astype(float)
    df[prev_col]  = df[prev_col].astype(float)

    # Use ALL symbols in Bhavcopy (market-wide), not our universe subset
    mask_adv = df[close_col] > df[prev_col]
    mask_dec = df[close_col] < df[prev_col]

    advances  = int(mask_adv.sum())
    declines  = int(mask_dec.sum())
    unchanged = int((~mask_adv & ~mask_dec).sum())
    total     = advances + declines + unchanged

    advance_rate = advances / (advances + declines) if (advances + declines) > 0 else 0.5
    log.info(
        f"Bhavcopy breadth (NSE-wide, {total} symbols): "
        f"ADV={advances} DEC={declines} UNC={unchanged} "
        f"advance_rate={advance_rate:.3f} ({advances/(declines or 1):.2f}:1)"
    )

    return {
        "advances":    advances,
        "declines":    declines,
        "unchanged":   unchanged,
        "new_highs":   0,  # Bhavcopy fallback doesn't track 52w H/L
        "new_lows":    0,
        "total_traded": total,
        "source":      "BHAVCOPY_FULL",
        "fetch_time":  datetime.now().isoformat(),
    }


def compute_breadth_score(
    breadth_data: dict,
    above_50_ema_pct: float = None,
    above_200_ema_pct: float = None,
) -> dict:
    """
    Compute final breadth score (0–10) from fetched market data.

    Args:
        breadth_data:       Output of fetch_nse_wide_breadth() or
                            fetch_breadth_from_bhavcopy()
        above_50_ema_pct:   % of our 401-stock universe above 50 EMA
                            (computed internally from price DB)
        above_200_ema_pct:  % above 200 EMA (optional)

    Returns dict with:
        breadth_score (0–10), ad_ratio, nh_nl_ratio,
        above_50_ema_pct, regime, confidence, warnings
    """
    if breadth_data is None:
        log.error("compute_breadth_score called with None breadth_data.")
        return _default_neutral_breadth("NO_DATA")

    advances  = breadth_data.get("advances",  0)
    declines  = breadth_data.get("declines",  0)
    new_highs = breadth_data.get("new_highs", 0)
    new_lows  = breadth_data.get("new_lows",  0)
    source    = breadth_data.get("source",    "UNKNOWN")

    total_active = advances + declines
    if total_active == 0:
        log.error("Both advances and declines are 0 — data fetch likely failed.")
        return _default_neutral_breadth("ZERO_DATA")

    # A/D ratio = advances / declines (industry standard: how many advanced per decline)
    # e.g. 1092 adv / 2088 dec = 0.523 (mild down day)
    # NOT: advances / total (that is "advance rate", a different metric = 0.343)
    ad_ratio      = round(advances / max(declines, 1), 4)
    advance_rate  = round(advances / total_active, 4)   # kept for reference only

    total_hl   = new_highs + new_lows
    nh_nl_ratio = round(new_highs / total_hl, 4) if total_hl > 0 else 0.5

    # ── Core breadth score (0–10) ─────────────────────────────────────────────
    # A/D ratio is centered at 1.0 (equal advances and declines = neutral).
    # Linear mapping (ad_ratio * 5) is wrong because it treats 0.5 as "half"
    # when 0.5 actually means "for every 2 stocks that fell, 1 rose" — mild down day.
    # Use a proper lookup table calibrated to NSE historical breadth.
    def _ad_to_score(r: float) -> float:
        if r >= 4.0:  return 9.5   # surge: >4:1 advance ratio
        if r >= 3.0:  return 9.0
        if r >= 2.5:  return 8.0
        if r >= 2.0:  return 7.5   # strong bull day
        if r >= 1.5:  return 6.5
        if r >= 1.2:  return 5.5
        if r >= 1.0:  return 5.0   # neutral: equal adv/dec
        if r >= 0.75: return 4.5   # mild down day (A/D=0.52 → here)
        if r >= 0.5:  return 4.0
        if r >= 0.33: return 3.0   # weak: 1 adv for every 3 dec
        if r >= 0.25: return 2.0
        if r >= 0.15: return 1.5
        return 1.0                 # crash: < 1 in 6 stocks advanced

    ad_score = _ad_to_score(ad_ratio)
    if source == "BHAVCOPY_FULL" or nh_nl_ratio == 0.5:
        breadth_score = round(ad_score, 2)   # pure A/D when no H/L data
    else:
        breadth_score = round(ad_score * 0.60 + nh_nl_ratio * 10 * 0.40, 2)

    breadth_score = max(0.0, min(10.0, breadth_score))

    # ── Sanity check: internal consistency ───────────────────────────────────
    warnings    = []
    confidence  = "HIGH"
    needs_refetch = False

    if above_50_ema_pct is not None:
        # ── IMPORTANT: above_50_ema and daily A/D measure DIFFERENT things ──
        # above_50_ema = structural (weeks/months): % of stocks in uptrend
        # A/D ratio    = daily (one session): how many stocks rose vs fell TODAY
        #
        # A market where 63% of stocks are above 50 EMA CAN have A/D = 0.34.
        # That simply means "the trend is up, but today was a pullback day."
        # This is COMPLETELY NORMAL and must NOT be flagged as a contradiction.
        #
        # Only flag EXTREME divergences that suggest actual data corruption:
        #   Bull structure (>60% above 50EMA) + CRASH-level decline (A/D < 0.15)
        #   Bear structure (<30% above 50EMA) + SURGE-level advance (A/D > 4.0)
        # ──────────────────────────────────────────────────────────────────────

        # Log normal divergences as INFO only (not warnings)
        if above_50_ema_pct > 55 and ad_ratio < 0.50:
            log.info(
                f"Breadth note: above_50_ema={above_50_ema_pct:.1f}% (bullish structure) "
                f"but A/D={ad_ratio:.3f} (down day). "
                f"Normal pullback in uptrend — not a contradiction."
            )
        elif above_50_ema_pct < 45 and ad_ratio > 1.5:
            log.info(
                f"Breadth note: above_50_ema={above_50_ema_pct:.1f}% (bearish structure) "
                f"but A/D={ad_ratio:.3f} (up day). "
                f"Normal bounce in downtrend — not a contradiction."
            )

        # Only flag TRUE data errors: crash-level daily move vs structural trend
        if above_50_ema_pct > 60 and ad_ratio < 0.15:
            msg = (
                f"SANITY FAIL: above_50_ema={above_50_ema_pct:.1f}% "
                f"but A/D={ad_ratio:.3f} (crash-level decline in bull structure). "
                f"Likely Bhavcopy date mismatch or data corruption. Source={source}."
            )
            log.warning(msg)
            warnings.append(msg)
            confidence    = "LOW"
            needs_refetch = True

        if above_50_ema_pct < 30 and ad_ratio > 4.0:
            msg = (
                f"SANITY FAIL: above_50_ema={above_50_ema_pct:.1f}% "
                f"but A/D={ad_ratio:.3f} (surge-level advance in bear structure). "
                f"Likely Bhavcopy date mismatch or data corruption. Source={source}."
            )
            log.warning(msg)
            warnings.append(msg)
            confidence = "LOW"

        # Also flag if breadth data was clearly not computed (zeros with non-zero claim)
        if advances == 0 and declines == 0 and source != "DEFAULT":
            msg = f"SANITY FAIL: Advances=0, Declines=0 but source={source}. Breadth not computed."
            log.warning(msg)
            warnings.append(msg)
            confidence = "LOW"

    # ── Regime mapping ────────────────────────────────────────────────────────
    if breadth_score >= 7.5:
        regime = "BULL"
        regime_label = "Bullish breadth — internals healthy"
    elif breadth_score >= 5.5:
        regime = "NEUTRAL"
        regime_label = "Neutral breadth"
    elif breadth_score >= 3.5:
        regime = "WEAK"
        regime_label = "Weakening breadth — reduce exposure"
    else:
        regime = "BEAR"
        regime_label = "Bearish breadth — defensive mode"

    # If confidence is LOW, cap the regime at WEAK (don't call BEAR on bad data)
    if confidence == "LOW" and regime == "BEAR":
        regime       = "WEAK"
        regime_label = "Weakening breadth — LOW CONFIDENCE (sanity check failed)"
        log.warning("Regime capped at WEAK due to LOW_CONFIDENCE breadth reading.")

    # ── Max position size ─────────────────────────────────────────────────────
    max_position = {
        "BULL":    "2R",
        "NEUTRAL": "1R",
        "WEAK":    "0.5R",
        "BEAR":    "0R (No new longs)",
    }[regime]

    result = {
        "date":            date.today().isoformat(),
        "advances":        advances,
        "declines":        declines,
        "new_highs":       new_highs,
        "new_lows":        new_lows,
        "ad_ratio":        ad_ratio,        # advances/declines e.g. 0.523
        "advance_rate":    advance_rate,    # advances/total e.g. 0.343 (for reference)
        "nh_nl_ratio":     nh_nl_ratio,
        "above_50_ema_pct": above_50_ema_pct,
        "breadth_score":   breadth_score,
        "regime":          regime,
        "regime_label":    regime_label,
        "max_position":    max_position,
        "confidence":      confidence,
        "source":          source,
        "warnings":        warnings,
        "needs_refetch":   needs_refetch,
    }

    # ── Write calibration log ─────────────────────────────────────────────────
    _write_calibration_log(result)

    return result


def _default_neutral_breadth(reason: str) -> dict:
    """Return a neutral breadth dict when data is unavailable."""
    log.warning(f"Using default neutral breadth. Reason: {reason}")
    return {
        "date":            date.today().isoformat(),
        "advances":        0,
        "declines":        0,
        "new_highs":       0,
        "new_lows":        0,
        "ad_ratio":        1.0,
        "nh_nl_ratio":     0.5,
        "above_50_ema_pct": None,
        "breadth_score":   5.0,
        "regime":          "NEUTRAL",
        "regime_label":    f"Neutral (default — {reason})",
        "max_position":    "1R",
        "confidence":      "LOW",
        "source":          "DEFAULT",
        "warnings":        [f"Breadth defaulted to NEUTRAL: {reason}"],
        "needs_refetch":   True,
    }


def _write_calibration_log(result: dict):
    """Append a JSON line to the calibration log for audit."""
    try:
        os.makedirs(os.path.dirname(_CAL_LOG), exist_ok=True)
        entry = {
            "ts":               datetime.now().isoformat(),
            "date":             result["date"],
            "advances":         result["advances"],
            "declines":         result["declines"],
            "ad_ratio":         result["ad_ratio"],
            "above_50_ema_pct": result["above_50_ema_pct"],
            "breadth_score":    result["breadth_score"],
            "regime":           result["regime"],
            "confidence":       result["confidence"],
            "source":           result["source"],
            "warnings":         result["warnings"],
        }
        with open(_CAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning(f"Could not write calibration log: {e}")


def print_breadth_dashboard(result: dict):
    """Pretty-print the breadth snapshot to stdout."""
    conf_indicator = "✅" if result["confidence"] == "HIGH" else "⚠️ LOW CONFIDENCE"
    warn_block = ""
    if result.get("warnings"):
        warn_block = "\n║  ⚠️  " + "\n║  ⚠️  ".join(result["warnings"])

    above_50 = (
        f"{result['above_50_ema_pct']:.1f}%"
        if result.get("above_50_ema_pct") is not None
        else "N/A"
    )

    print(f"""
╔══════════════════════════════════════════════╗
║         MARKET BREADTH SNAPSHOT              ║
╠══════════════════════════════════════════════╣
║  Date        : {result['date']}
║  Source      : {result['source']}
║  Advances    : {result['advances']:>6,}
║  Declines    : {result['declines']:>6,}
║  A/D Ratio   : {result['ad_ratio']:.3f}  (adv/dec)
║  Adv Rate    : {result.get('advance_rate', 0):.3f}  (adv/total)
║  New Highs   : {result['new_highs']:>6,}
║  New Lows    : {result['new_lows']:>6,}
║  Above 50EMA : {above_50}
║  Breadth     : {result['breadth_score']:.1f} / 10
║  Regime      : {result['regime']} — {result['regime_label']}
║  Max Size    : {result['max_position']}
║  Confidence  : {conf_indicator}{warn_block}
╚══════════════════════════════════════════════╝""")
