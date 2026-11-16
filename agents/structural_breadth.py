"""
NSE Momentum -- Structural (multi-week) market breadth
=======================================================

WHY THIS EXISTS (2026-11-17 root-cause of the 16 Nov "Correction on an up day"):
    The live breadth_score was NOT an independent signal. On GitHub runners the
    NSE live breadth API fails, so breadth comes from the bhavcopy, which never
    fills 52-week highs/lows (always 0/0). With no H/L data,
    compute_breadth_score() falls back to a pure lookup of the same single-day
    A/D ratio. RegimeClassifier then scored that one number twice (ad_pts AND
    breadth adj) and MacroAgent did the same (adv_dec pts AND breadth pts).
    One session's mood could move the regime 1-2 levels.

    Separately, every historical replay that calibrated the score gates and
    the -12 Regime D penalty (regime_backfill_deep.py) used a STRUCTURAL
    breadth proxy (% of stocks above 50-EMA), never the daily A/D. So the
    live regime was not the regime the system was validated against.

WHAT IT MEASURES (all over weeks, none from a single session):
    1. Advance rate, averaged over the last `ad_window` sessions
    2. % of the universe closing above its own 50-EMA
    3. 52-week new highs vs new lows, summed over the same `ad_window`
       sessions (only when enough exist to be meaningful; otherwise the
       component is omitted, not faked as 0.5)
    Each maps linearly to 0-10 on a fixed, UNTUNED scale (0.5 advance rate ->
    5, 50% above EMA -> 5, equal highs/lows -> 5). No thresholds are fitted
    to history, so the replay that validates it cannot be overfit.

DESIGN: one vectorised function over a wide close/high/low matrix, used by
both the live scan (last row) and validation replay (every row). Same code,
same numbers -- the validated thing is exactly the deployed thing.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AD_WINDOW = 10          # sessions averaged for the advance rate
EMA_SPAN = 50           # matches the existing above_50_ema_pct definition
HL_WINDOW = 252         # 52 weeks of sessions
HL_MIN_EVENTS = 5       # fewer combined highs+lows than this = no usable signal
MIN_STOCKS = 100        # below this a "market" breadth reading is not credible


def structural_breadth_series(close: pd.DataFrame,
                              high: Optional[pd.DataFrame] = None,
                              low: Optional[pd.DataFrame] = None,
                              member_mask: Optional[pd.DataFrame] = None,
                              ad_window: int = AD_WINDOW,
                              hl_window: int = HL_WINDOW) -> pd.DataFrame:
    """Per-date structural breadth from wide (date x ticker) price matrices.

    member_mask (same shape, bool) restricts each date to the stocks that were
    actually in the universe then -- used by the historical replay to avoid
    survivorship bias. Live passes None (stock_data already IS the universe).

    Returns columns: n_stocks, adv_rate_1d, adv_rate, above_50_pct,
    new_highs, new_lows, score (NaN where not computable).
    """
    close = close.sort_index()
    high = close if high is None else high.reindex_like(close)
    low = close if low is None else low.reindex_like(close)
    valid = close.notna()
    if member_mask is not None:
        valid &= member_mask.reindex_like(close).fillna(False).astype(bool)

    # 1. Advance rate. A stock counts only if it traded both today and the
    # previous session; unchanged closes are excluded, matching the A/D rule.
    chg = close.diff()
    traded = valid & close.shift().notna()
    adv = ((chg > 0) & traded).sum(axis=1)
    dec = ((chg < 0) & traded).sum(axis=1)
    adv_rate_1d = adv / (adv + dec).replace(0, np.nan)
    adv_rate = adv_rate_1d.rolling(ad_window, min_periods=ad_window).mean()

    # 2. % above own 50-EMA. Need EMA_SPAN bars of history before it counts,
    # otherwise a recent listing's EMA is just its first few prices.
    ema = close.ewm(span=EMA_SPAN, adjust=False, min_periods=EMA_SPAN).mean()
    has_ema = valid & ema.notna()
    above_50_pct = ((close > ema) & has_ema).sum(axis=1) / has_ema.sum(axis=1).replace(0, np.nan) * 100

    # 3. 52-week highs/lows against the PRIOR window (today excluded), with a
    # full window required so a short history can't manufacture "new highs".
    prior_hi = high.shift().rolling(hl_window, min_periods=hl_window).max()
    prior_lo = low.shift().rolling(hl_window, min_periods=hl_window).min()
    new_highs = ((high > prior_hi) & valid).sum(axis=1)
    new_lows = ((low < prior_lo) & valid).sum(axis=1)
    # Scored on the same AD_WINDOW sum, not today's count: a single-day H/L
    # tally swings from all-highs to none on one bad session, which would
    # reintroduce the one-day sensitivity this module exists to remove.
    highs_w = new_highs.rolling(ad_window, min_periods=ad_window).sum()
    lows_w = new_lows.rolling(ad_window, min_periods=ad_window).sum()

    out = pd.DataFrame({
        "n_stocks": valid.sum(axis=1),
        "adv_rate_1d": adv_rate_1d,
        "adv_rate": adv_rate,
        "above_50_pct": above_50_pct,
        "new_highs": new_highs,          # today's count, for display
        "new_lows": new_lows,
        "new_highs_w": highs_w,          # window sums, used in the score
        "new_lows_w": lows_w,
    })

    c_ad = out["adv_rate"] * 10
    c_50 = out["above_50_pct"] / 10
    hl_total = highs_w + lows_w
    c_hl = (highs_w / hl_total.replace(0, np.nan) * 10).where(hl_total >= HL_MIN_EVENTS)

    comps = pd.concat([c_ad, c_50, c_hl], axis=1)
    score = comps.mean(axis=1, skipna=True)
    # The two always-available components are required; H/L is optional.
    score = score.where(c_ad.notna() & c_50.notna() & (out["n_stocks"] >= MIN_STOCKS))
    out["score"] = score.clip(0, 10).round(2)
    return out


def _wide(stock_data: dict, col: str) -> pd.DataFrame:
    series = {}
    for ticker, df in stock_data.items():
        if df is None or getattr(df, "empty", True) or col not in df.columns:
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):      # yfinance multi-index leftovers
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:     # mixed tz-aware/naive sources must align on date
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        series[ticker] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(series).sort_index() if series else pd.DataFrame()


def compute_structural_breadth(stock_data: dict) -> Optional[dict]:
    """Live entry point: structural breadth as of the latest session in stock_data.

    Returns None (never raises) when it can't produce a credible reading; the
    caller then treats breadth as NEUTRAL rather than reusing the daily A/D.
    """
    try:
        close = _wide(stock_data, "Close")
        if close.empty:
            return None
        # Only the trailing year-plus matters; trimming keeps the live call fast.
        close = close.iloc[-(HL_WINDOW + EMA_SPAN + 5):]
        high = _wide(stock_data, "High").reindex_like(close)
        low = _wide(stock_data, "Low").reindex_like(close)
        row = structural_breadth_series(close, high, low).iloc[-1]
        if pd.isna(row["score"]):
            log.warning(f"Structural breadth not computable (n_stocks={int(row['n_stocks'])}) "
                        f"-- regime breadth input will be neutral.")
            return None
        result = {
            "score": float(row["score"]),
            "adv_rate": round(float(row["adv_rate"]), 4),
            "above_50_pct": round(float(row["above_50_pct"]), 1),
            "new_highs": int(row["new_highs"]),
            "new_lows": int(row["new_lows"]),
            "n_stocks": int(row["n_stocks"]),
            "as_of": close.index[-1].date().isoformat(),
            "source": "UNIVERSE_STRUCTURAL",
        }
        log.info(f"Structural breadth {result['score']:.1f}/10 | {AD_WINDOW}d adv rate "
                 f"{result['adv_rate']:.2f} | above 50-EMA {result['above_50_pct']:.1f}% | "
                 f"52w H/L {result['new_highs']}/{result['new_lows']} | n={result['n_stocks']}")
        return result
    except Exception as e:  # breadth is an input, never a reason to kill the scan
        log.warning(f"Structural breadth failed ({type(e).__name__}: {e}) -- using neutral.")
        return None
