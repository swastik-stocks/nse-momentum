"""
NSE Momentum v5.4 - Near Breakout Scanner

BUG-6 FIX: CMP now sourced from bhavcopy_cmp_map (NSE official close)
instead of df["Close"].iloc[-1] (stale Yahoo cache).

The old code used price = float(df["Close"].iloc[-1]) which was the same
stale Yahoo price causing the 5.8% POLYCAB error. The gap_pct calculation
(breakout - price) / price was therefore wrong for every stock in the
near-breakout watchlist.

Fix: find_near_breakout_stocks() now accepts bhavcopy_cmp_map parameter.
If the stock exists in the map, use that as price. Fall back to df close
only if Bhavcopy doesn't have the stock (rare — Bhavcopy covers ~3200 NSE stocks).
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NEAR_BREAKOUT_PCT = 3.0   # within 3% of breakout level


def find_near_breakout_stocks(
    universe_items: list,
    stock_data: dict,
    delivery_data: dict,
    existing_tickers: set,
    bhavcopy_cmp_map: dict = None,      # BUG-6 FIX: official CMP from Bhavcopy
) -> list:
    """
    Called after main scoring loop in run_universe().
    Returns list of dicts for email display — top 10 closest to breakout.

    Args:
        bhavcopy_cmp_map: {TICKER.NS: close_price} from BhavcopyFetcher.
                          If provided, used as authoritative CMP.
                          If None (e.g. Bhavcopy unavailable), falls back
                          to df["Close"].iloc[-1] with a warning logged once.
    """
    from agents.pattern_agent   import PatternAgent
    from agents.liquidity_agent import LiquidityAgent

    cmp_map        = bhavcopy_cmp_map or {}
    warned_no_bhav = False
    near_breakout  = []
    cmp_source_counts = {"bhavcopy": 0, "cache_fallback": 0}

    for item in universe_items:
        ticker, name, sector, universe = item

        if ticker in existing_tickers:
            continue

        df = stock_data.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 50:
            continue

        try:
            liq = LiquidityAgent(df, universe=universe)
            if not liq.passes():
                continue

            pa = PatternAgent(df)
            if not pa.pattern or pa.breakout_level <= 0:
                continue

            # BUG-6 FIX: use Bhavcopy official close as CMP
            if ticker in cmp_map:
                price = cmp_map[ticker]
                cmp_source_counts["bhavcopy"] += 1
            else:
                # Fallback to cache — log once so we know it happened
                if not warned_no_bhav and not cmp_map:
                    log.warning(
                        "near_breakout: bhavcopy_cmp_map empty — "
                        "using stale cache prices. Gap % may be inaccurate."
                    )
                    warned_no_bhav = True
                price = float(df["Close"].iloc[-1])
                cmp_source_counts["cache_fallback"] += 1

            breakout = pa.breakout_level

            # Must be BELOW breakout (not already broken out)
            if price >= breakout:
                continue

            gap_pct = (breakout - price) / price * 100
            if gap_pct > NEAR_BREAKOUT_PCT:
                continue

            # Uptrend structure required: price > EMA20 > EMA50
            close = df["Close"].squeeze()
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            if not (price > ema20 > ema50):
                continue

            # RSI 50–72 (approaching breakout with momentum, not overbought)
            try:
                import ta
                rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
            except Exception:
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss  = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                rsi   = 100 - 100 / (1 + gain / loss) if loss != 0 else 70.0

            if not (50 <= rsi <= 72):
                continue

            # Stop = 10-day low * 0.997
            low      = df["Low"].squeeze().to_numpy(dtype=float)
            stop_10d = float(np.min(low[-10:])) * 0.997
            stop_pct = round((breakout - stop_10d) / breakout * 100, 1)

            # RVOL
            vol    = df["Volume"].squeeze().to_numpy(dtype=float)
            avg20v = float(np.mean(vol[-20:])) if len(vol) >= 20 else 1
            rvol   = round(float(vol[-1]) / avg20v, 2) if avg20v > 0 else 0

            near_breakout.append({
                "ticker":     ticker,
                "name":       name,
                "sector":     sector,
                "universe":   universe,
                "price":      round(price, 2),
                "breakout":   round(breakout, 2),
                "gap_pct":    round(gap_pct, 1),
                "pattern":    pa.pattern,
                "rsi":        round(rsi, 1),
                "rvol":       rvol,
                "stop_10d":   round(stop_10d, 2),
                "stop_pct":   stop_pct,
                "cmp_source": "bhavcopy" if ticker in cmp_map else "cache",
            })

        except Exception as e:
            log.debug("near_breakout error %s: %s", ticker, e)
            continue

    if near_breakout:
        log.info(
            f"  near_breakout: {len(near_breakout)} candidates | "
            f"CMP from bhavcopy={cmp_source_counts['bhavcopy']} "
            f"cache_fallback={cmp_source_counts['cache_fallback']}"
        )

    near_breakout.sort(key=lambda x: x["gap_pct"])
    return near_breakout[:10]
