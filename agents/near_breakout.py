"""
NSE Momentum v5.0 - Near Breakout Scanner
==========================================
Finds stocks within 3% of their breakout level.
These haven't triggered yet — they are WATCHLIST entries.
Shows up in email as "Stocks Approaching Breakout - Set Alerts"
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NEAR_BREAKOUT_PCT = 3.0   # within 3% of breakout level


def find_near_breakout_stocks(universe_items: list,
                               stock_data: dict,
                               delivery_data: dict,
                               existing_tickers: set) -> list:
    """
    Called after main scoring loop in run_universe().
    Returns list of dicts for email display — top 10 closest to breakout.
    """
    from agents.pattern_agent  import PatternAgent
    from agents.liquidity_agent import LiquidityAgent

    near_breakout = []

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

            price    = float(df["Close"].iloc[-1])
            breakout = pa.breakout_level

            # Must be BELOW breakout
            if price >= breakout:
                continue

            gap_pct = (breakout - price) / price * 100
            if gap_pct > NEAR_BREAKOUT_PCT:
                continue

            # Uptrend structure required
            close = df["Close"].squeeze()
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            if not (price > ema20 > ema50):
                continue

            # RSI 50-72
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

            low      = df["Low"].squeeze().to_numpy(dtype=float)
            stop_10d = float(np.min(low[-10:])) * 0.997
            stop_pct = round((breakout - stop_10d) / breakout * 100, 1)

            vol    = df["Volume"].squeeze().to_numpy(dtype=float)
            avg20v = float(np.mean(vol[-20:])) if len(vol) >= 20 else 1
            rvol   = round(float(vol[-1]) / avg20v, 2) if avg20v > 0 else 0

            near_breakout.append({
                "ticker":   ticker,
                "name":     name,
                "sector":   sector,
                "universe": universe,
                "price":    round(price, 2),
                "breakout": round(breakout, 2),
                "gap_pct":  round(gap_pct, 1),
                "pattern":  pa.pattern,
                "rsi":      round(rsi, 1),
                "rvol":     rvol,
                "stop_10d": round(stop_10d, 2),
                "stop_pct": stop_pct,
            })

        except Exception as e:
            log.debug("near_breakout error %s: %s", ticker, e)
            continue

    near_breakout.sort(key=lambda x: x["gap_pct"])
    return near_breakout[:10]
