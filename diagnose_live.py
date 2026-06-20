"""
Shows exactly what the new RiskAgent computes for 10 real stocks.
Run from: C:\Users\User\Desktop\nse_momentum\
"""
import sys, warnings
warnings.filterwarnings('ignore')

from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
import numpy as np
import pandas as pd

tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'RELIANCE.NS',
    'SBIN.NS', 'AXISBANK.NS', 'LT.NS',
    'BAJFINANCE.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'MARUTI.NS'
]

print("Fetching data...")
dfs = fetch_batch_ohlcv(tickers)
print(f"Loaded {len(dfs)} stocks\n")

print(f"{'Ticker':<16} {'Entry':>8} {'Stop':>8} {'Stop%':>6} {'T1':>8} {'Reward%':>8} {'RRR':>5} {'Result'}")
print("-" * 85)

for t in tickers:
    df = dfs.get(t, pd.DataFrame())
    if df.empty:
        print(f"{t:<16} -- no data")
        continue

    price = float(df['Close'].iloc[-1])
    low   = df['Low'].squeeze().to_numpy(dtype=float)

    pa = PatternAgent(df)
    bo = pa.breakout_level if pa.breakout_level > 0 else price

    ra = RiskAgent(df, bo,
                   pa.entry_low  or price * 0.99,
                   pa.entry_high or price * 1.01,
                   universe='LARGE')

    if ra.passes():
        stop_pct   = round((ra.entry - ra.stop)   / ra.entry * 100, 2)
        reward_pct = round((ra.target1 - ra.entry) / ra.entry * 100, 2)
        result = f"PASS rr={ra.rrr}x"
    else:
        # Compute what stop WOULD be to show the problem
        ema21    = float(pd.Series(df['Close'].squeeze()).ewm(span=21).mean().iloc[-1])
        stop_ema = ema21 * 0.993
        stop_10d = float(np.min(low[-10:])) * 0.997
        entry    = ra.entry_high or price * 1.01
        stop_used = max([s for s in [stop_ema, stop_10d] if 0 < s < entry] or [entry*0.95])
        stop_pct   = round((entry - stop_used) / entry * 100, 2)
        reward_pct = 0.0
        result = f"FAIL: {ra.reject_reason()}"

    print(f"{t:<16} {ra.entry:>8.2f} {ra.stop:>8.2f} {stop_pct:>5.1f}% "
          f"{ra.target1:>8.2f} {reward_pct:>7.1f}%  {ra.rrr:>4.2f}  {result}")

print()
print("KEY: If Stop% > 2%, AsymmetryGate will reject even if RiskAgent passes.")
print("     If Result=FAIL with R:R reason, RiskAgent internal gate is too tight.")
print("     If Stop% < 2% but Result=FAIL, target calculation is the issue.")
