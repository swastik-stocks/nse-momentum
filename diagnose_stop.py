import sqlite3
import pandas as pd
import numpy as np
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
from hourly_scaling import BARS_PER_DAY

conn = sqlite3.connect("data/momentum_v4.db")
df = pd.read_sql(
    "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
    "WHERE ticker='ABB.NS' ORDER BY datetime ASC", conn
)
conn.close()
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)
df.columns = ["Open", "High", "Low", "Close", "Volume"]

pa = PatternAgent(df, bars_per_day=BARS_PER_DAY)
print(f"Pattern: {pa.pattern}")

risk = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high, universe="LARGE", bars_per_day=BARS_PER_DAY)
entry = risk.entry_high
close = df["Close"].squeeze().to_numpy(dtype=float)

ema21_period = round(21 * BARS_PER_DAY)
alpha = 2 / (ema21_period + 1)
ema21 = close[0]
for c in close[1:]:
    ema21 = alpha * c + (1 - alpha) * ema21
ema21_dist_pct = (entry - ema21) / entry * 100

swing_lookback = round(5 * BARS_PER_DAY)
swing_low = float(df["Low"].squeeze().iloc[-swing_lookback:].min())
swing_low_pct = (entry - swing_low) / entry * 100

atr_period = round(14 * BARS_PER_DAY)
from agents.asymmetry_gate import _compute_atr14
atr = _compute_atr14(df, period=atr_period)
atr_pct_of_entry = atr / entry * 100

print(f"Entry: {entry:.2f}")
print(f"ATR-based component:       {atr_pct_of_entry * 1.5:.2f}%  (atr%={atr_pct_of_entry:.2f}% x 1.5 mult)")
print(f"EMA21-floor component:     {ema21_dist_pct:.2f}%")
print(f"Swing-low-floor component: {swing_low_pct:.2f}%")
print(f"Winner (max of the 3):     {max(atr_pct_of_entry*1.5, ema21_dist_pct, swing_low_pct):.2f}%")
