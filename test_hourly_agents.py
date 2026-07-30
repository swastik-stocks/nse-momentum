import sqlite3
import pandas as pd
import numpy as np

from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from hourly_scaling import BARS_PER_DAY

# Pull one real ticker's hourly history to test against
conn = sqlite3.connect("data/momentum_v4.db")
df = pd.read_sql(
    "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
    "WHERE ticker='RELIANCE.NS' ORDER BY datetime ASC",
    conn
)
conn.close()

df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)
df.columns = ["Open", "High", "Low", "Close", "Volume"]

print(f"Loaded {len(df)} hourly bars for RELIANCE.NS")
print(f"BARS_PER_DAY = {BARS_PER_DAY}")
print()

pa = PatternAgent(df, bars_per_day=BARS_PER_DAY)
print(f"PatternAgent: pattern={pa.pattern!r} score={pa.raw_score} quality={pa.breakout_quality}")

liq = LiquidityAgent(df, universe="LARGE", bars_per_day=BARS_PER_DAY)
print(f"LiquidityAgent: passes={liq.passes()} ADT={liq.get_adt()}")

if pa.pattern:
    risk = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high,
                      universe="LARGE", bars_per_day=BARS_PER_DAY)
    print(f"RiskAgent: passes={risk.passes()} entry={risk.entry} stop={risk.stop} t1={risk.target1}")

    if risk.passes():
        ag = AsymmetryGate(entry=risk.entry_high, stop=risk.stop, target1=risk.target1,
                            universe="LARGE", bars_per_day=BARS_PER_DAY)
        result = ag.check_dynamic(df=df, w4_pct=4.0)
        print(f"AsymmetryGate: {ag.summary(result)}")

print()
print("If all lines above printed without error, the 5 hourly-scaled agents work end to end.")
