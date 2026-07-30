import sqlite3
import pandas as pd

from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from hourly_scaling import BARS_PER_DAY

conn = sqlite3.connect("data/momentum_v4.db")
tickers = [r[0] for r in conn.execute(
    "SELECT DISTINCT ticker FROM price_history_hourly LIMIT 40"
).fetchall()]

found = 0
for t in tickers:
    df = pd.read_sql(
        "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
        "WHERE ticker=? ORDER BY datetime ASC", conn, params=(t,)
    )
    if len(df) < 500:
        continue
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    pa = PatternAgent(df, bars_per_day=BARS_PER_DAY)
    if pa.pattern:
        found += 1
        risk = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high,
                          universe="LARGE", bars_per_day=BARS_PER_DAY)
        ag_line = ""
        if risk.passes():
            ag = AsymmetryGate(entry=risk.entry_high, stop=risk.stop, target1=risk.target1,
                                universe="LARGE", bars_per_day=BARS_PER_DAY)
            result = ag.check_dynamic(df=df, w4_pct=4.0)
            ag_line = f" | {ag.summary(result)}"
        print(f"{t}: {pa.pattern} score={pa.raw_score} | RiskAgent passes={risk.passes()}{ag_line}")

conn.close()
print(f"\n{found}/{len(tickers)} tickers had a live pattern")
