import warnings, pandas as pd
warnings.filterwarnings('ignore')

from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate

df = fetch_batch_ohlcv(['ANGELONE.NS'])['ANGELONE.NS']
pa = PatternAgent(df)
price = float(df['Close'].iloc[-1])

ra = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high, universe='MID')

print("=== ANGELONE.NS Pipeline Test ===")
print("Pattern    :", pa.pattern)
print("Breakout   :", pa.breakout_level)
print("Entry      :", ra.entry)
print("Stop       :", ra.stop)
print("Target1    :", ra.target1)
print("Stop%      :", ra.stop_pct)
print("Gain%      :", ra.gain_pct)
print("RRR        :", ra.rrr)
print("RiskAgent  :", "PASS" if ra.passes() else "FAIL - " + ra.reject_reason())

if ra.passes():
    ag = AsymmetryGate(ra.entry, ra.stop, ra.target1, 'MID')
    res = ag.check()
    qualified = res["qualified"]
    print("Asymmetry  :", "PASS" if qualified else "FAIL - " + res["fail_reason"])
    print("Risk%      :", res["risk_pct"])
    print("Reward%    :", res["reward_pct"])
    print("R:R        :", res["rr_ratio"])
