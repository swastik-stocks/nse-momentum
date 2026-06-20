import warnings
warnings.filterwarnings('ignore')
import sys

from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from agents.liquidity_agent import LiquidityAgent
import pandas as pd
import numpy as np

TARGET = 'POLYCAB.NS'
print("Fetching POLYCAB only...")
dfs = fetch_batch_ohlcv([TARGET])
df = dfs.get(TARGET, pd.DataFrame())

if df is None or df.empty:
    print("NO DATA"); sys.exit()

price = float(df['Close'].iloc[-1])
print(f"Price: {price:.2f} | Bars: {len(df)}")

universe_tag = "LARGE"
for item in NSE_UNIVERSE:
    if item[0] == TARGET:
        universe_tag = item[3]
        break
print(f"Universe: {universe_tag}")
print()

# G1
liq = LiquidityAgent(df, universe=universe_tag)
print(f"G1 Liquidity:  {'PASS' if liq.passes() else 'FAIL - ' + liq.reject_reason()}")

# G2
pa = PatternAgent(df)
print(f"G2 Pattern:    {pa.pattern or 'NONE'} | breakout={pa.breakout_level:.0f}")
if not pa.pattern:
    print("BLOCKED G2"); sys.exit()

# G3 - skip real RS, just show what range it would be
print(f"G3 RS:         SKIP (needs full universe) - assume passes")

# G4
ra = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high, universe=universe_tag)
print(f"G4 RiskAgent:  {'PASS' if ra.passes() else 'FAIL - ' + ra.reject_reason()}")
if ra.passes():
    print(f"   entry={ra.entry:.2f} stop={ra.stop:.2f} stop%={ra.stop_pct} T1={ra.target1:.2f}")
if not ra.passes():
    print("BLOCKED G4"); sys.exit()

# G5
ag = AsymmetryGate(ra.entry, ra.stop, ra.target1, universe_tag)
res = ag.check()
print(f"G5 Asymmetry:  {'PASS' if res['qualified'] else 'FAIL'}")
print(f"   risk={res['risk_pct']}% reward={res['reward_pct']}% RR={res['rr_ratio']}x")
if not res['qualified']:
    print(f"   REASON: {res['fail_reason']}")
    print("BLOCKED G5"); sys.exit()

# G6
vcpg = VCPContractionGate(df=df)
vcp = vcpg.check()
print(f"G6 VCPGate:    hard_reject={vcp['hard_reject']} w4={vcp['w4_pct']}% penalty={vcp['penalty']}")
if vcp['hard_reject']:
    print(f"   REASON: {vcp['fail_reason']}")
    print("BLOCKED G6 - THIS IS WHY POLYCAB IS MISSING")
    sys.exit()

# G7
headroom = round((ra.target1 - ra.entry) / ra.entry * 100, 2) if ra.entry > 0 else 0
print(f"G7 Headroom:   {headroom}%  {'PASS' if headroom >= 4.5 else 'FAIL < 4.5%'}")
if headroom < 4.5:
    print("BLOCKED G7"); sys.exit()

# Show 10-day low for context
low = df['Low'].squeeze().to_numpy()
low10 = float(np.min(low[-10:]))
ema21 = float(pd.Series(df['Close'].squeeze()).ewm(span=21).mean().iloc[-1])
print()
print(f"Context:")
print(f"  EMA21:    {ema21:.2f}")
print(f"  10d low:  {low10:.2f}")
print(f"  Range10d: {round((price - low10)/price*100,1)}% from current price")
print()
print("All gates passed - Polycab should score.")
