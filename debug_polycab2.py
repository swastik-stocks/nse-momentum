"""
Full pipeline trace for POLYCAB in the context of the real universe.
Run from: C:\Users\User\Desktop\nse_momentum\
"""
import warnings
warnings.filterwarnings('ignore')
import sys, logging
logging.basicConfig(level=logging.WARNING)

from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from agents.liquidity_agent import LiquidityAgent
import pandas as pd
import numpy as np

TARGET = 'POLYCAB.NS'

# Get all tickers for proper RS ranking
all_tickers = [item[0] for item in NSE_UNIVERSE]
print(f"Fetching {len(all_tickers)} universe stocks for real RS ranking...")
dfs = fetch_batch_ohlcv(all_tickers + ['^NSEI'])
nifty = dfs.get('^NSEI', pd.DataFrame())
df    = dfs.get(TARGET, pd.DataFrame())

if df is None or df.empty:
    print("ERROR: No data for POLYCAB"); sys.exit()

print(f"Loaded {len(df)} bars for POLYCAB. Price: {float(df['Close'].iloc[-1]):.2f}")
print()

# Compute real universe ranks
stock_data = {k: v for k, v in dfs.items() if k != '^NSEI'}
ranks = compute_universe_ranks({"stock_data": stock_data, "nifty50_data": nifty})
pct = ranks.get(TARGET, 0)
print(f"POLYCAB RS percentile in full universe: {pct}")

universe_tag = "LARGE"
sector = "Unknown"
for item in NSE_UNIVERSE:
    if item[0] == TARGET:
        universe_tag = item[3]
        sector = item[2]
        break

cfg = UNIVERSE_CONFIG[universe_tag]
print(f"Universe: {universe_tag} | Sector: {sector}")
print()

# Gate 1: Liquidity
liq = LiquidityAgent(df, universe=universe_tag)
print(f"G1 Liquidity:  {'PASS' if liq.passes() else 'FAIL - ' + liq.reject_reason()}")

# Gate 2: Pattern
pa = PatternAgent(df)
print(f"G2 Pattern:    {pa.pattern or 'NONE'} | breakout={pa.breakout_level:.0f}")
if not pa.pattern:
    print("BLOCKED at G2"); sys.exit()

# Gate 3: RS with real universe ranks
rs_agent = RSAgent(df, nifty, universe_ranks=ranks, ticker=TARGET)
print(f"G3 RS:         percentile={pct:.0f} | passes_gate={rs_agent.passes_gate()} (gate=30)")
if not rs_agent.passes_gate():
    print("BLOCKED at G3"); sys.exit()

# Gate 4: Risk
ra = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high, universe=universe_tag)
print(f"G4 RiskAgent:  {'PASS' if ra.passes() else 'FAIL - ' + ra.reject_reason()}")
print(f"   Entry={ra.entry:.2f} Stop={ra.stop:.2f} T1={ra.target1:.2f} Stop%={ra.stop_pct}")
if not ra.passes():
    print("BLOCKED at G4"); sys.exit()

# Gate 5: Asymmetry
ag = AsymmetryGate(ra.entry, ra.stop, ra.target1, universe_tag)
res = ag.check()
print(f"G5 Asymmetry:  {'PASS' if res['qualified'] else 'FAIL - ' + res['fail_reason']}")
print(f"   risk={res['risk_pct']}% reward={res['reward_pct']}% RR={res['rr_ratio']}x")
if not res['qualified']:
    print("BLOCKED at G5"); sys.exit()

# Gate 6: VCP
vcpg = VCPContractionGate(df=df)
vcp = vcpg.check()
print(f"G6 VCPGate:    hard_reject={vcp['hard_reject']} penalty={vcp['penalty']} w4={vcp['w4_pct']}%")
if vcp['hard_reject']:
    print("BLOCKED at G6 - VCPGate W4 too wide"); sys.exit()

# Gate 7: Headroom
headroom = round((ra.target1 - ra.entry) / ra.entry * 100, 2) if ra.entry > 0 else 0
print(f"G7 Headroom:   {headroom}% | {'PASS' if headroom >= 4.5 else 'FAIL < 4.5%'}")
if headroom < 4.5:
    print("BLOCKED at G7"); sys.exit()

print()
print("All gates PASSED. Polycab should appear in results.")
print("If it still does not appear, check scanner.py for exception handling.")
