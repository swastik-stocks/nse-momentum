import warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, 'agents')

from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.volume_agent import VolumeAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from agents.liquidity_agent import LiquidityAgent
from agents.sector_agent import SectorAgent
from trade_logger import get_dynamic_weight
import pandas as pd
import numpy as np

TARGET = 'POLYCAB.NS'

print("Fetching data...")
dfs = fetch_batch_ohlcv([TARGET, '^NSEI'])
df    = dfs.get(TARGET, pd.DataFrame())
nifty = dfs.get('^NSEI', pd.DataFrame())

universe_tag = "LARGE"
sector = "Unknown"
for item in NSE_UNIVERSE:
    if item[0] == TARGET:
        universe_tag = item[3]
        sector = item[2]
        break

cfg   = UNIVERSE_CONFIG[universe_tag]
price = float(df['Close'].iloc[-1])
print(f"Price: {price} | Universe: {universe_tag} | Sector: {sector}")
print()

# Gate 1: Liquidity
liq = LiquidityAgent(df, universe=universe_tag)
print(f"G1 Liquidity:  {'PASS' if liq.passes() else 'FAIL - ' + liq.reject_reason()}")

# Gate 2: Pattern
pa = PatternAgent(df)
print(f"G2 Pattern:    {pa.pattern or 'NONE'} | breakout={pa.breakout_level:.0f}")
if not pa.pattern:
    print("BLOCKED at G2"); sys.exit()

# RS
universe_ranks = compute_universe_ranks({"stock_data": {TARGET: df}, "nifty50_data": nifty})
rsa = RSAgent(df, nifty, universe_ranks=universe_ranks, ticker=TARGET)
rs_score = min(int(rsa.score() * cfg.get("rs_weight_mult", 1.0)), 20)
rs_pct   = rsa.get_percentile()
print(f"G3 RS:         percentile={rs_pct:.0f} score={rs_score} | {'PASS' if rs_pct >= 40 else 'FAIL < 40'}")
if rs_pct < 40:
    print("BLOCKED at G3"); sys.exit()

# Gate 4: Risk
ra = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high, universe=universe_tag)
print(f"G4 RiskAgent:  {'PASS' if ra.passes() else 'FAIL - ' + ra.reject_reason()}")
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
print(f"G6 VCPGate:    hard_reject={vcp['hard_reject']} penalty={vcp['penalty']} w4={vcp['w4_pct']}")
if vcp['hard_reject']:
    print("BLOCKED at G6"); sys.exit()

# Gate 7: Headroom
headroom = round((ra.target1 - ra.entry) / ra.entry * 100, 2) if ra.entry > 0 else 0
print(f"G7 Headroom:   {headroom}% | {'PASS' if headroom >= 4.5 else 'FAIL < 4.5%'}")
if headroom < 4.5:
    print("BLOCKED at G7"); sys.exit()

# Score
dyn_weight    = get_dynamic_weight(pa.pattern)
pattern_score = min(dyn_weight, 18)
ema_score     = pa.get_ema_score()
macd_score    = pa.get_macd_score()
rsi_score     = pa.get_rsi_score()

vol    = df["Volume"].squeeze().to_numpy(dtype=float)
avg20v = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1
rvol   = round(float(vol[-1]) / avg20v, 2)

raw_score = rs_score + pattern_score + rsi_score + ema_score + macd_score
print()
print(f"=== SCORE BREAKDOWN ===")
print(f"RS score:      {rs_score}/20")
print(f"Pattern score: {pattern_score}/18  ({pa.pattern})")
print(f"RSI score:     {rsi_score}/15")
print(f"EMA score:     {ema_score}/10")
print(f"MACD score:    {macd_score}/8")
print(f"RVOL:          {rvol}x")
print(f"Partial raw:   {raw_score}")
print()
print(f"Score gate for {universe_tag}: {cfg['score_gate']}")
print(f"T2 gate: 55 | T3 gate: 42")
print()

if raw_score >= cfg['score_gate']:
    print(f"VERDICT: Would be TIER 1 (score {raw_score} >= gate {cfg['score_gate']})")
elif raw_score >= 55:
    print(f"VERDICT: Would be TIER 2 (score {raw_score})")
elif raw_score >= 42:
    print(f"VERDICT: Would be TIER 3 (score {raw_score})")
else:
    print(f"VERDICT: REJECTED - score {raw_score} below T3 threshold 42")
    print("This is why Polycab is missing — score too low despite good gates")
