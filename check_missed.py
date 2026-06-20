import warnings
warnings.filterwarnings('ignore')

from nse_universe import NSE_UNIVERSE
from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
import pandas as pd

tickers_in_universe = [item[0] for item in NSE_UNIVERSE]

targets = ['ITCHOTELS.NS', 'POLYCAB.NS']
print("=== Universe Check ===")
for t in targets:
    print(t, "-> IN UNIVERSE:", t in tickers_in_universe)

print()
dfs = fetch_batch_ohlcv(targets)

for t in targets:
    df = dfs.get(t, pd.DataFrame())
    print()
    print(t)
    if df is None or df.empty:
        print("  NO DATA")
        continue

    bars  = len(df)
    price = float(df['Close'].iloc[-1])
    print("  Bars:", bars, "| Price:", round(price, 2))

    if bars < 60:
        print("  BLOCKED: Less than 60 bars")
        continue

    pa = PatternAgent(df)
    print("  Pattern:", pa.pattern or "NONE")
    print("  Breakout:", round(pa.breakout_level, 2))

    if pa.breakout_level > 0:
        gap = (pa.breakout_level - price) / price * 100
        print("  Gap to breakout:", "BROKEN OUT" if gap <= 0 else str(round(gap,1)) + "% away")

    universe_tag = "LARGE"
    for item in NSE_UNIVERSE:
        if item[0] == t:
            universe_tag = item[3]
            break
    print("  Universe tag:", universe_tag)

    if pa.pattern and pa.breakout_level > 0:
        ra = RiskAgent(df, pa.breakout_level, pa.entry_low, pa.entry_high,
                       universe=universe_tag)
        if ra.passes():
            print("  RiskAgent: PASS | Entry:", ra.entry, "Stop:", ra.stop, "T1:", ra.target1)
            print("  Stop%:", ra.stop_pct, "| Gain%:", ra.gain_pct, "| RRR:", ra.rrr)
            ag = AsymmetryGate(ra.entry, ra.stop, ra.target1, universe_tag)
            res = ag.check()
            print("  AsymmetryGate:", "PASS" if res["qualified"] else "FAIL")
            print("  Risk%:", res["risk_pct"], "| Reward%:", res["reward_pct"], "| RR:", res["rr_ratio"])
            if not res["qualified"]:
                print("  Fail:", res["fail_reason"])
        else:
            print("  RiskAgent: FAIL -", ra.reject_reason())
