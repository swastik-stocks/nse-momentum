import sys, sqlite3, numpy as np, pandas as pd, logging
sys.path.insert(0, '.')
logging.disable(logging.CRITICAL)

conn = sqlite3.connect('data/momentum_v4.db')
rows = conn.execute(
    'SELECT date,open,high,low,close,volume FROM price_history WHERE ticker=? ORDER BY date',
    ('AARTIIND.NS',)
).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=['Date','Open','High','Low','Close','Volume'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').astype(float)
print(f'AARTIIND.NS: {len(df)} bars | last close={df["Close"].iloc[-1]:.2f}')
print()

from agents.liquidity_agent import LiquidityAgent
from agents.pattern_agent import PatternAgent
from agents.risk_agent import RiskAgent
from agents.rs_agent import RSAgent
from agents.volume_agent import VolumeAgent

# 1. Liquidity
liq = LiquidityAgent(df, 'MID')
print(f'[1] Liquidity: {"PASS ADT=Rs."+str(round(liq.get_adt(),0))+"Cr" if liq.passes() else "FAIL: "+liq.reject_reason()}')

# 2. Pattern
pa = PatternAgent(df)
print(f'[2] Pattern:   "{pa.pattern}" score={pa.raw_score} breakout={pa.breakout_level:.1f}')

c = df['Close'].to_numpy(dtype=float)
h = df['High'].to_numpy(dtype=float)
v = df['Volume'].to_numpy(dtype=float)
e50 = float(pd.Series(c).ewm(span=50, adjust=False).mean().iloc[-1])
e200 = float(pd.Series(c).ewm(span=200, adjust=False).mean().iloc[-1])
w52h = float(h[-252:].max()) if len(h) >= 252 else float(h.max())
rvol_y = float(v[-2]) / float(np.mean(v[-21:-1])) if len(v) > 21 else 0

print(f'     above50={c[-1]>e50} above200={c[-1]>e200}')
print(f'     at_52wh={c[-1]/w52h:.1%} near52w(80%)={c[-1]>=0.80*w52h}')
print(f'     rvol_yesterday={rvol_y:.2f}')

if not pa.pattern:
    print('     WHY NO PATTERN:')
    swing = float(h[-30:-5].max())
    rec60 = float(h[-60:-5].max())
    print(f'       swing_res={swing:.1f}  price/swing={c[-1]/swing:.1%}  need>=99%')
    print(f'       rec60_res={rec60:.1f}  price/rec60={c[-1]/rec60:.1%}  need>=92% + above50')
    db_l = df['Low'].to_numpy()[-40:]
    b1 = db_l[:20].min(); b2 = db_l[20:].min()
    neck = float(h[-40:].max()) * 0.99
    print(f'       dbl_bot: b1={b1:.0f} b2={b2:.0f} similar(3%)={abs(b1-b2)/b1<=0.03} near_neck={c[-1]>=neck*0.95}')
    mid = len(c) // 2
    la = np.mean(c[:mid//2]); ta = np.mean(c[mid//2:mid]); ra = np.mean(c[mid:])
    print(f'       rounded: left={la:.0f} trough={ta:.0f} right={ra:.0f}')
    print(f'         shape_ok={ta<la*0.92 and ra>ta*1.05 and ra>=la*0.75}')
    low60 = float(df['Low'].to_numpy()[-60:].min())
    gain = (c[-1] - low60) / low60
    hh = float(h[-10:].max()) > float(h[-30:-10].max())
    print(f'       momentum_rising: gain_from_60d_low={gain:.1%} higher_highs={hh} need>=12%+above50')
    print()
    print('  *** All pattern conditions FAILED for this stock ***')

# 3. RS
print()
rsa = RSAgent(df, pd.DataFrame(), universe_ranks={'AARTIIND.NS': 65.0}, ticker='AARTIIND.NS')
print(f'[3] RS:        percentile=65 score={rsa.score()}/20  PASS')

# 4. Volume
va = VolumeAgent(df, delivery_pct=35.0, universe='MID')
print(f'[4] Volume:    score={va.score()}/12  rvol={va.get_rvol():.2f}')

# 5. Risk (even without pattern, test with 30-bar high)
bo = float(h[-30:].max())
risk = RiskAgent(df, bo, c[-1]*0.998, bo*1.003, 'MID')
print(f'[5] Risk:      {"PASS: Entry="+str(round(risk.entry,1))+" SL="+str(round(risk.stop,1))+" ("+str(risk.stop_pct)+"%) T1="+str(round(risk.target1,1))+" RRR="+str(risk.rrr)+"x" if risk.passes() else "FAIL: "+risk.reject_reason()}')

# 6. Full score
print()
if pa.pattern and risk.passes():
    raw = rsa.score() + pa.raw_score + pa.get_rsi_score() + va.score() + pa.get_ema_score() + 4 + pa.get_macd_score() + 5 + 2
    print(f'[6] SCORE:     RS={rsa.score()} Pat={pa.raw_score} RSI={pa.get_rsi_score()} Vol={va.score()} EMA={pa.get_ema_score()} MACD={pa.get_macd_score()} Mkt=4 Sec=5 Bonus=2')
    print(f'               RAW={raw}/100  gate=64  -> {"✅ CLEARS T1" if raw>=64 else "⚡ T2" if raw>=55 else "👁 T3" if raw>=42 else "❌ BELOW"}')
elif risk.passes():
    raw = rsa.score() + 14 + pa.get_rsi_score() + va.score() + pa.get_ema_score() + 4 + pa.get_macd_score() + 5 + 2
    print(f'[6] SCORE EST: (no pattern detected, using 14 as placeholder)')
    print(f'               RAW~{raw}/100  gate=64  -> {"✅ CLEARS T1" if raw>=64 else "⚡ T2" if raw>=55 else "👁 T3" if raw>=42 else "❌ BELOW"}')
    print(f'               Risk would be: Entry={risk.entry:.0f} SL={risk.stop:.0f} T1={risk.target1:.0f} RRR={risk.rrr:.1f}x')
