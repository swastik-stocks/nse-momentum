import pandas as pd, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'agents')

from data_fetcher import fetch_batch_ohlcv
from agents.pattern_agent import PatternAgent

tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
dfs = fetch_batch_ohlcv(tickers)

for t in tickers:
    df = dfs.get(t, pd.DataFrame())
    if df.empty:
        continue

    high  = df["High"].squeeze().to_numpy(dtype=float)
    low   = df["Low"].squeeze().to_numpy(dtype=float)
    close = df["Close"].squeeze().to_numpy(dtype=float)
    price = close[-1]

    pa    = PatternAgent(df)
    bo    = pa.breakout_level or price
    entry = pa.entry_high or price * 1.01

    ema21 = float(pd.Series(close).ewm(span=21, adjust=False).mean().iloc[-1])

    # Replicate RiskAgent target logic exactly
    lookback   = min(len(low), 60)
    base_low   = float(np.min(low[-lookback:]))
    pat_height = max(bo - base_low, 0)

    res_3m  = float(np.max(high[-65:]))
    res_6m  = float(np.max(high[-130:])) if len(high) >= 130 else res_3m
    res_12m = float(np.max(high[-252:])) if len(high) >= 252 else res_6m

    raw_t1 = entry + pat_height
    resistance_levels = sorted([r for r in [res_3m, res_6m, res_12m] if r > entry * 1.03])

    stop_ema = ema21 * 0.993
    stop_pct = round((entry - stop_ema) / entry * 100, 2)

    print(f"\n{t}")
    print(f"  price={price:.2f}  entry={entry:.2f}  EMA21={ema21:.2f}")
    print(f"  stop_ema_floor={stop_ema:.2f}  stop_pct={stop_pct}%")
    print(f"  pattern='{pa.pattern}'  breakout_level={bo:.2f}")
    print(f"  base_low={base_low:.2f}  pat_height={pat_height:.2f}")
    print(f"  raw_t1={raw_t1:.2f}")
    print(f"  res_3m={res_3m:.2f}  res_6m={res_6m:.2f}  res_12m={res_12m:.2f}")
    print(f"  resistance_levels above entry*1.03: {[round(r,2) for r in resistance_levels]}")

    if resistance_levels:
        t1_capped = min(raw_t1, resistance_levels[0] * 0.99)
        t1_final  = max(t1_capped, entry * 1.045)
        print(f"  t1_capped={t1_capped:.2f}  t1_final={t1_final:.2f}")
    else:
        t1_final = max(raw_t1, entry * 1.045)
        print(f"  t1_final={t1_final:.2f} (no resistance cap)")

    reward_pct = round((t1_final - entry) / entry * 100, 2)
    risk_pct   = stop_pct
    rr         = round(reward_pct / risk_pct, 2) if risk_pct > 0 else 0
    print(f"  reward_pct={reward_pct}%  risk_pct={risk_pct}%  R:R={rr}x")
    print(f"  VERDICT: {'PASS' if reward_pct >= 4.5 and rr >= 1.5 else 'FAIL'}")
