import pandas as pd, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'agents')

from data_fetcher import fetch_batch_ohlcv
from agents.risk_agent import RiskAgent
from agents.pattern_agent import PatternAgent

tickers = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','TATAMOTORS.NS','ICICIBANK.NS']
dfs = fetch_batch_ohlcv(tickers)

print("\n{:<20s} {:>6s} {:>6s} {:>6s} {:>8s} {}".format(
    "Ticker", "Passes", "Stop%", "RRR", "EMA21", "Reason"))
print("-" * 80)

for t in tickers:
    df = dfs.get(t, pd.DataFrame())
    if df.empty:
        print(t, "-- no data")
        continue

    close = df["Close"].squeeze()
    ema21 = round(float(close.ewm(span=21, adjust=False).mean().iloc[-1]), 2)
    price = round(float(close.iloc[-1]), 2)

    pa = PatternAgent(df)
    entry_ref = price

    ra = RiskAgent(
        df,
        pa.breakout_level or entry_ref,
        pa.entry_low  or entry_ref * 0.99,
        pa.entry_high or entry_ref * 1.01,
        universe='LARGE'
    )

    stop_pct = 0.0
    if ra.entry > 0 and ra.stop > 0:
        stop_pct = round((ra.entry - ra.stop) / ra.entry * 100, 2)

    reason = ra.reject_reason() if not ra.passes() else "-"

    print("{:<20s} {:>6s} {:>5.1f}%  {:>5.2f}x  EMA21={:<8.2f}  {}".format(
        t,
        str(ra.passes()),
        stop_pct,
        ra.rrr,
        ema21,
        reason
    ))

    # Extra detail: show raw stop vs EMA21 to see which is controlling
    if ra.entry > 0:
        stop_ema_floor = round(ema21 * 0.993, 2)
        gap_pct = round((ra.entry - stop_ema_floor) / ra.entry * 100, 2)
        print("  entry={} stop={} ema21_floor={} gap_pct={}% pattern={}".format(
            ra.entry, ra.stop, stop_ema_floor, gap_pct,
            pa.pattern or "none"
        ))
