import sqlite3
import pandas as pd
from agents.asymmetry_gate import _compute_atr14
from hourly_scaling import BARS_PER_DAY

conn = sqlite3.connect("data/momentum_v4.db")

dfh = pd.read_sql(
    "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
    "WHERE ticker='ABB.NS' ORDER BY datetime ASC", conn
)
dfh["datetime"] = pd.to_datetime(dfh["datetime"])
dfh.set_index("datetime", inplace=True)
dfh.columns = ["Open", "High", "Low", "Close", "Volume"]

try:
    dfd = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history_deep "
        "WHERE ticker='ABB.NS' ORDER BY date ASC", conn
    )
    dfd["date"] = pd.to_datetime(dfd["date"])
    dfd.set_index("date", inplace=True)
    dfd.columns = ["Open", "High", "Low", "Close", "Volume"]
    have_daily = True
except Exception as e:
    have_daily = False
    print("No daily data available:", e)

conn.close()

atr_hourly_period = round(14 * BARS_PER_DAY)
atr_h = _compute_atr14(dfh, period=atr_hourly_period)
price_h = float(dfh["Close"].iloc[-1])
print(f"Hourly ATR({atr_hourly_period} bars = ~14 trading days): {atr_h:.2f}  ({atr_h/price_h*100:.2f}% of price)")

if have_daily:
    atr_d = _compute_atr14(dfd, period=14)
    price_d = float(dfd["Close"].iloc[-1])
    print(f"Daily  ATR(14 bars = 14 trading days):              {atr_d:.2f}  ({atr_d/price_d*100:.2f}% of price)")

