"""
Drop-in replacement for the buggy tvDatafeed RVOL path.

Fyers returns correctly IST-timestamped OHLCV bars, so this removes
the UTC-mislabeled-as-IST bug that was silently falling back to
yfinance in your current _rvol_tvdatafeed function.
"""

from datetime import datetime, timedelta
import pandas as pd

from fyers_auth import get_fyers_client


def get_intraday_bars(symbol: str, days: int = 5, resolution: str = "5") -> pd.DataFrame:
    """
    symbol: e.g. "RELIANCE" (NSE suffix added automatically)
    days: how many calendar days of history to pull
    resolution: candle size in minutes ("1", "5", "15", "60", "D")

    Returns a DataFrame with columns: datetime (IST, tz-aware), open,
    high, low, close, volume -- ready to plug into your RVOL calc.
    """
    fy = get_fyers_client()

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    response = fy.history({
        "symbol": f"NSE:{symbol}-EQ",
        "resolution": resolution,
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    })

    if response.get("s") != "ok" or not response.get("candles"):
        raise ValueError(f"Fyers returned no data for {symbol}: {response}")

    df = pd.DataFrame(response["candles"],
                       columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Fyers timestamps are Unix epoch UTC seconds -- convert explicitly to IST
    df["datetime"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                         .dt.tz_convert("Asia/Kolkata"))
    df = df.drop(columns=["timestamp"]).set_index("datetime")
    return df


def calculate_rvol(symbol: str, lookback_days: int = 20) -> float:
    """
    Relative volume = today's volume-so-far / average volume at the
    same time-of-day over the last `lookback_days` trading sessions.
    """
    bars = get_intraday_bars(symbol, days=lookback_days + 5, resolution="5")

    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    todays_bars = bars[bars.index.normalize() == today]
    if todays_bars.empty:
        raise ValueError(f"No bars for {symbol} today -- market may not have opened yet")

    latest_time = todays_bars.index[-1].time()
    todays_volume = todays_bars["volume"].sum()

    historical = bars[bars.index.normalize() < today]
    historical_same_window = historical[historical.index.time <= latest_time]
    avg_daily_volume = (historical_same_window
                         .groupby(historical_same_window.index.normalize())["volume"]
                         .sum()
                         .tail(lookback_days)
                         .mean())

    if avg_daily_volume == 0 or pd.isna(avg_daily_volume):
        raise ValueError(f"Insufficient historical data for {symbol} RVOL calc")

    return todays_volume / avg_daily_volume


if __name__ == "__main__":
    # Quick smoke test
    rvol = calculate_rvol("RELIANCE")
    print(f"RELIANCE RVOL: {rvol:.2f}x")
