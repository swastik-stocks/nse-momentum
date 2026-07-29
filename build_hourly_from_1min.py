"""
Resamples per-ticker 1-minute CSVs (stockdata_1 folder(s), one file per
ticker like "360ONE_minute.csv") into ONE combined hourly OHLCV file,
matching the nifty500_1d.csv / nifty500_15m.csv convention already used
elsewhere in this project.

WHY: uploading ~500 individual 1-min files (30-55MB each) isn't practical.
Resampling locally into one compact hourly file is the same pattern
already used for every other dataset in this project.

Set FOLDERS below to every folder containing *_minute.csv files (Part 1/
2/3 may be separate folders — list all of them).

Usage:
    python build_hourly_from_1min.py
    -> writes nifty500_1h_built.csv in the current folder
"""
import glob
import os
import pandas as pd

# EDIT THIS: list every folder containing *_minute.csv files
FOLDERS = [
    r"C:\Users\hp\Downloads\stockdata",
    # add Part 2 / Part 3 folder paths here if they're separate locations
]

OUTPUT_FILE = "nifty500_1h_built.csv"

all_hourly = []
files_processed = 0
files_failed = []

for folder in FOLDERS:
    pattern = os.path.join(folder, "*_minute.csv")
    files = glob.glob(pattern)
    print(f"{folder}: found {len(files)} files")

    for filepath in files:
        ticker = os.path.basename(filepath).replace("_minute.csv", "").replace("_minute", "")
        try:
            df = pd.read_csv(filepath, usecols=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            hourly = df.resample("1h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(subset=["close"])

            hourly["Ticker"] = ticker
            hourly = hourly.reset_index().rename(columns={
                "date": "Datetime", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            })
            all_hourly.append(hourly)
            files_processed += 1

            if files_processed % 50 == 0:
                print(f"  ...{files_processed} tickers processed")

        except Exception as e:
            files_failed.append((ticker, str(e)))

print(f"\nProcessed {files_processed} tickers, {len(files_failed)} failed")
if files_failed:
    print("Failed:", files_failed[:10])

combined = pd.concat(all_hourly, ignore_index=True)
combined = combined[["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
combined.to_csv(OUTPUT_FILE, index=False)

print(f"\nWrote {OUTPUT_FILE}: {len(combined):,} rows, "
      f"{combined['Ticker'].nunique()} tickers, "
      f"{combined['Datetime'].min()} to {combined['Datetime'].max()}")

