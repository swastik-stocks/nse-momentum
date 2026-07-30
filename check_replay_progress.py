import sqlite3

conn = sqlite3.connect("data/momentum_v4.db")

done = conn.execute("SELECT COUNT(*) FROM pipeline_replay_hourly_progress").fetchone()[0]
total_signals = conn.execute("SELECT COALESCE(SUM(signals), 0) FROM pipeline_replay_hourly_progress").fetchone()[0]

try:
    total_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM price_history_hourly").fetchone()[0]
except Exception:
    total_tickers = None

conn.close()

print(f"Tickers completed: {done}" + (f" / {total_tickers}" if total_tickers else ""))
print(f"Gate-cleared signals so far: {total_signals:,}")
if total_tickers and done:
    pct = done / total_tickers * 100
    print(f"Progress: {pct:.1f}%")
