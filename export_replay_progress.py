"""
Exports pipeline_replay_deep_progress (the per-ticker, per-trade evidence
data with dates) to a single compact JSON file -- so we don't need to
upload the whole momentum_v4.db (which also carries price_history_deep's
1.75M rows we don't need for this).

Usage:
    python export_replay_progress.py
    -> writes replay_progress_export.json in the current folder
"""
import sqlite3
import json

conn = sqlite3.connect("data/momentum_v4.db")
rows = conn.execute(
    "SELECT ticker, signals, results_json, completed_at FROM pipeline_replay_deep_progress"
).fetchall()
conn.close()

export = []
total_trades = 0
for ticker, signals, results_json, completed_at in rows:
    parsed = json.loads(results_json) if results_json else {}
    for pat, outcomes in parsed.items():
        total_trades += len(outcomes)
    export.append({
        "ticker": ticker,
        "signals": signals,
        "results": parsed,
        "completed_at": completed_at,
    })

with open("replay_progress_export.json", "w") as f:
    json.dump(export, f)

print(f"Exported {len(export)} tickers, {total_trades:,} total trades across all patterns.")
print("Wrote replay_progress_export.json")
