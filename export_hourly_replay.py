import sqlite3
import json

conn = sqlite3.connect("data/momentum_v4.db")
rows = conn.execute(
    "SELECT ticker, signals, results_json, completed_at FROM pipeline_replay_hourly_progress"
).fetchall()
conn.close()

export = []
total = 0
for ticker, signals, results_json, completed_at in rows:
    parsed = json.loads(results_json) if results_json else {}
    for pat, outcomes in parsed.items():
        total += len(outcomes)
    export.append({"ticker": ticker, "signals": signals, "results": parsed, "completed_at": completed_at})

with open("hourly_replay_export.json", "w") as f:
    json.dump(export, f)

print(f"Exported {len(export)} tickers, {total:,} trades -> hourly_replay_export.json")
