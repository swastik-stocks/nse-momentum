"""
Run from your repo root: python verify_staleness_fix.py

Reads your REAL picks_latest.json and runs it through the fixed
staleness check ONLY -- does not call get_live_price, get_rvol, or
send any email. Safe to run any time, any number of times.
"""

import json
from datetime import date
from market_calendar.staleness_check import check_staleness, StaleDataError

PICKS_JSON_PATH = "picks_latest.json"

with open(PICKS_JSON_PATH, encoding="utf-8") as f:
    picks = json.load(f)

print(f"Loaded {len(picks)} picks from {PICKS_JSON_PATH}")

picks_meta = picks[0] if picks else {}
scan_date_str = picks_meta.get("scan_date", "")
today_iso = date.today().isoformat()

print(f"scan_date in file : {scan_date_str}")
print(f"today             : {today_iso}")

if not scan_date_str:
    print("\nNo scan_date field found in picks_latest.json -- check the key name.")
else:
    picks_date = date.fromisoformat(scan_date_str)
    try:
        check_staleness(picks_date)
        print("\nRESULT: PASS -- picks file is fresh (or correctly one trading day old).")
        print("The fix would NOT block confirmation. main() would proceed normally.")
    except StaleDataError as e:
        print(f"\nRESULT: STALE -- {e}")
        print("The fix WOULD correctly block confirmation here -- genuine gap detected.")
