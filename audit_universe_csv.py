"""
NSE Momentum v6 — Universe CSV Data Quality Audit
Cross-checks the universe_snapshots CSV's symbols that appear to have
"permanently vanished" partway through against your CURRENT actual
NIFTY 500 list (nse_universe.py). Any symbol flagged as both "still in
today's real universe" AND "permanently removed years ago per the CSV"
is a CONFIRMED bug in the CSV — those two facts cannot both be true.

This is how we know RELIANCE and SBIN are real bugs (both are obviously
still trading, still huge NIFTY 50 constituents) without having to
manually judge every one of the ~135 flagged symbols by hand — most of
which (DHANLAXMI, GTL, NDTV, RUCHISOYA, CAIRN, etc.) are very likely
GENUINE historical removals (delistings, mergers, bankruptcies) and not
bugs at all.

Usage:
    python audit_universe_csv.py nifty500_2016-01-01_to_2026-12-31.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nse_universe import NSE_UNIVERSE

if len(sys.argv) < 2:
    print("Usage: python audit_universe_csv.py <path_to_universe_csv>")
    sys.exit(1)

csv_path = sys.argv[1]
current_symbols = {t[0].replace(".NS", "") for t in NSE_UNIVERSE}

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

all_symbols = set()
snapshot_sets = []
for r in rows:
    s = set(r["symbols"].split(","))
    snapshot_sets.append(s)
    all_symbols |= s

confirmed_bugs = []
for sym in all_symbols:
    presence = [sym in s for s in snapshot_sets]
    if presence[0] and not presence[-1]:
        last_true = max(i for i, p in enumerate(presence) if p)
        if all(not p for p in presence[last_true+1:]):
            gap_len = len(presence) - 1 - last_true
            if gap_len >= 3 and sym in current_symbols:
                confirmed_bugs.append((sym, rows[last_true]["effective_date"], gap_len))

confirmed_bugs.sort(key=lambda x: -x[2])
print(f"\nTotal snapshots in file: {len(rows)}")
print(f"Total distinct symbols ever seen: {len(all_symbols)}")
print(f"\nCONFIRMED BUGS (currently-active NIFTY 500 members the CSV shows as "
      f"permanently removed): {len(confirmed_bugs)}")
print(f"{'Symbol':<15}{'CSV says last seen':<20}{'Snapshots wrongly missing'}")
for sym, last_date, gap in confirmed_bugs:
    print(f"{sym:<15}{last_date:<20}{gap}")

if not confirmed_bugs:
    print("\nNo confirmed bugs found — every currently-active symbol's history "
          "in this file is internally consistent.")
