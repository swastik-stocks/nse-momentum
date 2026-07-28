"""
NSE Momentum v6 — Universe CSV Patcher
Fixes the 15 confirmed data bugs found by audit_universe_csv.py: symbols
that are still active NIFTY 500 constituents today but were incorrectly
shown as "permanently removed" partway through the snapshot history.

For each confirmed-bug symbol, re-adds it to every snapshot's symbol list
from (and including) the snapshot immediately AFTER its wrongly-recorded
"last seen" date, through to the final snapshot — i.e. treats it as
continuously present, since there's no evidence it was ever legitimately
removed (it's still active today) and no evidence of a legitimate later
re-addition event either.

NOTE: after patching, affected snapshot rows will have slightly MORE than
500 symbols (500 + however many of the 15 bugs applied to that date) —
that's expected and correct; it reflects the true point-in-time
membership, not a formatting requirement that every row be exactly 500.

Usage:
    python patch_universe_csv.py nifty500_2016-01-01_to_2026-12-31.csv
    (writes nifty500_2016-01-01_to_2026-12-31_patched.csv alongside it)

Then reload the patched file:
    python load_deep_history.py --universe-csv nifty500_2016-01-01_to_2026-12-31_patched.csv

And re-run the audit to confirm zero remaining confirmed bugs:
    python audit_universe_csv.py nifty500_2016-01-01_to_2026-12-31_patched.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nse_universe import NSE_UNIVERSE

# The 15 confirmed bugs found by audit_universe_csv.py — hardcoded here
# rather than re-detected, so the patch logic is explicit and auditable
# rather than silently re-running the same heuristic twice.
CONFIRMED_BUGS = [
    "GABRIEL", "USHAMART", "SBIN", "SPLPETRO", "APARINDS", "ADANIENT",
    "BANKINDIA", "JPPOWER", "INOXWIND", "RKFORGE", "TATACHEM", "RPOWER",
    "OIL", "ASAHIINDIA", "RELIANCE",
]


def patch(csv_path: str) -> str:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # For each bug symbol, find the last snapshot INDEX where it's genuinely
    # present, then re-add it to every later row.
    for sym in CONFIRMED_BUGS:
        last_present_idx = None
        for i, r in enumerate(rows):
            if sym in r["symbols"].split(","):
                last_present_idx = i
        if last_present_idx is None:
            print(f"  WARNING: {sym} never found present in any row — skipping (check spelling)")
            continue

        added_count = 0
        for i in range(last_present_idx + 1, len(rows)):
            symbols_list = rows[i]["symbols"].split(",")
            if sym not in symbols_list:
                symbols_list.append(sym)
                rows[i]["symbols"] = ",".join(symbols_list)
                added_count += 1
        print(f"  {sym}: re-added to {added_count} snapshots after {rows[last_present_idx]['effective_date']}")

    out_path = str(Path(csv_path).with_name(Path(csv_path).stem + "_patched.csv"))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_universe_csv.py <path_to_universe_csv>")
        sys.exit(1)

    print(f"Patching {len(CONFIRMED_BUGS)} confirmed-bug symbols...\n")
    out_path = patch(sys.argv[1])
    print(f"\nPatched file written to: {out_path}")
    print("Next steps:")
    print(f"  python audit_universe_csv.py {out_path}   # should show 0 confirmed bugs now")
    print(f"  python load_deep_history.py --universe-csv {out_path}")
