"""
triage_scratch_scripts.py — dump the first ~15 lines (docstring/header)
of each untracked scratch script, so we can decide commit vs delete
without opening each file by hand.

Usage: python triage_scratch_scripts.py
"""

files = [
    "audit_universe_csv.py",
    "check_adaniports.py",
    "fyers_diagnostic.py",
    "load_deep_history.py",
    "nse_universe_builder.py",
    "patch_universe_csv.py",
    "regime_backfill.py",
    "regime_backfill_deep.py",
    "validation/pipeline_replay.py",
    "validation/split_period_significance.py",
    "verify_staleness_fix.py",
]

for f in files:
    print("=" * 90)
    print(f"  {f}")
    print("=" * 90)
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 15:
                    print("  ...")
                    break
                print(f"  {line.rstrip()}")
    except FileNotFoundError:
        print("  (file not found -- may already be gone, or path differs)")
    print()
