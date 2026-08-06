"""
verify_p305.py — quick verification that pct_above_sma50_mcap is populated.
Run from F:\\nse_momentum after applying the patch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sector_breadth import compute_sector_breadth

results = compute_sector_breadth()

print(f"\n{'SECTOR':<38} {'SMA50 Count':>12} {'SMA50 MCap':>11} {'DIFF':>7}")
print("-" * 75)

n_with_mcap = 0
for r in sorted(results, key=lambda x: -x["pct_above_sma50"]):
    count_pct = r["pct_above_sma50"]
    mcap_pct  = r.get("pct_above_sma50_mcap")
    if mcap_pct is not None:
        diff = mcap_pct - count_pct
        diff_str = f"{diff:+.1f}%"
        n_with_mcap += 1
    else:
        diff_str = "  n/a"
        mcap_pct = float('nan')
    mcap_str = f"{mcap_pct:>10.1f}%" if mcap_pct == mcap_pct else "       n/a"
    print(f"{r['sector']:<38} {count_pct:>11.1f}% {mcap_str} {diff_str:>7}")

print(f"\n{n_with_mcap}/{len(results)} sectors have MCap-weighted SMA50 data")
print("\nKey insight: sectors where MCap% >> Count% are dominated by large-cap stocks above SMA50.")
print("Sectors where MCap% << Count% have megacaps BELOW SMA50 despite many smaller stocks above.")
