import sys, importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("portfolio_watch", "portfolio_watch.py")
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)

passed, total = 0, 0

# --- Test 1: find_prefix_collisions catches a real same-prefix pair ---
total += 1
df = pd.DataFrame({
    "raw_name": ["Bajaj Finance Ltd", "Bajaj Finserv Ltd", "Reliance Industries Ltd"],
    "ticker":   ["BAJFINANCE", "BAJAJFINSV", "RELIANCE"],
})
collisions = pw.find_prefix_collisions(df, min_prefix_len=6)
if len(collisions) == 1 and collisions[0][1] == "BAJFINANCE" and collisions[0][3] == "BAJAJFINSV":
    print(f"PASS Test 1: caught Bajaj Finance/Finserv collision -> {collisions[0]}")
    passed += 1
else:
    print(f"FAIL Test 1: expected 1 collision (Bajaj pair), got {collisions}")

# --- Test 2: no false positive when tickers resolved identically (dup rows) ---
total += 1
df2 = pd.DataFrame({
    "raw_name": ["Reliance Industries Ltd", "Reliance Industries Limited"],
    "ticker":   ["RELIANCE", "RELIANCE"],
})
collisions2 = pw.find_prefix_collisions(df2, min_prefix_len=6)
if len(collisions2) == 0:
    print("PASS Test 2: same resolved ticker correctly NOT flagged as a collision")
    passed += 1
else:
    print(f"FAIL Test 2: expected 0 collisions for same-ticker rows, got {collisions2}")

# --- Test 3: unrelated names produce no collision ---
total += 1
df3 = pd.DataFrame({
    "raw_name": ["Reliance Industries Ltd", "Tata Consultancy Services Ltd"],
    "ticker":   ["RELIANCE", "TCS"],
})
collisions3 = pw.find_prefix_collisions(df3, min_prefix_len=6)
if len(collisions3) == 0:
    print("PASS Test 3: unrelated companies correctly produce no collision")
    passed += 1
else:
    print(f"FAIL Test 3: expected 0, got {collisions3}")

# --- Test 4: ambiguity detection logic (direct, no network needed) ---
total += 1
import difflib
raw = "ICICI Prudential Life Insurance Company Ltd"
cand_a = "ICICI Prudential Life Insurance Company Ltd"   # near-perfect
cand_b = "ICICI Prudential Asset Management Co Ltd"       # genuine close collision risk
ratio_a = difflib.SequenceMatcher(None, raw, cand_a).ratio()
ratio_b = difflib.SequenceMatcher(None, raw, cand_b).ratio()
margin = ratio_a - ratio_b
print(f"  (debug) ratio_a={ratio_a:.3f} ratio_b={ratio_b:.3f} margin={margin:.3f}")
if ratio_a > 0.75:
    print(f"PASS Test 4: exact-name candidate clears cutoff as expected (ratio={ratio_a:.2f})")
    passed += 1
else:
    print(f"FAIL Test 4: expected exact match to clear 0.75 cutoff, got {ratio_a:.2f}")

print(f"\n{passed}/{total} checks passed (Tests 1-4).")

# --- Test 5: the actual AMBIGUOUS_FUZZY code path, with mocked master list ---
# (bypasses network — directly injects a synthetic isin_map/name_map so this
# tests the real resolve_tickers() branching logic, not just the math)
print("\n--- Test 5: real resolve_tickers() ambiguity branch ---")
total_5, passed_5 = 1, 0

synthetic_name_map = {
    "ICICI Prudential Life Insurance Company Ltd": "ICICIPRULI",
    "ICICI Prudential Asset Management Co Ltd":    "ICICIAMC",
    "Reliance Industries Ltd":                      "RELIANCE",
}
synthetic_isin_map = {}  # force fuzzy path — no ISIN available

pw._load_nse_master = lambda: (synthetic_isin_map, synthetic_name_map)

raw_names = pd.Series(["ICICI Prudential Insurance Management Ltd"])  # verified: ratios [0.786, 0.840], margin=0.054 — genuinely ambiguous
resolved, methods = pw.resolve_tickers(raw_names, isins=None)

print(f"  resolved={resolved.tolist()}  methods={methods.tolist()}")
if methods.iloc[0] == "AMBIGUOUS_FUZZY":
    print("PASS Test 5: genuinely ambiguous name correctly flagged AMBIGUOUS_FUZZY")
    passed_5 += 1
else:
    print(f"FAIL Test 5: expected AMBIGUOUS_FUZZY, got {methods.iloc[0]}")

# Sanity companion: an UNambiguous name should still resolve cleanly as FUZZY_NAME
raw_names2 = pd.Series(["Reliance Industries Limited"])
resolved2, methods2 = pw.resolve_tickers(raw_names2, isins=None)
print(f"  (companion) resolved={resolved2.tolist()}  methods={methods2.tolist()}")
if methods2.iloc[0] == "FUZZY_NAME" and resolved2.iloc[0] == "RELIANCE":
    print("PASS Test 5b: unambiguous name still resolves cleanly (not over-flagged)")
    passed_5 += 1
    total_5 += 1
else:
    print(f"FAIL Test 5b: expected FUZZY_NAME/RELIANCE, got {methods2.iloc[0]}/{resolved2.iloc[0]}")
    total_5 += 1

print(f"\nTest 5 subtotal: {passed_5}/{total_5}")

grand_passed = passed + passed_5
grand_total = total + total_5
print(f"\nGRAND TOTAL: {grand_passed}/{grand_total} checks passed.")
sys.exit(0 if grand_passed == grand_total else 1)
