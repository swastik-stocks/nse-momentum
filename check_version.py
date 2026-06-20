import sys
sys.path.insert(0, 'agents')

# Force fresh import - bypass any cache
import importlib
import agents.risk_agent as ra_mod
importlib.reload(ra_mod)

print("File loaded from:", ra_mod.__file__)
print()

# Check which version is running by looking for FIX markers
import inspect
src = inspect.getsource(ra_mod.RiskAgent._compute)

if 'stop_10d' in src:
    print("VERSION: v5 (10-day low fix present)")
elif 'stop_ema = ema21 * 0.993' in src:
    print("VERSION: v5-partial (EMA21 fix only, no 10-day low)")
else:
    print("VERSION: v4.3 OLD (none of the fixes loaded)")

print()
# Show the actual stop calculation lines
for i, line in enumerate(src.splitlines()):
    if any(x in line for x in ['stop_10d', 'stop_ema', 'stop_atr', 'candidates', 'max_stop']):
        print(f"  line {i:3d}: {line}")
