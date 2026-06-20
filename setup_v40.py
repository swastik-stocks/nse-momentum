"""
NSE Momentum v4.0 — First-Time Setup
Run this ONCE to initialise all v4.0 tables and seed the local price database.

Steps:
  1. Creates all SQLite tables (momentum_v4.db)
  2. Seeds dynamic_weights from 264 training examples
  3. Downloads 2yr OHLCV for all 504 stocks (~30-45 min)

Usage:
    python setup_v40.py             # full setup (all 504 stocks)
    python setup_v40.py --fast      # tables + weights only (no price download)
    python setup_v40.py --check     # check what's already done

NOTE: Only run this ONCE on a fresh install.
      Daily updates are handled by scanner.py + collectors/price_collector.py
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import init_all_tables, get_table_stats, DB_PATH
from trade_logger import init_tables as init_trade_tables

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         NSE Momentum v4.3 — First-Time Setup                        ║
║         504 stocks | 12 agents | 19 patterns | All free data        ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def step1_init_tables():
    print("─" * 60)
    print("[1/3] Initialising database tables...")
    init_all_tables()
    init_trade_tables()
    print(f"  ✅ Database: {DB_PATH}")
    stats = get_table_stats()
    for t, n in stats.items():
        print(f"       {t:<30} {n:>8,} rows")


def step2_seed_weights():
    print("\n─" * 60)
    print("[2/3] Seeding pattern weights from 264 training examples...")
    # Seeding happens inside init_tables() → _seed_weights()
    # Just verify it ran
    from database.schema import get_connection
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM dynamic_weights").fetchone()[0]
    conn.close()
    print(f"  ✅ {n} patterns seeded in dynamic_weights")


def step3_seed_prices(fast: bool = False):
    print("\n─" * 60)
    if fast:
        print("[3/3] Skipping price download (--fast mode)")
        print("  ℹ  Run 'python collectors/price_collector.py' separately to download prices")
        return

    print("[3/3] Downloading 2yr price history for 504 stocks...")
    print("  ⏳ This takes ~30-45 minutes. Progress shown every 50 stocks.")
    print("  ℹ  You can interrupt with Ctrl+C and resume later (incremental).")
    print()

    from collectors.price_collector import seed_all
    seed_all()


def check_setup():
    print("\n─" * 60)
    print("Setup Status Check:")
    if not DB_PATH.exists():
        print(f"  ❌ Database not found at {DB_PATH}")
        return

    stats = get_table_stats()
    for t, n in stats.items():
        status = "✅" if n > 0 else "⚠ "
        print(f"  {status} {t:<30} {n:>8,} rows")

    ph = stats.get("price_history", 0)
    dw = stats.get("dynamic_weights", 0)
    if ph >= 100_000:
        print(f"\n  🟢 Price history: READY ({ph:,} rows = ~{ph//504:.0f} days per stock)")
    elif ph > 0:
        print(f"\n  🟡 Price history: PARTIAL ({ph:,} rows — run price_collector.py to complete)")
    else:
        print("\n  🔴 Price history: EMPTY — run 'python setup_v40.py' to download")

    if dw >= 19:
        print(f"  🟢 Pattern weights: SEEDED ({dw} patterns)")
    else:
        print(f"  🔴 Pattern weights: NOT SEEDED — run 'python setup_v40.py --fast'")


if __name__ == "__main__":
    args = sys.argv[1:]
    print_banner()

    if "--check" in args:
        check_setup()
        sys.exit(0)

    fast = "--fast" in args

    step1_init_tables()
    step2_seed_weights()
    step3_seed_prices(fast=fast)

    print("""
─────────────────────────────────────────────────────────
✅ Setup complete!

Daily workflow:
  1. Double-click run_scanner.bat (or: python scanner.py)
  2. Email arrives in ~75 seconds

Trade logging:
  from trade_logger import log_entry, log_exit, print_stats

Single stock deep-dive:
  python single_stock.py RELIANCE

Portfolio review:
  python portfolio_engine.py review

Pattern leaderboard:
  python trade_logger.py   (or: from trade_logger import print_stats; print_stats())
─────────────────────────────────────────────────────────
""")
