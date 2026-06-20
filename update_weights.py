"""
NSE Momentum v4.3 — Update Pattern Weights
Recomputes dynamic weights from all closed trades.
Run: python update_weights.py
"""

import sqlite3, logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "momentum_v4.db"


def update_weights():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Add reason column if not exists (migration guard)
    try:
        conn.execute("ALTER TABLE dynamic_weights ADD COLUMN reason TEXT")
        conn.commit()
    except Exception:
        pass

    patterns = conn.execute(
        "SELECT DISTINCT pattern FROM trades_v4 WHERE status='CLOSED'"
    ).fetchall()

    if not patterns:
        print("No closed trades yet. Weights remain at seed values.")
        conn.close()
        return

    print("\n" + "="*70)
    print("  UPDATING PATTERN WEIGHTS FROM LIVE TRADE EVIDENCE")
    print(f"  {datetime.today().strftime('%d %b %Y %H:%M')}")
    print("="*70)
    print(f"  {'PATTERN':<25} {'OLD':>4} {'NEW':>4}  EVIDENCE")
    print("  " + "-"*64)

    for row in patterns:
        pat = row["pattern"]
        trades = conn.execute("""
            SELECT r_multiple FROM trades_v4
            WHERE pattern=? AND status='CLOSED' AND r_multiple IS NOT NULL
        """, (pat,)).fetchall()

        if len(trades) < 5:
            print(f"  {pat:<25} (need ≥5 trades, have {len(trades)})")
            continue

        rs    = [t["r_multiple"] for t in trades]
        wins  = sum(1 for r in rs if r > 0)
        wr    = wins / len(rs)
        exp_r = sum(rs) / len(rs)
        wins_r = sum(r for r in rs if r > 0)
        loss_r = abs(sum(r for r in rs if r < 0))
        pf = wins_r / loss_r if loss_r > 0 else 2.0

        # Evidence-based weight
        new_wt = 10 + int((wr - 0.50) / 0.05)
        new_wt = max(5, min(18, new_wt))

        # ±30% change cap per cycle
        cur_row = conn.execute(
            "SELECT weight FROM dynamic_weights WHERE pattern=?", (pat,)
        ).fetchone()
        old_wt = int(cur_row["weight"]) if cur_row else 12
        cap    = int(old_wt * 0.30)
        new_wt = max(old_wt - cap, min(old_wt + cap, new_wt))

        reason = f"n={len(rs)} | WR={wr:.0%} | Exp={exp_r:.2f}R | PF={pf:.1f}"
        print(f"  {pat:<25} {old_wt:>4} {new_wt:>4}  {reason}")

        conn.execute("""
            INSERT OR REPLACE INTO dynamic_weights
            (pattern, weight, win_rate, expectancy_r, profit_factor,
             sample_count, last_updated, reason)
            VALUES (?,?,?,?,?,?,?,?)
        """, (pat, new_wt, wr, exp_r, pf, len(rs),
              datetime.today().strftime("%Y-%m-%d"), reason))

    conn.commit()
    conn.close()
    print("="*70)
    print("  Weights updated. Next scan will use these values.\n")


if __name__ == "__main__":
    update_weights()
