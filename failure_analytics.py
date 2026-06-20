"""
NSE Momentum v5.0 - Failure Analytics Engine
Classifies failed trades by cause cluster, not just pattern.
Run manually after 10+ closed trades are in trades_v4.

Usage:
    python failure_analytics.py

Output:
    - Console report of failure clusters
    - Recommendations for gate recalibration
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / "data" / "momentum_v5.db"

# Failure cluster definitions
CLUSTERS = {
    "MACRO_HOSTILE":       "Failed during HOSTILE macro state",
    "LOW_HEADROOM":        "Headroom < 6% — resistance too close to entry",
    "WEAK_SECTOR":         "Sector rank > 8 or failure rate > 40%",
    "POST_BREAKOUT_REVERSAL": "Stop hit within 3 sessions of entry",
    "EVENT_DAY_FAILURE":   "WATCH or HIGH_RISK event on entry day",
    "DELIVERY_COLLAPSE":   "Low delivery % — no institutional backing",
    "SMALL_CAP_FRAGILITY": "SMALL universe — liquidity collapsed",
    "WIDE_STOP":           "Stop % > 4% — entry was extended",
    "OTHER":               "Unclassified failure",
}


def classify_failure(row: dict) -> str:
    """Classify a single failed trade into a cause cluster."""
    macro  = (row.get("macro_state")  or "MIXED").upper()
    event  = (row.get("event_risk")   or "NORMAL").upper()
    univ   = (row.get("universe")     or "LARGE").upper()
    hdroom = float(row.get("headroom_pct") or 0)
    deliv  = float(row.get("delivery_pct") or 0)
    vcp_w4 = float(row.get("vcp_w4")       or 0)
    hold   = int(row.get("hold_days")       or 99)

    # Priority order — first match wins
    if macro == "HOSTILE":
        return "MACRO_HOSTILE"
    if event in ("WATCH", "HIGH_RISK"):
        return "EVENT_DAY_FAILURE"
    if hold <= 3:
        return "POST_BREAKOUT_REVERSAL"
    if hdroom < 6.0:
        return "LOW_HEADROOM"
    if univ == "SMALL":
        return "SMALL_CAP_FRAGILITY"
    if deliv < 20:
        return "DELIVERY_COLLAPSE"
    if vcp_w4 > 10:
        return "WIDE_STOP"
    return "OTHER"


def run_analysis() -> dict:
    """
    Analyse all failed trades in trades_v4.
    Returns cluster counts and recommendations.
    """
    if not DB_PATH.exists():
        print("No database found at", DB_PATH)
        return {}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # Get all closed failed trades (r_multiple < 0 = stop hit)
    cur.execute("""
        SELECT *,
               CAST(julianday(exit_date) - julianday(entry_date) AS INTEGER) as hold_days
        FROM trades_v4
        WHERE status = 'CLOSED'
          AND r_multiple IS NOT NULL
          AND r_multiple < 0
        ORDER BY entry_date DESC
    """)
    failed = [dict(r) for r in cur.fetchall()]

    # Get all closed trades for context
    cur.execute("SELECT COUNT(*) FROM trades_v4 WHERE status='CLOSED'")
    total_closed = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM trades_v4
        WHERE status='CLOSED' AND r_multiple IS NOT NULL AND r_multiple > 0
    """)
    total_wins = cur.fetchone()[0]

    conn.close()

    if not failed:
        print("\nNo failed trades to analyse yet.")
        print(f"Total closed trades: {total_closed} | Wins: {total_wins}")
        return {}

    # Classify each failure
    clusters: dict = {k: [] for k in CLUSTERS}
    for row in failed:
        cluster = classify_failure(row)
        clusters[cluster].append(row["ticker"])

    total_failures = len(failed)
    win_rate = round(total_wins / total_closed * 100, 1) if total_closed > 0 else 0

    # Print report
    width = 70
    print()
    print("=" * width)
    print(f"  FAILURE ANALYTICS REPORT - {datetime.today().strftime('%d %b %Y')}")
    print(f"  Total closed: {total_closed} | Wins: {total_wins} "
          f"({win_rate}%) | Failures: {total_failures}")
    print("=" * width)

    print(f"\n  {'CLUSTER':<28} {'COUNT':>6} {'%':>6}  EXAMPLES")
    print("  " + "-" * 65)

    sorted_clusters = sorted(
        clusters.items(), key=lambda x: len(x[1]), reverse=True
    )
    recommendations = []

    for cluster, tickers in sorted_clusters:
        count = len(tickers)
        if count == 0:
            continue
        pct     = round(count / total_failures * 100, 1)
        examples = ", ".join(tickers[:3])
        if len(tickers) > 3:
            examples += f" +{len(tickers)-3}"
        print(f"  {cluster:<28} {count:>6} {pct:>5.1f}%  {examples}")

        # Generate recommendations
        if cluster == "MACRO_HOSTILE" and pct > 20:
            recommendations.append(
                "Consider skipping new entries when MacroAgent = HOSTILE"
            )
        elif cluster == "LOW_HEADROOM" and pct > 20:
            recommendations.append(
                "Raise headroom gate from 4.5% to 6% — too many stocks "
                "hitting resistance immediately after entry"
            )
        elif cluster == "POST_BREAKOUT_REVERSAL" and pct > 25:
            recommendations.append(
                "ConfirmationAgent needs stricter criteria — too many "
                "breakouts failing within 3 sessions"
            )
        elif cluster == "WIDE_STOP" and pct > 20:
            recommendations.append(
                "Lower VCP hard reject from 15% to 12% — wide-stop "
                "entries have high failure rate"
            )
        elif cluster == "DELIVERY_COLLAPSE" and pct > 15:
            recommendations.append(
                "Add delivery % floor gate of 25% minimum before scoring"
            )

    print()
    if recommendations:
        print("  RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No gate changes recommended yet (need more data).")

    print("=" * width)
    print()

    return {
        "total_closed":    total_closed,
        "total_wins":      total_wins,
        "total_failures":  total_failures,
        "win_rate":        win_rate,
        "clusters":        {k: len(v) for k, v in clusters.items()},
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_analysis()
