"""
patch_sector_breadth_p305.py — P3-05

Patches sector_breadth.py to add MCap-weighted SMA50 breadth alongside the
existing count-based breadth. Run this ONCE from F:\\nse_momentum.

Changes made (all additive, nothing existing is altered):
  1. Adds _load_mcap_weights() function — reads ticker_mcap from Turso.
  2. In compute_sector_breadth(): loads MCap weights and accumulates
     mcap_total / mcap_above_sma50 per sector alongside existing count logic.
  3. In results.append(): adds pct_above_sma50_mcap field.
  4. In _ensure_bhav_columns(): adds pct_above_sma50_mcap column to Turso.
  5. In publish_to_turso(): writes pct_above_sma50_mcap to Turso.

Why only SMA50 for MCap weighting:
  SMA50 is the primary trend signal used in P3-07's composite rotation score
  (it's what the sector concentration alert and LiveSectorBreadth score both
  key off). Adding MCap weighting to all 5 breadth signals would add columns
  without clear use cases. SMA50 MCap-weighted is the one that matters for
  the "2-3 megacaps carrying breadth" regime detection described in P3-05.

Usage:
    python patch_sector_breadth_p305.py          # applies the patch
    python patch_sector_breadth_p305.py --check  # dry-run: shows what would change
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("sector_breadth.py")


def check_already_patched(content: str) -> bool:
    return "_load_mcap_weights" in content


def apply_patch(content: str) -> str:
    # ── 1. Add _load_mcap_weights() after _load_bhavcopy_metrics() ──────────
    # Insert before def _compute_stock_flags
    mcap_fn = '''
def _load_mcap_weights() -> dict:
    """
    P3-05: {ticker (with .NS): mcap_cr (float)} from Turso ticker_mcap table.
    Built by fetch_mcap.py from NSE's official daily MCap report.
    Returns {} on any failure — MCap weighting gracefully degrades to None
    for affected sectors, consistent with the P1-06 fallback pattern used
    throughout this codebase (Bhavcopy unavailability never breaks price breadth).
    """
    try:
        from turso_sync import get_client
        client = get_client()
    except (SystemExit, Exception) as e:
        log.warning(f"  MCap weights unavailable ({e}) — pct_above_sma50_mcap will be None")
        return {}
    try:
        rs = client.execute("SELECT ticker, mcap_cr FROM ticker_mcap")
        result = {row[0]: float(row[1]) for row in rs.rows if row[1]}
        log.info(f"  MCap weights loaded for {len(result)} tickers (P3-05)")
        return result
    except Exception as e:
        log.warning(f"  ticker_mcap read failed ({e}) — pct_above_sma50_mcap will be None")
        return {}
    finally:
        client.close()


'''
    content = content.replace(
        "def _compute_stock_flags(close: np.ndarray) -> dict:",
        mcap_fn + "def _compute_stock_flags(close: np.ndarray) -> dict:"
    )

    # ── 2. In compute_sector_breadth(): load MCap weights ───────────────────
    content = content.replace(
        '    log.info("  Loading Bhavcopy volume-quality metrics (P3-06)...")',
        '    log.info("  Loading Bhavcopy volume-quality metrics (P3-06)...")\n'
        '    log.info("  Loading MCap weights (P3-05)...")\n'
        '    mcap_weights = _load_mcap_weights()'
    )

    # ── 3. In sector_flags defaultdict: add mcap accumulators ───────────────
    content = content.replace(
        '        "above_vwap": 0, "high_delivery": 0, "turnover_sum": 0.0, "bhav_coverage": 0,',
        '        "above_vwap": 0, "high_delivery": 0, "turnover_sum": 0.0, "bhav_coverage": 0,\n'
        '        "mcap_total": 0.0, "mcap_above_sma50": 0.0,  # P3-05'
    )

    # ── 4. In per-ticker loop: accumulate MCap after existing flag logic ─────
    content = content.replace(
        '        # P3-06: fold in Bhavcopy metrics for this ticker, if it matched.',
        '        # P3-05: accumulate MCap weighting for SMA50 breadth.\n'
        '        ticker_mcap = mcap_weights.get(ticker)\n'
        '        if ticker_mcap and ticker_mcap > 0 and flags is not None:\n'
        '            s["mcap_total"] += ticker_mcap\n'
        '            if flags["above_sma50"]:\n'
        '                s["mcap_above_sma50"] += ticker_mcap\n'
        '\n'
        '        # P3-06: fold in Bhavcopy metrics for this ticker, if it matched.'
    )

    # ── 5. In results.append(): add pct_above_sma50_mcap field ─────────────
    content = content.replace(
        '            "bhav_coverage":     bhav_n,\n        })',
        '            "bhav_coverage":     bhav_n,\n'
        '            # P3-05: MCap-weighted SMA50 breadth — None if no MCap data for sector.\n'
        '            "pct_above_sma50_mcap": (\n'
        '                round(s["mcap_above_sma50"] / s["mcap_total"] * 100, 1)\n'
        '                if s["mcap_total"] > 0 else None\n'
        '            ),\n'
        '        })'
    )

    # ── 6. In _ensure_bhav_columns(): add new column ────────────────────────
    content = content.replace(
        '        "bhav_coverage":     "INTEGER",\n    }',
        '        "bhav_coverage":     "INTEGER",\n'
        '        "pct_above_sma50_mcap": "REAL",   # P3-05\n'
        '    }'
    )

    # ── 7. In publish_to_turso INSERT: add column and value ─────────────────
    content = content.replace(
        '                    INSERT INTO sector_breadth (\n'
        '                        sector, breadth_date, pct_above_sma20, pct_above_sma50,\n'
        '                        pct_above_sma100, pct_above_rsi50, pct_above_rs55,\n'
        '                        stock_count, pct_above_vwap, pct_high_delivery,\n'
        '                        avg_turnover_cr, bhav_coverage, published_at\n'
        '                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        '                    INSERT INTO sector_breadth (\n'
        '                        sector, breadth_date, pct_above_sma20, pct_above_sma50,\n'
        '                        pct_above_sma100, pct_above_rsi50, pct_above_rs55,\n'
        '                        stock_count, pct_above_vwap, pct_high_delivery,\n'
        '                        avg_turnover_cr, bhav_coverage, pct_above_sma50_mcap, published_at\n'
        '                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    )

    # Update the ON CONFLICT SET clause
    content = content.replace(
        '                        pct_above_sma50=excluded.pct_above_sma50,\n'
        '                        pct_above_sma100=excluded.pct_above_sma100,\n'
        '                        pct_above_rsi50=excluded.pct_above_rsi50,\n'
        '                        pct_above_rs55=excluded.pct_above_rs55,\n'
        '                        stock_count=excluded.stock_count,\n'
        '                        pct_above_vwap=excluded.pct_above_vwap,\n'
        '                        pct_high_delivery=excluded.pct_high_delivery,\n'
        '                        avg_turnover_cr=excluded.avg_turnover_cr,\n'
        '                        bhav_coverage=excluded.bhav_coverage,\n'
        '                        published_at=excluded.published_at',
        '                        pct_above_sma50=excluded.pct_above_sma50,\n'
        '                        pct_above_sma100=excluded.pct_above_sma100,\n'
        '                        pct_above_rsi50=excluded.pct_above_rsi50,\n'
        '                        pct_above_rs55=excluded.pct_above_rs55,\n'
        '                        stock_count=excluded.stock_count,\n'
        '                        pct_above_vwap=excluded.pct_above_vwap,\n'
        '                        pct_high_delivery=excluded.pct_high_delivery,\n'
        '                        avg_turnover_cr=excluded.avg_turnover_cr,\n'
        '                        bhav_coverage=excluded.bhav_coverage,\n'
        '                        pct_above_sma50_mcap=excluded.pct_above_sma50_mcap,\n'
        '                        published_at=excluded.published_at'
    )

    # Update the values list — add r["pct_above_sma50_mcap"] before now
    content = content.replace(
        '                     r["stock_count"], r["pct_above_vwap"], r["pct_high_delivery"],\n'
        '                     r["avg_turnover_cr"], r["bhav_coverage"], now],',
        '                     r["stock_count"], r["pct_above_vwap"], r["pct_high_delivery"],\n'
        '                     r["avg_turnover_cr"], r["bhav_coverage"],\n'
        '                     r.get("pct_above_sma50_mcap"), now],'
    )

    return content


def run(check_only: bool):
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from F:\\nse_momentum.")
        sys.exit(1)

    content = TARGET.read_text(encoding="utf-8")

    if check_already_patched(content):
        print("Already patched — _load_mcap_weights already present in sector_breadth.py")
        sys.exit(0)

    patched = apply_patch(content)

    if check_only:
        # Show a diff summary
        orig_lines = content.splitlines()
        new_lines  = patched.splitlines()
        added = len(new_lines) - len(orig_lines)
        print(f"[DRY RUN] Would add ~{added} lines to sector_breadth.py")
        print("Changes:")
        print("  + _load_mcap_weights() function")
        print("  + MCap weight loading in compute_sector_breadth()")
        print("  + mcap_total / mcap_above_sma50 accumulators in sector_flags")
        print("  + Per-ticker MCap accumulation in the ticker loop")
        print("  + pct_above_sma50_mcap field in results.append()")
        print("  + pct_above_sma50_mcap column in _ensure_bhav_columns()")
        print("  + pct_above_sma50_mcap in publish_to_turso() INSERT + UPDATE")
        return

    # Backup
    backup = f"sector_breadth.py.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(TARGET, backup)
    print(f"Backup: {backup}")

    TARGET.write_text(patched, encoding="utf-8")
    print(f"Patched: {TARGET}")
    print("\nVerify with:")
    print("  python -m py_compile sector_breadth.py && echo SYNTAX OK")
    print("  python sector_breadth.py --dry-run")
    print("\nExpected: pct_above_sma50_mcap column in the output table.")
    print("If MCap data is in Turso, each sector should show a value.")
    print("Financial Services will likely show a lower MCap-weighted SMA50 than")
    print("count-based (megacaps like HDFCBANK/ICICIBANK skew the sector reading).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P3-05: patch sector_breadth.py")
    ap.add_argument("--check", action="store_true", help="Dry-run: show what would change")
    args = ap.parse_args()
    run(check_only=args.check)
