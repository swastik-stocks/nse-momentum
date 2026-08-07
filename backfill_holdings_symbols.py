"""
backfill_holdings_symbols.py — P1-05

One-off script (not scheduled) to fix existing holdings rows in Turso whose
`symbol` column holds a free-text company name instead of a real NSE ticker
— e.g. "ASTER DM QUALITY CARE" / "SAI LIFE SCIENCES" — the exact defect that
motivated the symbol resolver. Portfolio Dashboard's add_holding() had no
validation until this same change added it (see F:\\PortfolioDashboard's
entry-time resolution), so rows entered before that fix need a one-time
correction.

SAFETY: default is dry-run — prints every row where the resolved symbol
differs from what's stored, but writes nothing. --apply only writes back
EXACT_SYMBOL/ISIN_EXACT/FUZZY_NAME matches; AMBIGUOUS_FUZZY and UNRESOLVED
rows are always printed for manual review and never auto-written, same
"never silently pick the wrong company" rule symbol_resolver.py documents.

Usage:
    python backfill_holdings_symbols.py            # dry-run, print proposed fixes
    python backfill_holdings_symbols.py --apply     # write back safe matches
"""

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from symbol_resolver import load_nse_master, build_upper_index, resolve_one

# Methods safe to auto-write — a resolver result this confident that the
# stored value is wrong. AMBIGUOUS_FUZZY/UNRESOLVED are deliberately
# excluded: see symbol_resolver.resolve_tickers()'s docstring on why a
# silent wrong-stock match on holdings data is a real, not cosmetic, risk.
AUTO_APPLY_METHODS = {"EXACT_SYMBOL", "ISIN_EXACT", "FUZZY_NAME"}


def fetch_holdings_rows() -> list:
    from turso_sync import get_client
    try:
        client = get_client()
    except SystemExit as e:
        log.error(f"Turso unavailable — {e}")
        sys.exit(1)
    try:
        rs = client.execute("SELECT id, symbol, company_name FROM holdings")
        return [{"id": r[0], "symbol": r[1], "company_name": r[2]} for r in rs.rows]
    finally:
        client.close()


def apply_fix(holding_id: int, new_symbol: str) -> None:
    from turso_sync import get_client
    client = get_client()
    try:
        client.execute("UPDATE holdings SET symbol = ? WHERE id = ?", [new_symbol, holding_id])
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser(
        description="P1-05: Backfill bad symbol values in Turso holdings table"
    )
    ap.add_argument("--apply", action="store_true",
                     help="Write back safe (non-ambiguous) fixes. Default is dry-run.")
    args = ap.parse_args()

    rows = fetch_holdings_rows()
    if not rows:
        log.info("No holdings rows found (or Turso holdings table is empty).")
        return

    log.info(f"Checking {len(rows)} holdings rows against the NSE symbol master...")
    isin_map, name_map = load_nse_master()
    upper_index = build_upper_index(name_map)
    symbol_set = set(name_map.values()) | set(isin_map.values())

    to_apply, to_review, unchanged = [], [], 0

    for row in rows:
        old_symbol = row["symbol"] or ""

        # Portfolio Dashboard's add_holding() always appends .NS/.BO if the
        # typed value lacks one (F:\PortfolioDashboard\app.py:134-137) — that
        # suffix is the established stored convention, not part of the bad
        # data. The NSE master (and resolve_one) works in bare symbols, so
        # strip it before resolving and re-attach it to whatever comes back,
        # rather than writing back a bare symbol that breaks every row's
        # format, correct rows included (confirmed against production data:
        # without this, all 18 already-correct rows got flagged as "fixes"
        # purely for losing their .NS suffix).
        exchange_suffix = ""
        bare_old = old_symbol
        for suf in (".NS", ".BO"):
            if old_symbol.upper().endswith(suf):
                exchange_suffix = suf
                bare_old = old_symbol[: -len(suf)]
                break

        # Prefer resolving from company_name when present (more likely to
        # be the real free-text problem); fall back to the bare symbol.
        source_text = row["company_name"] or bare_old
        resolved, method = resolve_one(str(source_text), None, isin_map, upper_index, symbol_set)

        if not resolved:
            unchanged += 1
            continue

        symbol = resolved + exchange_suffix
        if symbol == old_symbol:
            unchanged += 1
            continue

        entry = {"id": row["id"], "old": old_symbol, "new": symbol,
                  "method": method, "source": source_text}
        if method in AUTO_APPLY_METHODS:
            to_apply.append(entry)
        else:
            to_review.append(entry)

    print(f"\n{'ID':<6} {'OLD SYMBOL':<28} {'RESOLVED':<14} {'METHOD':<16} SOURCE")
    print("-" * 100)
    for e in to_apply:
        print(f"{e['id']:<6} {e['old']:<28} {e['new']:<14} {e['method']:<16} {e['source']}")
    if to_review:
        print(f"\n  --- NEEDS MANUAL REVIEW (not auto-applied even with --apply) ---")
        for e in to_review:
            print(f"{e['id']:<6} {e['old']:<28} {e['new']:<14} {e['method']:<16} {e['source']}")

    print(f"\n{len(to_apply)} safe fix(es), {len(to_review)} needing manual review, "
          f"{unchanged} already correct/unresolved.")

    if not args.apply:
        print("\nDry-run — no changes written. Re-run with --apply to write the safe fixes above.")
        return

    if not to_apply:
        print("\nNothing to apply.")
        return

    for e in to_apply:
        apply_fix(e["id"], e["new"])
        log.info(f"  Updated id={e['id']}: '{e['old']}' -> '{e['new']}'")
    print(f"\nApplied {len(to_apply)} fix(es).")


if __name__ == "__main__":
    main()
