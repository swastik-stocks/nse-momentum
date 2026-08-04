"""
signal_attribution.py — P4-02

Signal→outcome attribution: joins scanner picks (trades_v4, populated by
orchestrator.py's auto_log_t1_picks every evening) against broker-confirmed
realized trades (realized_trades, populated by import_broker_trades.py /
P4-01) to answer the real-money version of the backtest question:

    "Are the scanner's actual live signals producing real profits?"

This is distinct from validation/backtest.py and pipeline_replay_deep.py,
which test HISTORICAL signals against simulated forward returns. This tests
LIVE signals (from the real gate chain, real regime, real day's data) against
REAL broker execution — the only honest answer to whether the scanner works
with actual capital.

Join logic:
    trades_v4.ticker = realized_trades.ticker
    AND realized_trades.buy_date >= trades_v4.entry_date
    AND realized_trades.buy_date <= trades_v4.entry_date + 5 trading days
    (5-day window: signal fires EOD, you have up to 5 sessions to enter —
    matches realistic manual execution latency without contaminating the match
    with unrelated positions opened much later)

Attribution output (per matched signal):
    - scanner score, pattern, regime at signal time (from trades_v4)
    - actual buy/sell dates, net_pnl_pct, holding_days (from realized_trades)
    - signal_worked: net_pnl_pct > 0 (cost-inclusive, since realized_trades
      already stores net_pnl_pct after broker charges from import_broker_trades.py)

Aggregate output (published to Turso signal_attribution table + email):
    - per-pattern: N matched trades, win rate, avg net_pnl_pct, avg holding_days
    - per-score-bucket (50-59, 60-69, 70-79, 80+): same metrics
    - per-regime: same metrics
    - overall: matched vs unmatched signal count, total realized P&L

Usage:
    python signal_attribution.py              # full run + publish to Turso
    python signal_attribution.py --dry-run    # compute + print, skip Turso
    python signal_attribution.py --stats      # print current attribution table
"""

import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ENTRY_WINDOW_DAYS = 5   # how many calendar days after signal to accept a buy as "matched"


def _load_data() -> tuple:
    """
    Returns (signals: list[dict], realized: list[dict]).
    Reads trades_v4 (scanner signals) and realized_trades (broker data)
    from the local SQLite DB via database.schema.get_connection().
    Both return [] on failure — caller handles gracefully.
    """
    from database.schema import get_connection
    try:
        conn = get_connection()

        signals = []
        for row in conn.execute("""
            SELECT ticker, entry_date, pattern, total_score, regime,
                   breadth_score, status, r_multiple, pnl_pct, universe
            FROM trades_v4
            WHERE entry_date IS NOT NULL
            ORDER BY entry_date DESC
        """):
            signals.append({
                "ticker":       row[0],
                "entry_date":   row[1],
                "pattern":      row[2] or "Unknown",
                "total_score":  row[3] or 0,
                "regime":       row[4] or "?",
                "breadth_score": row[5] or 0,
                "status":       row[6] or "OPEN",
                "r_multiple":   row[7],
                "pnl_pct":      row[8],
                "tier":         row[9] or "LARGE",  # column is 'universe' in trades_v4
            })

        realized = []
        for row in conn.execute("""
            SELECT ticker, buy_date, sell_date, quantity,
                   buy_price, sell_price, net_pnl, net_pnl_pct, holding_days
            FROM realized_trades
            ORDER BY buy_date DESC
        """):
            realized.append({
                "ticker":       row[0],
                "buy_date":     row[1],
                "sell_date":    row[2],
                "quantity":     row[3],
                "buy_price":    row[4],
                "sell_price":   row[5],
                "net_pnl":      row[6],
                "net_pnl_pct":  row[7],
                "holding_days": row[8],
            })

        conn.close()
        log.info(f"  Loaded {len(signals)} scanner signals, {len(realized)} realized trades")
        return signals, realized

    except Exception as e:
        log.error(f"Failed to load attribution data: {e}")
        return [], []


def _match_signals(signals: list, realized: list) -> list:
    """
    Joins signals → realized trades on ticker + date proximity.
    One realized trade can match at most one signal (the closest preceding
    signal within ENTRY_WINDOW_DAYS). One signal can match at most one
    realized trade (the earliest buy within the window).

    Returns list of matched dicts with fields from both sides + derived
    attribution fields.
    """
    # Index realized trades by ticker for fast lookup
    by_ticker = {}
    for r in realized:
        by_ticker.setdefault(r["ticker"], []).append(r)

    matched = []
    used_realized_ids = set()

    for sig in signals:
        ticker = sig["ticker"]
        candidates = by_ticker.get(ticker, [])
        if not candidates:
            continue

        try:
            signal_date = datetime.strptime(sig["entry_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        window_end = signal_date + timedelta(days=ENTRY_WINDOW_DAYS)

        # Find earliest buy within the entry window that hasn't already been matched
        best = None
        for r in sorted(candidates, key=lambda x: x["buy_date"] or ""):
            try:
                buy_date = datetime.strptime(r["buy_date"], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            rid = (r["ticker"], r["buy_date"], r["sell_date"])
            if signal_date <= buy_date <= window_end and rid not in used_realized_ids:
                best = r
                break

        if best is None:
            continue

        rid = (best["ticker"], best["buy_date"], best["sell_date"])
        used_realized_ids.add(rid)

        net_pnl_pct = best["net_pnl_pct"] or 0.0
        signal_worked = net_pnl_pct > 0

        score = sig["total_score"]
        score_bucket = (f"{(score // 10) * 10}-{(score // 10) * 10 + 9}"
                        if score else "Unknown")

        matched.append({
            # Signal side
            "ticker":           ticker,
            "signal_date":      sig["entry_date"],
            "pattern":          sig["pattern"],
            "total_score":      sig["total_score"],
            "score_bucket":     score_bucket,
            "regime":           sig["regime"],
            "tier":             sig["tier"],
            "breadth_score":    sig["breadth_score"],
            # Realized side
            "buy_date":         best["buy_date"],
            "sell_date":        best["sell_date"],
            "buy_price":        best["buy_price"],
            "sell_price":       best["sell_price"],
            "net_pnl":          best["net_pnl"],
            "net_pnl_pct":      net_pnl_pct,
            "holding_days":     best["holding_days"],
            # Attribution
            "signal_worked":    signal_worked,
            "days_to_entry":    (datetime.strptime(best["buy_date"], "%Y-%m-%d").date()
                                 - datetime.strptime(sig["entry_date"], "%Y-%m-%d").date()).days
                                if best["buy_date"] and sig["entry_date"] else None,
        })

    log.info(f"  Matched {len(matched)} signals to realized trades "
             f"(out of {len(signals)} signals, {len(realized)} realized trades)")
    return matched


def _aggregate(matched: list) -> dict:
    """
    Aggregates matched trades into per-pattern, per-score-bucket,
    per-regime summary dicts. Each summary has:
    {n, wins, win_rate, avg_net_pnl_pct, avg_holding_days, total_net_pnl}
    """
    def _summarise(rows):
        if not rows:
            return {}
        wins = sum(1 for r in rows if r["signal_worked"])
        return {
            "n":                len(rows),
            "wins":             wins,
            "win_rate":         round(wins / len(rows) * 100, 1),
            "avg_net_pnl_pct":  round(sum(r["net_pnl_pct"] for r in rows) / len(rows), 2),
            "avg_holding_days": round(sum((r["holding_days"] or 0) for r in rows) / len(rows), 1),
            "total_net_pnl":    round(sum((r["net_pnl"] or 0) for r in rows), 0),
        }

    by_pattern = {}
    by_bucket  = {}
    by_regime  = {}
    for r in matched:
        by_pattern.setdefault(r["pattern"], []).append(r)
        by_bucket.setdefault(r["score_bucket"], []).append(r)
        by_regime.setdefault(r["regime"], []).append(r)

    return {
        "by_pattern": {k: _summarise(v) for k, v in sorted(by_pattern.items())},
        "by_score_bucket": {k: _summarise(v)
                            for k, v in sorted(by_bucket.items())},
        "by_regime":  {k: _summarise(v) for k, v in sorted(by_regime.items())},
        "overall":    _summarise(matched),
    }


def compute_signal_attribution() -> dict:
    """
    Full pipeline: load → match → aggregate.
    Returns {matched: list, aggregates: dict, as_of_date: str}.
    Returns empty dict on any load failure.
    """
    signals, realized = _load_data()
    if not signals or not realized:
        log.warning("  Insufficient data for attribution — need both trades_v4 "
                    "and realized_trades populated. Run import_broker_trades.py (P4-01) first.")
        return {}

    matched = _match_signals(signals, realized)
    aggregates = _aggregate(matched)
    return {
        "matched":    matched,
        "aggregates": aggregates,
        "as_of_date": date.today().isoformat(),
        "n_signals":  len(signals),
        "n_realized": len(realized),
        "n_matched":  len(matched),
    }


def publish_to_turso(result: dict) -> int:
    """
    Publishes aggregated attribution to Turso signal_attribution_summary
    table, one row per (dimension_type, dimension_value, as_of_date).
    Also publishes matched trade-level detail to signal_attribution_matches.
    Returns total rows published.
    """
    from turso_sync import get_client
    if not result:
        return 0

    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Publish skipped — {e}")
        return 0

    now = datetime.now().isoformat()
    as_of = result["as_of_date"]
    published = 0

    try:
        client.execute("""
            CREATE TABLE IF NOT EXISTS signal_attribution_summary (
                dimension_type   TEXT NOT NULL,
                dimension_value  TEXT NOT NULL,
                as_of_date       TEXT NOT NULL,
                n                INTEGER,
                wins             INTEGER,
                win_rate         REAL,
                avg_net_pnl_pct  REAL,
                avg_holding_days REAL,
                total_net_pnl    REAL,
                published_at     TEXT,
                PRIMARY KEY (dimension_type, dimension_value, as_of_date)
            )
        """)
        client.execute("""
            CREATE TABLE IF NOT EXISTS signal_attribution_matches (
                ticker           TEXT NOT NULL,
                signal_date      TEXT NOT NULL,
                buy_date         TEXT,
                pattern          TEXT,
                total_score      INTEGER,
                regime           TEXT,
                net_pnl_pct      REAL,
                net_pnl          REAL,
                holding_days     INTEGER,
                signal_worked    INTEGER,
                days_to_entry    INTEGER,
                as_of_date       TEXT NOT NULL,
                published_at     TEXT,
                PRIMARY KEY (ticker, signal_date, as_of_date)
            )
        """)

        agg = result["aggregates"]
        for dim_type, dim_dict in [("pattern",      agg["by_pattern"]),
                                    ("score_bucket", agg["by_score_bucket"]),
                                    ("regime",       agg["by_regime"])]:
            for dim_val, s in dim_dict.items():
                if not s:
                    continue
                try:
                    client.execute("""
                        INSERT INTO signal_attribution_summary
                        (dimension_type, dimension_value, as_of_date, n, wins,
                         win_rate, avg_net_pnl_pct, avg_holding_days, total_net_pnl,
                         published_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(dimension_type, dimension_value, as_of_date)
                        DO UPDATE SET
                            n=excluded.n, wins=excluded.wins,
                            win_rate=excluded.win_rate,
                            avg_net_pnl_pct=excluded.avg_net_pnl_pct,
                            avg_holding_days=excluded.avg_holding_days,
                            total_net_pnl=excluded.total_net_pnl,
                            published_at=excluded.published_at
                    """, [dim_type, str(dim_val), as_of, s["n"], s["wins"],
                          s["win_rate"], s["avg_net_pnl_pct"],
                          s["avg_holding_days"], s["total_net_pnl"], now])
                    published += 1
                except Exception as e:
                    log.warning(f"  Failed to publish {dim_type}={dim_val}: {e}")

        for m in result["matched"]:
            try:
                client.execute("""
                    INSERT INTO signal_attribution_matches
                    (ticker, signal_date, buy_date, pattern, total_score,
                     regime, net_pnl_pct, net_pnl, holding_days, signal_worked,
                     days_to_entry, as_of_date, published_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(ticker, signal_date, as_of_date) DO UPDATE SET
                        buy_date=excluded.buy_date,
                        net_pnl_pct=excluded.net_pnl_pct,
                        net_pnl=excluded.net_pnl,
                        holding_days=excluded.holding_days,
                        signal_worked=excluded.signal_worked,
                        published_at=excluded.published_at
                """, [m["ticker"], m["signal_date"], m["buy_date"],
                      m["pattern"], m["total_score"], m["regime"],
                      m["net_pnl_pct"], m["net_pnl"], m["holding_days"],
                      int(m["signal_worked"]), m["days_to_entry"],
                      as_of, now])
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish match {m['ticker']}/{m['signal_date']}: {e}")

    finally:
        client.close()

    return published


def print_attribution(result: dict):
    if not result:
        print("No attribution data — see log above.")
        return

    agg = result["aggregates"]
    overall = agg.get("overall", {})

    print(f"\n{'='*80}")
    print(f"  SIGNAL→OUTCOME ATTRIBUTION  (as of {result['as_of_date']})")
    print(f"  {result['n_signals']} scanner signals | {result['n_realized']} realized trades "
          f"| {result['n_matched']} matched")
    print(f"{'='*80}")

    def _print_section(title, data):
        if not data:
            return
        print(f"\n  {title}")
        print(f"  {'':25}{'N':>5}{'WIN%':>7}{'AVG P&L%':>10}{'AVG DAYS':>10}{'TOTAL P&L':>14}")
        print("  " + "-"*70)
        for k, s in sorted(data.items(), key=lambda x: -(x[1].get("avg_net_pnl_pct") or 0)):
            if not s:
                continue
            print(f"  {str(k):<25}{s['n']:>5}{s['win_rate']:>6.0f}%"
                  f"{s['avg_net_pnl_pct']:>9.2f}%{s['avg_holding_days']:>10.1f}"
                  f"  ₹{s['total_net_pnl']:>11,.0f}")

    _print_section("BY PATTERN",      agg.get("by_pattern", {}))
    _print_section("BY SCORE BUCKET", agg.get("by_score_bucket", {}))
    _print_section("BY REGIME",       agg.get("by_regime", {}))

    if overall:
        print(f"\n  OVERALL: {overall['n']} trades | "
              f"win rate {overall['win_rate']:.0f}% | "
              f"avg P&L {overall['avg_net_pnl_pct']:.2f}% | "
              f"total ₹{overall['total_net_pnl']:,.0f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print, skip Turso publish")
    ap.add_argument("--stats", action="store_true",
                    help="Print last published attribution from Turso")
    args = ap.parse_args()

    if args.stats:
        from turso_sync import get_client
        try:
            client = get_client()
            rows = client.execute("""
                SELECT dimension_type, dimension_value, n, win_rate,
                       avg_net_pnl_pct, total_net_pnl, as_of_date
                FROM signal_attribution_summary
                ORDER BY dimension_type, avg_net_pnl_pct DESC
            """)
            r = rows.rows if hasattr(rows, "rows") else rows
            print(f"\n{'DIM':<15}{'VALUE':<25}{'N':>5}{'WIN%':>7}{'AVG%':>9}"
                  f"{'TOTAL P&L':>14}{'DATE':>12}")
            print("-"*90)
            for row in r:
                print(f"{row[0]:<15}{str(row[1]):<25}{row[2]:>5}"
                      f"{row[3]:>6.0f}%{row[4]:>8.2f}%  ₹{row[5]:>10,.0f}  {row[6]}")
            client.close()
        except Exception as e:
            print(f"Could not read from Turso: {e}")
    else:
        result = compute_signal_attribution()
        print_attribution(result)
        if not args.dry_run and result:
            n = publish_to_turso(result)
            print(f"Published {n} rows to Turso signal_attribution tables.")
        elif args.dry_run:
            print(f"[DRY RUN] Would publish to Turso — skipped.")
