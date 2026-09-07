"""
sector_novelty.py

Flags sectors making their first appearance in the evening scan's picks in
the last LOOKBACK_DAYS calendar days, so a newly-emerging theme (e.g. a
Steel/Metal rotation showing up in Tier 3 / Near-Breakout before anything
else confirms it) is visually impossible to overlook instead of blending
into an unmarked sector column.

[2026-09-07] Added per user feedback: JSWSTEEL correctly surfaced in the
04-Sep-2026 email's Tier 3 watchlist as the first Steel/Metal name of the
week, but nothing distinguished it from a sector that's been showing up
for a month. See also the same day's JINDALSTEL Near-Breakout entry --
same theme, same blind spot.

Persistence: Turso (via turso_sync.publish_sector_history /
get_recent_sectors) is the source of truth, NOT a local file. The evening
scan runs on GitHub Actions with a fresh checkout every run (see
daily_scan.yml) -- anything written to the local filesystem is gone before
the next run, so a gitignored JSON here would silently never accumulate
any real history in production and this feature would never fire. Turso
is the same durable store scanner_signals/scan_metadata already use for
exactly this reason.

A local JSON cache (SECTOR_HISTORY_LOCAL_FALLBACK) is kept ONLY as a
best-effort fallback for ad-hoc/local runs with no Turso credentials
configured (e.g. single_stock.py, local dry-runs) -- it does NOT persist
across GitHub Actions runs and must never be relied on for the real
production signal.

Evidence-first note: "new" here means "not published to picks in the
lookback window," which is exactly what the user asked for and is easy to
verify by querying sector_pick_history directly -- no backtested threshold
or scoring model involved, so this intentionally stays out of scope for
the validate-before-wiring process that applies to actual trading-signal
candidates.
"""

import json
import logging
import os
from datetime import date as _date, timedelta

log = logging.getLogger(__name__)

# Local-only fallback path, used solely when Turso is unreachable (see
# module docstring -- this is NOT the production persistence mechanism).
SECTOR_HISTORY_LOCAL_FALLBACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sector_pick_history.json"
)

# How far back a sector must be ABSENT from picks to count as "new" again.
# ~2 trading weeks -- long enough that a sector merely skipping a single
# choppy day doesn't get re-flagged, short enough to still catch a real
# rotation while it's fresh.
LOOKBACK_DAYS = 15

# Local-fallback log prune horizon so it never grows unbounded.
MAX_HISTORY_DAYS = 60


def _load_local_history() -> dict:
    if not os.path.exists(SECTOR_HISTORY_LOCAL_FALLBACK):
        return {}
    try:
        with open(SECTOR_HISTORY_LOCAL_FALLBACK) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"sector_pick_history.json unreadable ({e}) -- starting fresh")
        return {}


def _save_local_history(hist: dict) -> None:
    try:
        with open(SECTOR_HISTORY_LOCAL_FALLBACK, "w") as f:
            json.dump(hist, f, indent=2, sort_keys=True)
    except OSError as e:
        log.warning(f"Could not write sector_pick_history.json: {e}")


def _recent_sectors_local_fallback(scan_date: str, lookback_days: int):
    """Mirrors turso_sync.get_recent_sectors()'s (prior_sectors, has_any_history)
    contract, but against the local JSON cache. Only used when Turso is
    unreachable -- see module docstring."""
    hist = _load_local_history()
    cutoff = (_date.fromisoformat(scan_date) - timedelta(days=lookback_days)).isoformat()
    prior_days = [d for d in hist if cutoff <= d < scan_date]
    has_any_history = any(d < scan_date for d in hist)
    prior_sectors = set()
    for d in prior_days:
        prior_sectors.update(hist[d])
    return prior_sectors, has_any_history


def _record_local_fallback(scan_date: str, sectors: set) -> None:
    hist = _load_local_history()
    hist[scan_date] = sorted(sectors)
    horizon = (_date.fromisoformat(scan_date) - timedelta(days=MAX_HISTORY_DAYS)).isoformat()
    hist = {d: s for d, s in hist.items() if d >= horizon}
    _save_local_history(hist)


def mark_new_sectors(results: list, near_breakout: list,
                      scan_date: str = None, dry_run: bool = False) -> set:
    """
    Sets `.sector_is_new` (bool) on each StockResult in `results` (T1+T2+T3
    picks -- these are the same object instances referenced inside
    orchestrator.py's `all_results`, so the mutation is visible there too,
    e.g. to emailer.py's Top-20 table) and `["is_new_sector"]` on each dict
    in `near_breakout`, then publishes today's sector set to Turso
    (sector_pick_history) so tomorrow's run can see it.

    scan_date: the verified Bhavcopy trading date (orchestrator.py's
    `effective_scan_date`), NOT date.today() -- for the same reason
    picks_latest.json's meta uses it: wall-clock date can lag the actual
    data vintage on a stale/fallback run.

    dry_run: forwarded to turso_sync.publish_sector_history so a test run
    never writes into the shared prod table (P1-06) -- flags are still
    computed and rendered for a dry-run email, just not persisted.

    Returns the set of sector names flagged new (for logging).

    Failure isolation: any exception here is the caller's problem to catch
    (see orchestrator.py call site) -- this must never be allowed to break
    the evening email, same pattern as auto_log_t1_picks/turso_sync.
    """
    scan_date = scan_date or _date.today().isoformat()

    today_sectors = set()
    for r in results:
        sec = getattr(r, "sector", None)
        if sec:
            today_sectors.add(sec)
    for nb in near_breakout:
        sec = nb.get("sector")
        if sec:
            today_sectors.add(sec)

    # Check credentials directly rather than relying on turso_sync's own
    # try/except: get_client() there calls sys.exit() on missing creds,
    # which get_recent_sectors/publish_sector_history already catch
    # internally and turn into a benign (set(), False)/0 return -- so a
    # bare try/except here around the *call* would never actually observe
    # "Turso isn't configured" and would wrongly skip the local fallback.
    prior_sectors, has_any_history, used_turso = set(), False, False
    try:
        import turso_sync
        used_turso = bool(turso_sync.TURSO_URL and turso_sync.TURSO_TOKEN)
        if used_turso:
            prior_sectors, has_any_history = turso_sync.get_recent_sectors(
                scan_date, LOOKBACK_DAYS
            )
    except Exception as e:
        log.warning(f"  sector_novelty: Turso read failed ({e}) -- "
                    f"falling back to local-only history (won't persist on CI)")
        used_turso = False

    if not used_turso:
        prior_sectors, has_any_history = _recent_sectors_local_fallback(scan_date, LOOKBACK_DAYS)

    # No prior days logged at all (fresh install, or the table is genuinely
    # empty) -- every sector would trivially be "new" on day one, which
    # isn't a useful signal. Seed the log silently instead of flagging the
    # whole email.
    new_sectors = (today_sectors - prior_sectors) if has_any_history else set()

    for r in results:
        r.sector_is_new = getattr(r, "sector", None) in new_sectors
    for nb in near_breakout:
        nb["is_new_sector"] = nb.get("sector") in new_sectors

    if used_turso:
        try:
            import turso_sync
            turso_sync.publish_sector_history(today_sectors, scan_date=scan_date, dry_run=dry_run)
        except Exception as e:
            log.warning(f"  sector_novelty: Turso publish failed ({e}) -- "
                        f"today's sectors won't be recorded for tomorrow's run")
    else:
        # Local-only fallback path (no Turso credentials available at all,
        # e.g. an ad-hoc local run) -- won't survive a GitHub Actions
        # checkout, but keeps local/dev iteration self-consistent.
        _record_local_fallback(scan_date, today_sectors)

    if new_sectors:
        log.info(f"  New sector(s) this run (first appearance in {LOOKBACK_DAYS}d): "
                  f"{', '.join(sorted(new_sectors))}")

    return new_sectors
