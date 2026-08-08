# Next Steps — P1 Symbol Resolver follow-ups

Tracking doc for what's left after the P1 symbol resolver work (2026-08-08).
See `check_scan_done.py`/`check_scan_health.py`/`daily_scan.yml` for the P0
scheduler work, and `symbol_resolver.py`/`publish_symbol_master.py`/
`backfill_holdings_symbols.py` for P1. Full plan history isn't preserved
elsewhere — this file is the durable record of what's still open.

## Pending

1. **New finding, not part of P1 scope**: `turso_sync.publish_holding_stops()`
   is defined but never called anywhere in the codebase — dead code. It
   writes `position_actions` rows with `action_type='HOLD_STOP'`, which
   Portfolio Dashboard's `get_holding_stops()` reads for its "Portfolio
   Heat" feature — meaning that feature has never had real data, regardless
   of the `get_holdings` fix. Doesn't affect the evening email's Section 6
   (EXIT/TRIM/heat there is computed fresh in-memory each run, not read
   back from `position_actions`). Worth deciding whether to wire this in or
   remove it as unused.

## Done (2026-08-08)

- **Cosmetic `company_name` fix** — the two backfilled holdings (id 22, 23)
  now show their real NSE names ("Aster DM Quality Care Limited" / "Sai
  Life Sciences Limited") instead of the original free-text entries.

- **Confirmed `get_holdings` fix + universe union in a real scan run** —
  the 2026-08-08 scan (`scan_metadata.published_at` 05:48 UTC, after both
  fixes were pushed) shows `tickers_from_dhan=502` against a 500-stock
  static universe; `get_holdings()` currently returns 3 tickers missing
  from the static list (SBIFUNDS.NS, SILVERBEES.NS, YATHARTH.NS) — 500+3=503
  requested, 502 fetched via Dhan is consistent with one Dhan-side miss on
  a newer/smaller name, not an error. Confirms both fixes are live.

- **Portfolio Dashboard `libsql` → `libsql_client` migration** (see that
  repo's `NEXT_STEPS.md` for detail) — fixed the indefinite `conn.sync()`
  hang that was blocking Portfolio Dashboard from loading past login.
  Turso re-enabled in `.env`, verified working end-to-end (reads + writes)
  against production.

- P0 scheduler: cron-job.org as primary trigger, GitHub `schedule` as
  backstop, dead man's switch — all live and confirmed working.
- P1 symbol resolver: `symbol_resolver.py` extracted from
  `portfolio_watch.py`, `nse_symbol_master` published nightly to Turso,
  `ASTERDM.NS`/`SAILIFE.NS` backfilled and confirmed live, held tickers
  unioned into the scan universe, entry-time resolution UI + real reason
  codes shipped in Portfolio Dashboard.
- Found and fixed along the way: case-sensitivity bug in the fuzzy matcher
  (would have silently failed on the exact defect strings it targets), and
  the `turso_sync.py` `get_holdings` `ImportError` that had been silently
  emptying Section 6 of the evening email.
