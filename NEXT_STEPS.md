# Next Steps — P1 Symbol Resolver follow-ups

Tracking doc for what's left after the P1 symbol resolver work (2026-08-08).
See `check_scan_done.py`/`check_scan_health.py`/`daily_scan.yml` for the P0
scheduler work, and `symbol_resolver.py`/`publish_symbol_master.py`/
`backfill_holdings_symbols.py` for P1. Full plan history isn't preserved
elsewhere — this file is the durable record of what's still open.

## Pending

1. **Confirm `get_holdings` fix + universe union in a real scan run**
   (nse_momentum) — `turso_sync.py`'s missing `get_holdings` definition was
   fixed and pushed (commit `d0774ac`), and `scanner.py` now folds held
   tickers missing from the static universe into the scan (commit
   `691b172`). Neither has been confirmed in an actual GitHub Actions run
   yet — the test run done during this session predates both fixes. Check
   the next real run (Monday 19:30 IST, or trigger manually via Actions →
   "Run workflow" or cron-job.org's Test run) for:
   - `[P2-01] Held stocks evaluated:` log line + a populated Section 6
     (Position Alerts) in the email
   - `Folded 3 held ticker(s) not in the static universe...` log line
     (SBIFUNDS.NS, SILVERBEES.NS, YATHARTH.NS)

2. **Optional — cosmetic**: `company_name` for the two backfilled holdings
   (id 22, 23 in the `holdings` table) still reads "ASTER DM QUALITY CARE"
   / "SAI LIFE SCIENCES" instead of their real NSE names ("Aster DM Quality
   Care Limited" / "Sai Life Sciences Limited"). Never affected anything
   functionally (display-only field) — low priority, only worth doing if
   it's bothering you.

## Done (2026-08-08)

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
