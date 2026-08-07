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

2. **Migrate Portfolio Dashboard's `db.py` from `libsql` to `libsql_client`**
   (F:\PortfolioDashboard) — `libsql`'s embedded-replica `conn.sync()` hangs
   indefinitely in this environment (confirmed twice, not a network issue —
   curl reaches the same Turso host in <1s). `db.py`'s own docstring
   explains it was switched TO `libsql` FROM `libsql_client` after a past
   `WSServerHandshakeError` when Turso deprecated the old websocket/hrana
   protocol — but nse_momentum's `turso_sync.py` uses `libsql_client`
   successfully against this exact same database throughout this session,
   suggesting that issue is resolved and `libsql_client` is now the
   healthier choice. This is a real rewrite (different cursor/commit
   semantics), not a one-line fix.

3. **Re-enable Turso in Portfolio Dashboard's `.env`** — currently commented
   out (`TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN`) as an immediate unblock so
   the app loads past login. Running on a stale local `portfolio.db`
   (dated 27 Jul — missing MANAPPURAM/ASTERDM/SAILIFE, has since-removed
   holdings like DIVISLAB/SMLMAH) until item 2 is done and these are
   uncommented again.

4. **Optional — cosmetic**: `company_name` for the two backfilled holdings
   (id 22, 23 in the `holdings` table) still reads "ASTER DM QUALITY CARE"
   / "SAI LIFE SCIENCES" instead of their real NSE names ("Aster DM Quality
   Care Limited" / "Sai Life Sciences Limited"). Never affected anything
   functionally (display-only field) — low priority, only worth doing if
   it's bothering you.

## Done (2026-08-08)

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
