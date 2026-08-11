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

2. **Confirm what P406 is meant to close out.** `p406_ca_flags.csv` /
   `migrate_p406_to_live.py` / `verify_p406_extension.py` are sitting
   untracked in the working tree — looks like an in-progress fix for
   unadjusted-split/bonus corporate-action noise in the price history
   (37+ tickers flagged with >40% single-day moves that are almost
   certainly bad data, not real price action). This is the same problem
   an external review (see below) flagged as "replace Yahoo Finance" —
   confirm P406 is already the fix in progress so it isn't duplicated.

## P0/P1 roadmap — from external feedback review (2026-08-11)

A consolidated "improvement roadmap" doc (unknown origin, not verified
against this codebase by its author) was reviewed against the actual code
before acting on any of it — see conversation 2026-08-11 for the full
verification. Several of its claims were wrong or already solved (listed
under "Rejected" below, kept here so a future session doesn't re-propose
them); the rest is real and prioritized below.

### P0 — do first (BTST is live and unvalidated/ungated on two real risks)

- [ ] **ASM/GSM hard-reject gate before BTST entry.** Confirmed absent —
      no NSE surveillance-list (Additional/Graded Surveillance Measure)
      handling anywhere in this repo. A stock under ASM/GSM has real
      trading restrictions (margin, price bands, sometimes trade-for-trade
      only) that make an overnight BTST hold specifically dangerous.
      Pull NSE's daily ASM/GSM list and add as a hard reject gate ahead of
      any BTST candidate in `confirm_picks.py`'s `classify_btst()`.
- [ ] **`agents/event_risk_agent.py`: hard reject (not score penalty) on
      earnings-eve for BTST.** Confirmed current behavior: only applies a
      -2/-5 score penalty near earnings (`_score_penalty` logic), never
      blocks. Fine for a same-day entry; not fine for holding a position
      into an earnings gap overnight. Needs a BTST-specific hard reject,
      separate from the existing penalty used by the morning scan.
- [ ] **Backtest `classify_btst()`'s actual thresholds** (2.0% off-high,
      1.5x RVOL, 70% T1-captured — see `confirm_picks.py`) via
      `monte_carlo_significance.py`-style validation on T+1 exits. These
      were reasoned out, not tested — same position VCP was in before its
      08-05 re-test. This codebase's whole culture is "don't trust a
      pattern until it clears Monte Carlo" (see `pattern_agent.py`
      docstring) — BTST shouldn't be the exception just because it's new.
- [ ] **Re-test VCP at current N.** Pruned 08-05 at N=70, p=0.1525 — the
      pattern's own detection logic is still live for exactly this
      purpose (`pattern_agent.py`: "still fires... for validation
      re-testing if a future replay with more data... wants to re-examine
      it"). Cheap, one `monte_carlo_significance.py --pattern VCP` run.

### P1 — next

- [ ] **Turn portfolio heat + sector concentration into real gates, not
      just reports.** Confirmed: `portfolio_heat.py` computes and
      publishes but explicitly does not block ("rather than blocking the
      email" — see its own module docstring); same for
      `sector_concentration_alert.py`, which orchestrator.py only threads
      into the email `tiers` dict, never into pick selection. Add an
      actual cap (e.g. total open 1R risk ≤ 6%, per-sector exposure ≤
      20%) that can veto/downsize a new pick.
- [ ] **Port walk-forward validation + discovery/holdout split into
      `monte_carlo_significance.py`.** Same idea already built and
      working in `E:\Simons Quant`'s `src/validation/walk_forward.py` +
      `bootstrap.py` — ties to the 2026-08-11 conversation reviewing that
      project. Not urgent for Cup & Handle/Swing High Breakout (both
      re-confirmed as N grew 32-41% with stable point estimates already —
      an informal version of the same check) but genuinely missing as a
      formal, always-run check for any future pattern.
- [ ] **Resolve the slippage + brokerage-rate cost-model question.**
      `pattern_agent.py` assumes 0.363% round-trip incl. 0.35%/leg
      brokerage; `E:\Simons Quant`'s `nse_costs.py` assumes 0.238% (zero
      brokerage, matching modern discount-broker delivery trades). Check
      against the real brokerage plan — if delivery trades are genuinely
      free, every backtested pattern's net R is currently understated by
      ~12bps round-trip. Separately, the flat cost model still has no
      market-impact/slippage term for low-ADT small-caps.

### Rejected / already done — do not re-propose these

- ~~Paper-trading verification~~ — **already built and better than what
  was proposed.** `signal_attribution.py` (P4-02) joins live scanner
  signals against real broker-confirmed trades (`import_broker_trades.py`,
  P4-01) — answers "does this work with real capital," not a simulation.
- ~~Replace Yahoo Finance (37 CA-flagged tickers)~~ — see pending item 2
  above, already in progress as P406.
- ~~"Replace NVIDIA NIM dependency"~~ — no NVIDIA NIM reference exists
  anywhere in this codebase. Fabricated/hallucinated claim in the source
  doc, not real. Treat any other unverified claim from that doc with the
  same skepticism before acting on it.
- ~~Dhan OAuth flow to avoid 24h token expiry~~ — not fixable this way;
  the 24h expiry is a SEBI regulation, not a Dhan design choice. No
  broker can legally offer a long-lived token here. Real mitigation:
  automate the refresh ceremony if Dhan supports headless re-auth, or
  lean harder on the existing Dhan→tvDatafeed→yfinance fallback chain so
  a stale token degrades gracefully instead of blocking the whole scan.
- ~~Migrate backtest engine to vectorbt~~ — not rejected outright, but
  don't swap the engine that produced the only 2 validated patterns
  without first proving it reproduces identical numbers on Cup & Handle /
  Swing High Breakout. Cross-validate before replacing.
- ~~Optuna/hyperopt weight sweeps~~ — deliberately deferred until the
  walk-forward/OOS discipline above is actually in place. Automated
  parameter search without that discipline first is a way to overfit
  faster, which undermines the exact concern the source doc raised in
  its own methodology section.
- ~~Migrate GitHub Actions to Prefect/Dagster, add a production execution
  engine (nautilus_trader)~~ — solves problems this project doesn't have
  yet (no live automated order execution — orders are placed manually
  from email signals) at real switching cost, right after this session
  spent real effort making GitHub Actions actually reliable. Not urgent
  at current scale.
- HMM regime detection, `mlfinlab` meta-labeling, `riskfolio-lib`,
  QuantStats tear-sheets — plausible future research, not prioritized.

## Done (2026-08-11)

- **`confirm.yml` artifact-lookup fix** — a `daily_scan.yml` run could
  complete all real work (scan + artifact upload) and still be marked
  "cancelled" overall by GH Actions infra seconds after finishing (seen
  2026-08-10: dispatch run 31395814390). The old query only trusted
  `status=success`, skipped that run, and both morning checkpoints failed
  with "Artifact not found." Now scans the last 10 completed runs
  (success OR cancelled) and trusts the first with a real, unexpired
  `picks_latest` artifact.
- **15:15 BTST (Buy Today Sell Tomorrow) scan** — new `confirm_picks.py
  --btst` mode. Evaluates only tickers CONFIRMED at an earlier checkpoint
  today (pulled from `confirm_state_<date>.json`) against two gates:
  closing strength (within 2% of today's high, full-day RVOL ≥ 1.5x) and
  R:R favorability (<70% of the entry→T1 move already captured). New
  `get_intraday_high_low()`, `classify_btst()`, `build_btst_html()`.
  `confirm.yml` takes a new `btst` workflow_dispatch input. Cron-job.org
  job added for 09:45 UTC (15:15 IST) Mon-Fri. **Not yet backtested — see
  P0 list above.**
- **Optional CC recipient for all outbound emails** — new
  `recipients_cc.txt` (gitignored, since the repo is public and this
  differs from the already-tracked `recipients.txt`), read by both
  `emailer.py` and `confirm_picks.py`. Workflows write it at runtime from
  a new `CC_RECIPIENTS` repo secret so the address never lives in the
  public repo itself.

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
