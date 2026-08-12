# Factor Library Implementation Plan

Source: `NSE_Scanner_Factor_Library_5.xlsx` (Downloads) + 8 primary-source PDFs
supplied and reviewed in full (2026-08-12). Plan approved 2026-08-12;
Tier 1 and Tier 2 have since been built, validated, and the results are in
— see "Validation results so far" below. Tiers 3+ and the backlog remain
unbuilt, awaiting the same process.

## Validation results so far (updated 2026-08-12)

| Item | Status | Result |
|---|---|---|
| 1. Volatility-adjusted momentum | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | p=0.0004, top tercile Avg R 1.11 vs pool 0.84, consistent across both halves of a 2007–2026, 2,500-signal replay (`validation/tier1_factor_validate.py`) |
| 2. Randomness/Run-Ratio pre-filter | **SHELVED** — failed validation | Inverted: "fails randomness test" bucket (Avg R 0.87) beat "passes" bucket (Avg R 0.64, p=0.92 vs pool); pass-bucket edge degraded further out-of-sample (0.87→0.41) |
| 3. 52-Week High proximity | **SHELVED** — failed validation | Inverted: bottom tercile (further from 52wk high) significantly beat top tercile — p=0.001 vs p=0.9993, Avg R 1.09 vs 0.60 (`validation/fifty_two_wk_high_validate.py`) |
| 4–6, backlog | Not yet built | See sections below |

**Why items 2 and 3 both inverted — a methodological note for items 4+:**
Both failures point the same direction: the validation population is
signals that already cleared `PatternAgent`'s breakout-pattern detection,
not a raw cross-sectional universe rank the way Raju/Desai's source papers
tested. Within an already-curated "this looks like a breakout" population,
"further below the 52-week high" plausibly means "earlier-stage breakout
with more room to run" rather than "weaker anchoring," and "fails a
randomness test" plausibly captures short-term consolidation/pullback
structure that a pattern detector already selects for, not "fake trend."
The lesson isn't "these two papers are wrong" — it's that **any factor
from the literature must be treated as unproven in this codebase's specific
gate-cleared context, regardless of its academic pedigree, until it clears
validation here.** Item 1 (Shenoy & Vijaykumar) cleared cleanly; items 2/3
didn't. No a priori reason to expect items 4/5 to land either way — same
validation-before-wiring discipline applies. This plan is the deliverable
the review phase was gated on. Nothing in it should be implemented without
running it through this same chain first.

Papers reviewed: Raju (2023) 52-Week High Effect · Shu (2009) Positive-Feedback
Trading · Desai (2014) Quantifying Randomness · Anonymous (2024) Momentum vs.
Fundamentals · Raju (2023) Number of Holdings & Universe Size · Chung (2019)
Retail Trading & Momentum · Beaudan & He (2019) ML for Trading Strategies ·
Shenoy & Vijaykumar (2020) Momentum Investing in Indian Equities · Ma (2013)
Momentum and Insider Trading.

---

## How to read this plan

Each item states: the exact formula from the source paper, what already
exists in this codebase that overlaps or doesn't, the specific file(s) it
touches, and the validation gate it must clear before `orchestrator.py` ever
sees it. This mirrors the existing discipline in this repo — see
`agents/weekly_trend_agent.py` (unwired prototype, `.passes_gate()` +
`.score_bonus()` dual interface) and `pattern_agent.py`'s `DEFAULT_WEIGHTS`
changelog (VCP/Bull Flag pruned after failing significance tests). Every item
below follows the same chain:

```
prototype agent (unwired) → pipeline_replay.py backtest
  → monte_carlo_significance.py (bootstrap/permutation p-value)
  → split_period_significance.py (out-of-sample check)
  → only then: wire into orchestrator.py
```

Nothing gets live weight without clearing that chain, regardless of how
compelling the source paper's backtest looks.

---

## Tier 1 — build first (near-zero cost, reuses existing code/data)

### 1. Volatility-adjusted momentum ranking — ✅ LIVE (2026-08-12)
**Source:** Shenoy & Vijaykumar (2020), Capitalmind.
**Formula:** `VolAdjMom = (12-month absolute return) / (annualized stddev of daily returns)`,
top-30 equal-weighted, monthly rebalance.
**Their result (NSE, 2000–2020):** Nifty 9.2% CAGR/0.07 Sharpe; naive momentum
15.9%/0.20; vol-adjusted momentum 22.1%/0.38 — nearly 2x the risk-adjusted
return of naive momentum, with a *smaller* max drawdown (-78% vs -82%).
**What exists:** `agents/rs_agent.py` already computes 4w/12w/26w weighted
return percentile (40/40/20) and sector-relative ranks. It ranks on raw
return only — nothing currently divides by realized volatility.
**Build:** Add a volatility-adjusted variant of the existing return calc in
`rs_agent.py` — reuse the same lookback windows, add an annualized-stddev
denominator, expose as a second ranking (not a replacement) so the
validation run can compare raw-RS vs. vol-adjusted-RS on identical history.
**Effort:** Low. No new data source — same OHLCV already loaded.

### 2. Randomness / Run-Ratio pre-filter — ❌ SHELVED (2026-08-12, failed validation)
**Source:** Desai (2014).
**Formula:** Run Ratio = actual number of price-direction runs over N days /
expected number of runs under a random walk (Wald–Wolfowitz runs test).
Z-scored to flag stocks whose trend is statistically distinguishable from
noise vs. those where the "trend" is a random walk artifact.
**What exists:** Nothing — no run-length or randomness test anywhere in the
gate chain (G1–G7) or scoring agents today.
**Build:** New pure-OHLCV utility (no new data source), used as a
pre-screen ahead of pattern/momentum scoring — filters out candidates whose
apparent trend has no statistical basis before they ever reach the scoring
agents, rather than adding another additive score component.
**Effort:** Trivial. Standalone module, easiest possible validation run
since it only needs close-price history already in memory.
**Result:** Inverted — the "fails randomness test" bucket outperformed the
"passes" bucket (Avg R 0.87 vs 0.64, p=0.92 vs pool for the pass bucket),
and the pass bucket's own edge degraded further out-of-sample (0.87→0.41
first half to second half). Not wired. See the methodological note at the
top of this document.

---

## Tier 2 — build next (new agent, still same data source)

### 3. 52-Week High proximity agent, cap-tier weighted — ❌ SHELVED (2026-08-12, failed validation)
**Source:** Raju (2023) 52-Week High Effect + Anonymous (2024) mid/small-cap
segmentation.
**Formula:** Rank by `Price / 52-week high` (George & Hwang anchoring
effect — proximity to the 52-week high, not raw return, is the driver).
Anonymous (2024) adds: the effect is stronger in mid/small-cap segments than
large-cap, where fundamentals dominate more.
**What exists:** `agents/near_breakout.py` computes gap-to-a-*pattern*
breakout level (from `PatternAgent`), which is a different number from the
literal 52-week high — it's breakout-proximity, not anchoring-proximity.
There is no dedicated 52wk-high ratio agent today.
**Build:** New `agents/fifty_two_wk_high_agent.py`. Needs rolling 52-week
high from the same daily OHLCV already fetched (no new data source) plus
`universe` (available on every scanned item, per `near_breakout.py`'s
existing `universe_items` tuple shape) to apply Anonymous (2024)'s cap-tier
weighting — larger bonus for mid/small-cap, smaller for large-cap.
**Effort:** Low. Highest standalone alpha signal of the whole library per
Raju's own backtest, and no new data plumbing required.
**Result:** Inverted — the bottom tercile (further from the 52-week high)
significantly beat the top tercile (Avg R 1.09 vs 0.60, p=0.001 vs
p=0.9993), consistent across both halves of the date range. Cap-tier
breakdown did show small > mid > large within the (underperforming) top
tercile, directionally consistent with Anonymous (2024)'s concentration
claim, but on the wrong side of the effect to be useful. Not wired. See
the methodological note at the top of this document.

---

## Tier 3 — build together (share the same missing infrastructure)

These two both need a data store this repo doesn't currently have: multi-
quarter shareholding *and* insider-disclosure history (today's
`_10d_shareholding` in `institutional_proxy_agent.py` only compares the
latest snapshot to the prior one — no historical store). Building that store
once and feeding both agents from it avoids building the same plumbing twice.

### 4. Shu (2009) MT-measure exit-timing overlay
**Formula:** MT = institutional-ownership delta over trailing 8 quarters,
weighted by the stock's past-return decile rank — i.e., are institutions
piling into a stock *because* it's already a momentum winner (positive
feedback), which historically precedes sharper reversals than
fundamentals-driven institutional buying.
**What exists:** `institutional_proxy_agent.py::_10d_shareholding` (NSE
`corporates-shp` API) does latest-vs-prior only — the gap vs. Shu's 8-quarter
requirement.
**Build:** Extend `_10d_shareholding` to store and read an 8-quarter
rolling history (new persistent store, e.g. a table alongside the existing
`momentum_v4.db` schema); compute the MT measure; feed as a time-stop
trigger to `portfolio_engine.py` (exit sooner on positions whose
institutional buying looks like late-stage positive feedback, not fresh
conviction).
**Effort:** Medium — the new quarterly history store is the real cost, the
MT formula itself is simple once the data exists.

### 5. Ma (2013) insider silence/traded flag
**Formula:** `NID = (insider shares bought − insider shares sold over
trailing 6mo) / shares outstanding`. **Silence** = zero insider trading
activity in trailing 6mo (proxy for suppressed *negative* information —
insiders avoid selling on bad news due to litigation risk, so absence of
trading, not selling, is the tell). **Traded** = any insider activity
(buy or sell) in trailing 6mo. Among momentum winners: "traded" winners kept
earning positive returns; "silence" winners reversed hard in year 2+
(-8.1% vs +11.1%, spread significant at t=-4.28). Same asymmetry among
losers.
**What exists:** Nothing equivalent — `_10e_bulk_deals` tracks *any* large
investor's bulk/block deals (a different NSE endpoint and a different
population), not promoter/director insider trades under SAST/PIT
disclosure rules.
**Build:** New data source — NSE's insider-disclosure endpoint (PIT/SAST
filings, distinct from `bulk-deals`), 6-month trailing window. Use as a
**filter/red-flag on momentum longs**, not a scoring bonus: require insider
activity (buy or sell — direction doesn't matter per Ma's finding) in the
trailing 6 months as a condition of remaining a "winner" candidate; flag
silence winners as an elevated-reversal-risk warning.
**Effort:** Medium — new external data source is the main cost; the
silence/traded classification itself is a simple boolean once the feed
exists.

### 6. Badrinath & Wahal entry/exit event refinement
**Source:** referenced in the summary sheet alongside Shu; not one of the
9 fully-reviewed PDFs, so this stays lower-confidence pending the primary
source if you want it pursued.
**Build (as currently understood):** Refine `_10d_shareholding` to
distinguish institutional *entry* events (first appearance in the
shareholding table) from *exit* events (dropping off it), rather than
treating any FII% delta uniformly. Naturally falls out of the same
quarterly history store Tier 3 already requires.
**Effort:** Low, once items 4/5's store exists.

---

## Backlog — deferred, with reasons

| Item | Source | Why deferred |
|---|---|---|
| Concentrated sizing (`MAX_POSITIONS` → 15 for a ~200–325-name universe) | Raju (2023) | Pure config tune, not a build — worth revisiting once Tier 1/2 signals are validated and the universe size this applies to is re-confirmed, so it's tuned against the *post-upgrade* scoring, not today's |
| Lottery Index (retail-trading amplification proxy) | Chung (2019) | Formula is in hand (price/vol/skew/max-return decile composite), but it was built as a *no-retail-data* proxy for the US market; NSE has no direct retail-participation feed either, so this is buildable, but needs its own validation that the proxy still holds on NSE microstructure before treating it as anything but speculative |
| ML logistic-regression momentum model | Beaudan & He (2019) | Confirmed as the largest lift in the library — full feature pipeline, regularization, retraining trigger. Reassess only after the simpler Tier 1–3 wins are validated and live; premature to build a model on top of scoring inputs that haven't themselves been proven yet |

---

## Suggested sequencing

1. ~~Items 1 + 2 together~~ — **done 2026-08-12.** Built as unwired
   prototypes, validated via `validation/tier1_factor_validate.py` (a
   purpose-built harness reusing the real gate chain against
   `price_history_deep`, 2007–2026). Item 1 cleared and is now live in
   `agents/rs_agent.py::RSAgent.score()`. Item 2 failed and is shelved.
2. ~~Item 3~~ — **done 2026-08-12.** Built (`agents/fifty_two_wk_high_agent.py`),
   validated via `validation/fifty_two_wk_high_validate.py`. Failed and is
   shelved — see the methodological note at the top of this document.
3. **Items 4 + 5 together** — both gated on building the same missing
   piece (multi-quarter shareholding/insider-disclosure history store).
   Build the store once, then both agents. Given items 2/3 both inverted
   within this codebase's gate-cleared population, budget for a real
   chance either of these needs the same treatment — don't assume a
   strong source paper implies a smooth validation pass.
4. **Item 6** — falls out of item 4/5's infrastructure once it exists.
5. Backlog items — revisit only after 1–6 are live and validated.

---

## What this plan is explicitly not

- Not a commitment to build all of this — each item still has to clear its
  own significance test independently; a Tier 1 item failing
  `monte_carlo_significance.py` gets pruned the same way VCP/Bull Flag were.
- Not a schedule/timeline — no dates attached, sequencing only.
- Not code — nothing above has been implemented. Awaiting sign-off on scope
  and sequencing before any prototype agent gets written.
