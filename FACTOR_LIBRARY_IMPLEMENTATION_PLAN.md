# Factor Library Implementation Plan

Source: `NSE_Scanner_Factor_Library_5.xlsx` (Downloads) + 8 primary-source PDFs
supplied and reviewed in full (2026-08-12). Plan approved 2026-08-12;
Tiers 1–3 (items 1–5), item 6, and the full backlog (concentrated sizing,
Lottery Index, ML momentum classifier) have all now been built/tested —
see "Validation results so far" below.

A second-round backlog (5 items sourced from public GitHub repos, not the
original Excel/8-paper set) opened 2026-08-14 and closed 2026-08-15: all
5 tested, 2 live (slope × R², volume dry-up), 2 shelved (Frog-in-the-Pan,
RRG sector rotation), 1 tested-but-not-wired as redundant (skewness
composite) — see the table below.

## Validation results so far (updated 2026-08-14)

| Item | Status | Result |
|---|---|---|
| 1. Volatility-adjusted momentum | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | p=0.0004, top tercile Avg R 1.11 vs pool 0.84, consistent across both halves of a 2007–2026, 2,500-signal replay (`validation/tier1_factor_validate.py`) |
| 2. Randomness/Run-Ratio pre-filter | **SHELVED** — failed validation | Inverted: "fails randomness test" bucket (Avg R 0.87) beat "passes" bucket (Avg R 0.64, p=0.92 vs pool); pass-bucket edge degraded further out-of-sample (0.87→0.41) |
| 3. 52-Week High proximity | **SHELVED** — failed validation | Inverted: bottom tercile (further from 52wk high) significantly beat top tercile — p=0.001 vs p=0.9993, Avg R 1.09 vs 0.60 (`validation/fifty_two_wk_high_validate.py`) |
| 4. Promoter feedback-trading (Shu 2009, adapted) | **LIVE (partial)** — wired into `agents/institutional_proxy_agent.py::_10f_promoter_momentum` | Only the `buying_winner` quadrant (promoter buying + RS percentile > 50) validated: N=130, WR=63.8%, Avg R 1.54, p≈0.0 (`validation/promoter_feedback_validate.py`, 1,021-signal replay). The other 3 quadrants were not significant and score neutral (0), never a penalty |
| 5. Insider silence/traded (Ma 2013) | **SHELVED** — failed validation, inverted | `silent` beat `traded` at both the ~1-month (p=0.03) and ~1-year (p=0.0015) horizons — opposite of Ma's thesis; `traded` bucket itself was indistinguishable from the pool (p≈0.97–1.0) at both horizons (`validation/insider_silence_validate.py`, 2,500-signal replay) |
| 6. Badrinath & Wahal entry/exit refinement | **SHELVED** — shelved on primary source, not built | Primary source (Badrinath & Wahal 2002, J. Finance 57(6)) found: institutions momentum-trade on entry, contrarian-trade on exit/adjustments, but the effect cancels out at the aggregate firm level. NSE's `shareholding_history` only has aggregate promoter%/public% — no per-institution entry/exit data exists to test the level where the paper found a real effect |
| Backlog: Lottery Index (Chung 2019) | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | Top tercile N=825, WR=65.9%, Avg R 1.39 vs pool 0.84, p=0.0; bottom tercile reliably underperformed the pool, p=1.0 (`validation/lottery_index_validate.py`, 2,500-signal replay, 2007–2026). Clean monotonic gradient matching Chung's own directional finding — unlike items 2/3/5, this one didn't invert |
| Backlog: ML momentum classifier (Beaudan & He 2019) | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | Predicted-positive class N=868, WR=61.8%, Avg R 1.05 vs pool 0.78, p=0.0; predicted-negative reliably underperformed, p=1.0 (`validation/ml_momentum_validate.py`, 8 independent walk-forward windows, 1,681 out-of-sample predictions, 2016–2026). First trained-model item in this codebase — see `train_ml_momentum_model.py` for the persistence/retraining mechanism |

### Second-round backlog (public-repo sourced, opened 2026-08-14)

| Item | Status | Result |
|---|---|---|
| Frog-in-the-Pan / Information Discreteness (Da, Gurun & Warachka 2014) | **SHELVED** — no signal | Flat null, not an inversion: top tercile (smoothest trend) N=811 Avg R 0.88 p=0.33, bottom tercile (choppiest) N=918 Avg R 0.85 p=0.45 — all three terciles statistically indistinguishable from the pool (0.79–0.88 Avg R band) (`validation/fip_validate.py`, 2,500-signal replay, 2007–2026). Likely explanation: path-smoothness information may already be "used up" by the pattern gate itself, since a detected VCP/Cup&Handle breakout is inherently a specific kind of price path |
| Skewness-penalized composite | **TESTED, NOT WIRED** — redundant with Lottery Index | Top tercile (most positive skew) N=825, WR=60.6%, Avg R 0.98, p=0.042 — signal is real but points OPPOSITE the item's own "penalize high skew" premise, and same direction as (and much weaker than) the already-live Lottery Index, which blends this same idiosyncratic-skew component with MAX/IVOL and found p=0.0, Avg R 1.39 (`validation/skew_penalty_validate.py`, 2,500-signal replay). Wiring both in would double-count the same underlying phenomenon under two modifiers |
| Exponential regression slope × R² (Clenow-style) | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | Strongest result of the whole project: top tercile N=825, WR=69.1%, Avg R 1.54 vs pool 0.84, PF=3.43, p=0.0; bottom tercile reliably below pool, p=1.0; holds up across both halves (1.87→1.21, both well above pool) (`validation/slope_r2_validate.py`, 2,500-signal replay, 2007–2026). Genuinely different momentum formula (trendline slope × fit quality) vs. RS percentile's multi-window relative-return composite, so not redundant with anything already live |
| RRG sector rotation | **SHELVED** — no signal | Flat null, and mildly inconsistent with the theory's own ordering: LEADING N=1073 Avg R 0.89 p=0.25 (not significant), but WEAKENING (the supposedly weaker of the two "outperforming" quadrants) scored HIGHER at Avg R 0.94; LAGGING N=381 Avg R 0.57 p=0.98 (directionally weakest, as expected, but not significant) (`validation/rrg_sector_validate.py`, 2,500-signal replay, 2007–2026, each signal attributed to its stock's sector's RRG quadrant via a synthetic equal-weighted sector index — an open-source approximation of the proprietary JdK/StockCharts formula, never independently validated) |
| Volume dry-up pre-breakout screen (PKScreener-style) | **LIVE** — wired into `agents/rs_agent.py::RSAgent.score()` | Driest tercile (thesis bucket) N=825, WR=64.0%, Avg R 1.20 vs pool 0.84, PF=2.65, p=0.0; wettest tercile below pool (0.75), consistent across both halves (1.49→0.91, both above pool) (`validation/volume_dryup_validate.py`, 2,500-signal replay, 2007–2026). Formula reuses this codebase's own 20-day RVOL baseline convention, not a new window |
| Volume dry-up pre-breakout screen (PKScreener-style) | **VALIDATED — pending wire-in** (batching with item 3) | Driest tercile (thesis bucket) N=825, WR=64.0%, Avg R 1.20 vs pool 0.84, PF=2.65, p=0.0; wettest tercile below pool (0.75), consistent across both halves (1.49→0.91, both above pool) (`validation/volume_dryup_validate.py`, 2,500-signal replay, 2007–2026). Formula reuses this codebase's own 20-day RVOL baseline convention, not a new window |

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

These two both needed a data store this repo didn't previously have: multi-
quarter shareholding *and* insider-disclosure history (`_10d_shareholding`
in `institutional_proxy_agent.py` previously only compared the latest
snapshot to the prior one — no historical store). Built once
(`load_shareholding_insider_history.py`, one-time bulk pull from NSE's
`corporate-share-holdings-master` and `corporates-pit` endpoints — both
return full history per call, no rolling-fetch infrastructure needed) and
fed both agents below from the resulting `shareholding_history` /
`insider_transactions` tables.

### 4. Shu (2009) MT-measure — ✅ LIVE, partial (2026-08-13)
**Formula:** MT = institutional-ownership delta over trailing 8 quarters,
weighted by the stock's past-return decile rank — i.e., are institutions
piling into a stock *because* it's already a momentum winner (positive
feedback), which historically precedes sharper reversals than
fundamentals-driven institutional buying.
**Reframed:** NSE has no FII-ownership field in the shareholding-master
endpoint (only promoter/public %), so this was adapted to **promoter**
feedback-trading instead of institutional — see `agents/promoter_feedback_agent.py`'s
docstring for the explicit caveat that promoters (information edge) are an
economically different actor than Shu's original institutions (no edge), so
no assumed direction was carried over from the paper.
**Built:** `agents/promoter_feedback_agent.py` (MT-measure calc,
`.passes_gate()`/`.score_bonus()`), wired via new
`agents/institutional_proxy_agent.py::_10f_promoter_momentum()`.
**Validated:** `validation/promoter_feedback_validate.py`, 1,021-signal
replay, bucketed by MT tercile and a 2x2 quadrant (buying/selling x
winner/laggard). Only the `buying_winner` quadrant (promoter buying into an
already-strong RS position) was significant — N=130, WR=63.8%, Avg R 1.54,
PF=3.35, **p≈0.0** — the single strongest result across the whole factor
library, and it *inverts* Shu's original reversal thesis (expected, given
the promoter/institution economic-actor substitution). The other three
quadrants (buying_laggard, selling_winner, selling_laggard) were not
statistically distinguishable from the pool (p=0.92–0.98).
**Wired conservatively:** `_10f_promoter_momentum()` scores a bonus (+1 to
+3) **only** in the validated `buying_winner` quadrant; every other
quadrant scores neutral (0), never a penalty — the untested three-quarters
of the hypothesis space gets no weight at all.
**Caveat:** the top-tercile edge decays across the date-range split (Avg R
1.22 first half → 0.47 second half), unlike item 1's improving split — worth
re-checking after more data accumulates.

### 5. Ma (2013) insider silence/traded flag — ❌ SHELVED (2026-08-13, failed validation, inverted)
**Formula:** `NID = (insider shares bought − insider shares sold over
trailing 6mo) / shares outstanding`. **Silence** = zero insider trading
activity in trailing 6mo (proxy for suppressed *negative* information —
insiders avoid selling on bad news due to litigation risk, so absence of
trading, not selling, is the tell). **Traded** = any insider activity
(buy or sell) in trailing 6mo. Ma found: among momentum winners, "traded"
winners kept earning positive returns; "silence" winners reversed hard in
year 2+ (-8.1% vs +11.1%, spread significant at t=-4.28). Same asymmetry
among losers.
**Built:** `agents/insider_silence_agent.py` (traded/silent classification
over a trailing-180-day window, `.passes_gate()`/`.score_bonus()`), fed
from the new `insider_transactions` table (NSE `corporates-pit`, PIT/SAST
Regulation 7(2) disclosures).
**Validated:** `validation/insider_silence_validate.py`, 2,500-signal
replay, testing both a ~1-month horizon (comparable to items 1–4) and a
~1-year horizon (closer to Ma's own "year 2+" finding, per the honest
horizon-mismatch note in the agent's docstring). Result inverted Ma's
thesis at both horizons: `silent` significantly *beat* the pool (20d:
WR=60.8%, p=0.03; 250d: WR=92.3%, p=0.0015), while `traded` was
statistically indistinguishable from the pool at both horizons (p=0.97,
p=0.998). A 20-ticker smoke test had briefly suggested the opposite
(p=0.0016 favoring `traded`) but did not replicate at the full 500-ticker
scale — noise from a small sample.
**Likely reason:** NSE's PIT/SAST disclosures are dominated by routine
ESOP exercises, family/trust transfers, and pledge-related filings rather
than the discretionary open-market buy/sell decisions Ma's US sample was
built on, so "traded" here isn't a clean proxy for "insider chose to act."
**Not wired.** No economic mechanism justifies inverting the score off the
one significant (but paper-contradicting) result; per the standing
discipline, a bare correlation without a mechanism doesn't get live weight.

### 6. Badrinath & Wahal entry/exit event refinement — ❌ SHELVED (2026-08-13, shelved on primary source, not built)
**Source:** Badrinath, S.G. & Wahal, S. (2002), "Momentum Trading by
Institutions," *Journal of Finance* 57(6), pp. 2449–2478. Tracked down
in full this round — was previously only referenced secondhand in the
summary sheet, not one of the 9 originally-reviewed PDFs.
**What the paper actually found:** using per-institution 13F-style
holdings data (~1,200 institutions, 1987–1995), they decomposed
institutional trading into entry (new positions), exit (closed
positions), and adjustments to ongoing holdings. Individual institutions
momentum-trade on entry (buy after positive returns) and contrarian-trade
on exit/adjustments. Critically: **this pattern does not survive
aggregation to the firm level** — momentum trading by some institutions
cancels out contrarian trading by others, so net institutional ownership
at the stock level shows no momentum effect.
**Why this is shelved without a prototype:** `shareholding_history` (the
store item 4 already built) only has *aggregate* promoter%/public% per
quarter — no per-institution entry/exit records exist in NSE's data (NSE
doesn't track FII/DII by name the way US 13F filings do). The paper's
positive result requires exactly the level of granularity (per-institution)
this codebase structurally cannot observe; the level we *can* observe
(aggregate holding change) is the one the paper's own result says nets to
zero. Unlike item 4's FII→promoter substitution (a different actor, still
testable), there's no substitute data here that tests the paper's actual
claim — building an aggregate-level proxy would just be re-testing item 4's
already-validated MT-measure under a different name, not this paper's
finding. Not built; no validation run needed to reach this conclusion.

---

## Backlog — all three items now tested (was "deferred, with reasons")

| Item | Source | Outcome |
|---|---|---|
| Concentrated sizing (`MAX_POSITIONS` → 15) | Raju (2023) | **TESTED 2026-08-13 — NO CHANGE.** Built a new day-by-day portfolio equity-curve simulator (`validation/portfolio_sizing_backtest.py`, didn't exist before — see its docstring for the full method/caveats) since this couldn't be evaluated as a pure config tune after all. Ran MAX_POSITIONS ∈ {6, 8, 10, 15} against a 2016–2026, 501-ticker, 1,699-signal replay: CAGR 14.4% / 20.6% / **22.0%** / 19.5%, Sharpe(approx) 0.52 / **0.65** / 0.55 / 0.43, max drawdown -26.6% / -25.2% / -27.8% / **-20.3%**. Non-monotonic — 10 (current live setting) already has the best raw CAGR, 8 has the best risk-adjusted (Sharpe) number, both 6 and 15 underperform the middle. No config change justified either direction |
| Lottery Index (retail-trading amplification proxy) | Chung (2019) | **BUILT & LIVE 2026-08-14.** Formula confirmed against the primary source: average of cross-sectional z-scores of max daily return, idiosyncratic volatility, and idiosyncratic skewness (`agents/lottery_index_agent.py`). Validated cleanly — top tercile p=0.0, Avg R 1.39 vs pool 0.84; bottom tercile reliably underperformed, p=1.0 (`validation/lottery_index_validate.py`, 2,500-signal replay). Wired into `RSAgent.score()`: +2 at ≥67th percentile, -1 at ≤33rd, matching only what was actually tercile-tested (no finer graded scale) |
| ML logistic-regression momentum model | Beaudan & He (2019) | **BUILT & LIVE 2026-08-14.** Full paper obtained and reviewed. Paper times ONE asset (SPX, invest-vs-cash); adapted to a per-stock classifier applied to gate-cleared candidates (pooled cross-sectional training, not the paper's single-asset design) per explicit scope decision. Cubic-polynomial logistic regression (455 features from 12 momentum/drawdown seeds, matching the paper's own feature count exactly), walk-forward retrained every ~2 years. 8 independent out-of-sample windows: predicted-positive p=0.0, Avg R 1.05 vs pool 0.78; predicted-negative reliably underperformed, p=1.0. First TRAINED-MODEL item in this codebase — needs periodic retraining via `train_ml_momentum_model.py`, unlike every closed-form item above |

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
3. ~~Items 4 + 5~~ — **done 2026-08-13.** Built the shared shareholding/
   insider-disclosure history store (`load_shareholding_insider_history.py`,
   one-time bulk pull), then both agents. Item 4 (reframed as promoter
   feedback-trading) cleared — partially, only its `buying_winner`
   quadrant — and is live in
   `agents/institutional_proxy_agent.py::_10f_promoter_momentum`. Item 5
   inverted Ma's thesis at both a ~1-month and ~1-year horizon and is
   shelved. As with items 2/3, a strong source paper did not guarantee a
   smooth (or even directionally correct) validation pass.
4. ~~Item 6~~ — **done 2026-08-13, shelved without a build.** Tracked down
   the primary source (Badrinath & Wahal 2002); its positive result is at
   a per-institution granularity NSE's data doesn't expose, and the
   aggregate-level signal we could compute is exactly what the paper
   itself found nets to zero. No prototype/validation needed to reach
   that conclusion.
5. ~~Backlog items~~ — **done 2026-08-14.** Concentrated sizing tested, no
   config change justified. Lottery Index and the ML momentum classifier
   both validated cleanly and are live in `RSAgent.score()`. Every item in
   this plan (1–6 plus the full backlog) has now been either shipped or
   shelved with a documented reason — nothing left unbuilt.

---

## What this plan is explicitly not

- Not a commitment to build all of this — each item still has to clear its
  own significance test independently; a Tier 1 item failing
  `monte_carlo_significance.py` gets pruned the same way VCP/Bull Flag were.
- Not a schedule/timeline — no dates attached, sequencing only.
- Not code — nothing above has been implemented. Awaiting sign-off on scope
  and sequencing before any prototype agent gets written.
