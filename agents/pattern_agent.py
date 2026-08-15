"""
NSE Momentum v6.3 — Pattern Agent

CHANGES vs v6.2 (this rebuild, 2026-08-05):
  [PRUNE] VCP moved from DEFAULT_WEIGHTS to PRUNED_PATTERNS (weight 8 -> 0).

    Evidence chain:
      - 2026-08-01 result (25 universe snapshots, basis for v6.2's weights):
        N=50, net avg R=0.68, PF=2.16, permutation p=0.1019 (INCONCLUSIVE —
        CI above zero, but not distinguishable from the pooled gate-chain
        baseline).
      - 2026-08-05 confirmation run (validation/pipeline_replay_deep.py
        --all-patterns --fresh, re-run after universe_snapshots was fixed
        from 25 -> 47 point-in-time snapshots per P4-05; then
        validation/monte_carlo_significance.py --all-patterns against the
        resulting 501/501-ticker, 6,106-trade checkpoint):
        N=70, net avg R=0.50, PF=1.78, permutation p=0.1525.

    Verdict: with 40% more data (N=50 -> 70), VCP's edge got WEAKER
    (avg R 0.68 -> 0.50, PF 2.16 -> 1.78) and LESS significant (p=0.1019
    -> 0.1525), not more. A real, gate-adjusted edge should tighten toward
    significance as N grows, not drift the other way. This is the
    signature of small-sample noise settling out, not a pattern losing
    edge over time (there's no reason to expect VCP's real edge decayed
    in the 4 days between runs — the added trades are historical, not
    new market activity). Treating the 08-01 result as noise rather than
    a decaying-but-real edge, VCP is pruned rather than kept at reduced
    weight.

    Detection logic is UNCHANGED (see [DETECT] below) — VCP still fires
    in self.all_detections for validation re-testing if a future replay
    with more data (e.g. once universe_snapshots extends further, or a
    12yr+ history becomes available) wants to re-examine it. Only its
    live weight changed.

  [CONFIRM] Cup & Handle and Swing High Breakout re-confirmed on the same
    2026-08-05 run, both with materially larger N than the 08-01 basis and
    unchanged significance (p=0.0000 both):
        Pattern                08-01 N   08-05 N   Net AvgR   PF     p
        Cup & Handle             847      1119      1.06     2.73   0.0000
        Swing High Breakout     1033      1456      0.67     1.98   0.0000
    Both patterns' point estimates were essentially unchanged as N grew by
    32-41% — the opposite of what happened to VCP, and the expected
    behaviour of a real, gate-adjusted edge. No weight change.

  [DATA] All other N-counts and p-values throughout this file (pruned
    patterns' code comments) updated to the 2026-08-05 confirmation run's
    figures, replacing the earlier 25-snapshot-basis numbers. No pruned
    pattern's verdict changed — the larger, 47-snapshot sample confirms
    the earlier prune decisions rather than reversing any of them.

  [DETECT] No detection logic changed. All 19 pattern detection blocks
    remain present (see v6.1 note below) — this is a pure weight/metadata
    change, nothing structural.

CHANGES vs v6.1 (retained for history):
  [WEIGHTS REBUILT — EVIDENCE-BASED] DEFAULT_WEIGHTS replaced, first time
  since v5.3, with numbers backed by a real full-pipeline replay instead
  of a disputed/stale backtest. Chain of evidence:

    1. validation/pipeline_replay_deep.py --all-patterns --fresh
       (2026-07-27, 10:17-11:37 AM): tested all 19 detected patterns
       through the REAL live gate chain (Pattern+RS+Liquidity+Risk+
       Asymmetry+VCP gates), 10yr point-in-time NIFTY 500 universe,
       501 tickers, real NSE transaction costs (0.363% round-trip,
       0.35%/leg brokerage confirmed via Yes Bank contract note
       analysis — corrected from a prior 0%-brokerage run same day).
       6,106 gate-cleared signals total.

    2. validation/monte_carlo_significance.py (2026-07-27, same day):
       bootstrap 95% CIs + permutation test (vs. the full 6,106-trade
       pooled pool) on the 5 candidate patterns, to separate real edge
       from sample noise.

       Cup & Handle and Swing High Breakout were unambiguous: large
       samples, both tests agree strongly. VCP's bootstrap CI barely
       cleared zero but the permutation test could not distinguish it
       from a random draw — kept at reduced weight and flagged
       low-conviction at the time (see v6.3 changes above for the
       follow-up that pruned it). Falling Wedge and Bull Flag did NOT
       hold up under the permutation test despite positive point
       estimates — both left pruned rather than promoted.

    3. The 5 previously-live patterns (High Tight Flag, Flat Base,
       Rounded Base, High Base, Volume Expansion) were, per the same
       2026-07-27 replay, at or below breakeven with real costs applied.
       High Tight Flag still too rare to register any gate-cleared
       signal in 10yr of history — untested, not disproven, but
       contributes nothing live either way. All 5 moved to
       PRUNED_PATTERNS.

  [PROMOTE] Cup & Handle (weight 20), Swing High Breakout (weight 16),
            VCP (weight 8, reduced conviction — see verdict above;
            subsequently pruned in v6.3, see top of this docstring).
  [PRUNE]   High Tight Flag, Flat Base, Rounded Base, High Base,
            Volume Expansion — moved from DEFAULT_WEIGHTS to
            PRUNED_PATTERNS. Falling Wedge and Bull Flag remain pruned
            (tested, did not clear significance).
  [DETECT]  No detection logic changed. All 19 pattern detection blocks
            were already present — this is a pure weight/metadata
            change, nothing structural.
  [DATA]    Corrected monte_carlo_significance DB table (was stale,
            dated 2026-07-26 pre-brokerage-fix; now reflects the
            2026-07-27 cost-corrected replay).

CHANGES vs v6.0 (unchanged, retained for history):
  [CORRECTION] The "+0.69% / +0.51% / +0.40% / +0.09% / 0.00% validated
  expectancy" claims previously here (and in PATTERN_EXPECTANCY) traced
  back to a v4.0-era backtest that was shown to be unreliable, for two
  confirmed, distinct reasons:

    1. That backtest had ALL 19 patterns competing for the same
       max(detections, key=weight) selection under 2023-era weights,
       where e.g. Flat Base was weighted 10/19 (near the bottom). Its
       old count (40 signals) measured "how often this pattern WON
       against 18 higher-weighted competitors," not "how often this
       shape occurs." Not a stable baseline to compare weights against.

    2. Independently, validation/backtest.py had a window-cap bug (fixed
       2026-07-25) that capped every test window to 120 bars, making
       `above_200` — and therefore High Base specifically — permanently
       undetectable in that script, regardless of real price action.

  DEFAULT_WEIGHTS was left unchanged at the time pending a full
  historical replay of the actual orchestrator.run() gate chain — that
  replay is what the [WEIGHTS REBUILT] section above reports on.
"""

import logging
import pandas as pd
import numpy as np

try:
    import pandas_ta_classic as ta
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    _PANDAS_TA_AVAILABLE = False

# Backtest-validated weights (Cup & Handle, Swing High Breakout — see
# pipeline_replay_deep.py results) were derived using manual (Cutler-style,
# simple-mean) RSI, not Wilder's smoothing. pandas_ta_classic.rsi() uses
# Wilder's EMA-based smoothing, which diverges from the manual calc by
# 8-18 RSI points on the same ticker/date (confirmed 05 Aug 2026,
# RELIANCE.NS, see rsi_check.py). Do NOT flip this to True without first
# re-running pipeline_replay_deep.py against Wilder's RSI and re-deriving
# pattern weights — see INTEGRATION_BACKLOG1.md P4-10.
_PANDAS_TA = False

log = logging.getLogger(__name__)

# ── Pattern weights (v6.3 — EVIDENCE-BASED, see docstring) ─────────────────
# Rebuilt 2026-07-27, RE-VERIFIED 2026-08-01, RE-CONFIRMED 2026-08-05 from
# validation/pipeline_replay_deep.py --all-patterns --fresh (real gate
# chain, point-in-time universe covering all 47 snapshots as of the 08-05
# run, real NSE costs) + validation/monte_carlo_significance.py --all-patterns
# (bootstrap CI + permutation test vs. the full 6,106-trade pooled pool).
# See module docstring for the full evidence chain and numbers.
#
# VCP was promoted here at reduced weight on 07-27/08-01 evidence, then
# PRUNED on 08-05 once a larger, 47-snapshot sample showed its edge
# weakening (not strengthening) with more data — see docstring [PRUNE]
# section. Only 2 patterns remain live as of v6.3.
DEFAULT_WEIGHTS = {
    "Cup & Handle":        20,   # N=1119, net avg R=1.06, PF=2.73, p=0.0000 — strong, confirmed 05 Aug
    "Swing High Breakout": 16,   # N=1456, net avg R=0.67, PF=1.98, p=0.0000 — strong, confirmed 05 Aug
}

# Pruned patterns — detected but weight=0 so never selected as best pattern.
# High Tight Flag / Flat Base / Rounded Base / High Base / Volume Expansion
# moved here 2026-07-27: all at or below breakeven with real costs applied
# in the full-pipeline replay. Falling Wedge and Bull Flag remain pruned —
# tested via Monte Carlo, did not clear significance. VCP moved here
# 2026-08-05 — see docstring [PRUNE] section for the full evidence chain
# (edge weakened, not strengthened, as sample size grew).
#
# All N-counts / p-values below are from the 2026-08-05 confirmation run
# (validation/monte_carlo_significance.py --all-patterns, 47-snapshot
# universe, 6,106 pooled gate-cleared trades across 17 patterns with any
# signals). This larger sample CONFIRMS every prune decision below — none
# reversed.
PRUNED_PATTERNS = {
    "High Tight Flag":      0,   # untested — 0 gate-cleared signals, 08-05 run same as prior runs
    "IPO Base":              0,   # untested — 0 gate-cleared signals, 08-05 run same as prior runs
    "Flat Base":             0,   # N=219, net avg R=-0.07, p=0.9570 — inconclusive, CI straddles zero
    "Rounded Base":          0,   # N=88,  net avg R=-0.47, p=0.9974 — likely negative
    "High Base":             0,   # N=654, net avg R=-0.46, p=1.0000 — negative, CI entirely below zero
    "Volume Expansion":      0,   # N=45,  net avg R=-0.65, p=0.9921 — negative, CI entirely below zero
    "Falling Wedge":         0,   # N=102, net avg R=0.22,  p=0.4804 — inconclusive, not significant
    "Bull Flag":             0,   # N=400, net avg R=0.16,  p=0.6642 — inconclusive, not significant
    "VCP":                   0,   # N=70,  net avg R=0.50,  p=0.1525 — edge weakened vs 08-01 (N=50, p=0.1019); pruned 08-05
    "Base Breakout":         0,   # N=49,  net avg R=-0.40, p=0.9652 — likely negative
    "52W Momentum":          0,   # N=715, net avg R=-0.48, p=1.0000 — negative, CI entirely below zero
    "Double Bottom":         0,   # N=692, net avg R=-0.14, p=1.0000 — likely negative
    "Ascending Triangle":    0,   # N=65,  net avg R=-0.12, p=0.8699 — inconclusive
    "Symmetrical Triangle":  0,   # N=13,  TOO THIN TO TEST (N<20) — do not draw conclusions
    "Descending Wedge":      0,   # N=65,  net avg R=-0.22, p=0.9368 — inconclusive
    "3-Weeks-Tight":         0,   # N=103, net avg R=-0.35, p=0.9922 — likely negative
    "Diamond Bottom":        0,   # N=251, net avg R=-0.42, p=1.0000 — negative, CI entirely below zero
}

ALL_WEIGHTS = {**DEFAULT_WEIGHTS, **PRUNED_PATTERNS}

# ── Evidence-based expectancy (v6.3) ────────────────────────────────────────
# Net avg R per trade (real NSE costs applied), from the 2026-08-05
# confirmation run (pipeline_replay_deep.py + monte_carlo_significance.py).
# Units are R-multiples (what pipeline_replay_deep.py and
# monte_carlo_significance.py both report in). Only the 2 currently-live
# patterns are listed — VCP removed here on 08-05 prune (see docstring).
PATTERN_EXPECTANCY = {
    "Cup & Handle":        1.06,   # p=0.0000 — statistically solid
    "Swing High Breakout": 0.67,   # p=0.0000 — statistically solid
}
# Threshold retained for any future pattern promoted at reduced conviction
# (the role VCP briefly played 07-27 through 08-05). Both current live
# patterns clear it comfortably (p=0.0000 each), so neither is flagged
# low-edge today.
LOW_EDGE_EXPECTANCY_THRESHOLD = 0.55


def is_low_edge_pattern(pattern_name: str) -> bool:
    """
    True if this pattern's real, evidence-based net expectancy (from the
    2026-08-05 confirmation run) is below the high-conviction threshold.
    As of v6.3, both live patterns (Cup & Handle, Swing High Breakout)
    are p=0.0000 and comfortably above threshold — this currently flags
    nothing. Kept for the next pattern that gets promoted at reduced
    conviction, the role VCP held until its 08-05 prune.
    """
    return PATTERN_EXPECTANCY.get(pattern_name, 1.0) < LOW_EDGE_EXPECTANCY_THRESHOLD

class PatternAgent:
    def __init__(self, df: pd.DataFrame, bars_per_day: float = 1.0):
        self.df               = df.copy()
        self.pattern          = ""
        self.breakout_level   = 0.0
        self.entry_low        = 0.0
        self.entry_high       = 0.0
        self.raw_score        = 0
        self.breakout_quality = "MINOR"
        self.indicators       = {}
        # [NEW] bars_per_day=1 (default) is exactly the original daily
        # behavior — the live scanner never passes this, unaffected. The
        # hourly replay passes bars_per_day=7 (see hourly_scaling.py).
        # SCOPE: only the indicator windows that feed real scoring
        # (EMA 10/21/50, RSI-14, MACD, ATR-14) and the 2 live-weighted
        # patterns' (Cup & Handle, Swing High Breakout) own detection
        # windows are scaled here. EMA-200/52-week-high and all 17
        # pruned patterns' windows are left untouched — they have weight
        # 0, so self.pattern can never be set to one (see the max()/
        # weight>0 gate below), meaning their windows have zero effect on
        # either live scoring or the replay's gate-cleared signal count.
        # Scaling them would be real work with provably no effect on the
        # result.
        self.bars_per_day = bars_per_day
        self._w5   = max(2, round(5   * bars_per_day))
        self._w10  = max(2, round(10  * bars_per_day))
        self._w14  = max(2, round(14  * bars_per_day))
        self._w15  = max(2, round(15  * bars_per_day))
        self._w20  = max(2, round(20  * bars_per_day))
        self._w21  = max(2, round(21  * bars_per_day))
        self._w25  = max(2, round(25  * bars_per_day))
        self._w26  = max(2, round(26  * bars_per_day))
        self._w27  = max(2, round(27  * bars_per_day))
        self._w30  = max(2, round(30  * bars_per_day))
        self._w35  = max(2, round(35  * bars_per_day))
        self._w45  = max(2, round(45  * bars_per_day))
        self._w50  = max(2, round(50  * bars_per_day))
        self._w60  = max(2, round(60  * bars_per_day))
        self._w130 = max(2, round(130 * bars_per_day))
        # v6.1: every pattern detected this bar, weighted or pruned —
        # [(pattern_name, breakout_level), ...]. self.pattern (below) still
        # picks a single "best" one for live conviction scoring, unchanged.
        # This list exists so validation/pipeline_replay_deep.py can test
        # ALL 19 patterns independently through the real gate chain,
        # instead of only the 2 that can currently win pattern selection.
        self.all_detections   = []
        self._compute_indicators()
        self._detect()

    # ── Indicator computation ───────────────────────────────────────────────

    def _compute_indicators(self):
        df    = self.df
        n     = len(df)
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        ind   = {}

        if _PANDAS_TA and n >= self._w26:
            try:
                ind["ema10"]  = float(ta.ema(close, length=self._w10).iloc[-1])
                ind["ema21"]  = float(ta.ema(close, length=self._w21).iloc[-1])
                ind["ema50"]  = float(ta.ema(close, length=self._w50).iloc[-1]) if n >= self._w50 else 0.0
                ind["ema200"] = float(ta.ema(close, length=200).iloc[-1]) if n >= 200 else 0.0

                rsi_s = ta.rsi(close, length=self._w14)
                ind["rsi14"] = float(rsi_s.iloc[-1]) if rsi_s is not None else 50.0

                macd_df = ta.macd(close, fast=round(12*self.bars_per_day), slow=self._w26, signal=round(9*self.bars_per_day))
                if macd_df is not None and not macd_df.empty:
                    cols = macd_df.columns.tolist()
                    ind["macd"]           = float(macd_df[cols[0]].iloc[-1])
                    ind["macd_signal"]    = float(macd_df[cols[1]].iloc[-1])
                    ind["macd_hist"]      = float(macd_df[cols[2]].iloc[-1])
                    ind["macd_hist_prev"] = float(macd_df[cols[2]].iloc[-2]) if n >= self._w27 else 0.0
                else:
                    ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = ind["macd_hist_prev"] = 0.0

                if n >= self._w15:
                    atr_s = ta.atr(high, low, close, length=self._w14)
                    ind["atr14"] = float(atr_s.iloc[-1]) if atr_s is not None else 0.0
                else:
                    ind["atr14"] = 0.0

            except Exception as e:
                log.debug(f"pandas-ta failed, using fallback: {e}")
                self._compute_indicators_fallback(ind)
        else:
            self._compute_indicators_fallback(ind)

        self.indicators = ind

    def _compute_indicators_fallback(self, ind: dict):
        df    = self.df
        n     = len(df)
        close = df["Close"].squeeze().to_numpy(dtype=float)
        high  = df["High"].squeeze().to_numpy(dtype=float)
        low   = df["Low"].squeeze().to_numpy(dtype=float)

        ind["ema10"]  = self._ema(close, self._w10)[-1]  if n >= self._w10  else 0.0
        ind["ema21"]  = self._ema(close, self._w21)[-1]  if n >= self._w21  else 0.0
        ind["ema50"]  = self._ema(close, self._w50)[-1]  if n >= self._w50  else 0.0
        ind["ema200"] = self._ema(close, 200)[-1] if n >= 200 else 0.0

        if n >= self._w15:
            delta  = np.diff(close[-self._w15:])
            gain   = np.where(delta > 0, delta, 0)
            loss   = np.where(delta < 0, -delta, 0)
            avg_g  = np.mean(gain) if gain.any() else 1e-9
            avg_l  = np.mean(loss) if loss.any() else 1e-9
            rs     = avg_g / avg_l if avg_l > 0 else 99
            ind["rsi14"] = 100 - 100 / (1 + rs)
        else:
            ind["rsi14"] = 50.0

        if n >= self._w26:
            ema12 = self._ema(close, round(12*self.bars_per_day))
            ema26 = self._ema(close, self._w26)
            macd  = ema12 - ema26
            sig   = self._ema(macd, round(9*self.bars_per_day))
            hist  = macd - sig
            ind["macd"]           = macd[-1]
            ind["macd_signal"]    = sig[-1]
            ind["macd_hist"]      = hist[-1]
            ind["macd_hist_prev"] = hist[-2] if n >= self._w27 else 0.0
        else:
            ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = ind["macd_hist_prev"] = 0.0

        if n >= self._w15:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:]  - close[:-1]))
            )
            ind["atr14"] = float(np.mean(tr[-self._w14:]))
        else:
            ind["atr14"] = 0.0

    # ── Public helpers ───────────────────────────────────────────────────────

    def get_atr_pct(self) -> float:
        price = float(self.df["Close"].squeeze().iloc[-1])
        atr   = self.indicators.get("atr14", 0.0)
        return round(atr / price * 100, 2) if price > 0 else 0.0

    def score(self) -> int:
        return self.raw_score

    def get_ema_score(self) -> int:
        if len(self.df) < self._w50:
            return 0
        ind   = self.indicators
        price = float(self.df["Close"].squeeze().iloc[-1])
        pts   = 0
        if price > ind.get("ema10",  0): pts += 2
        if price > ind.get("ema21",  0): pts += 2
        if price > ind.get("ema50",  0): pts += 2
        if (ind.get("ema10", 0) > ind.get("ema21", 0) > ind.get("ema50", 0)): pts += 2
        return min(pts, 8)

    def get_macd_score(self) -> int:
        if len(self.df) < self._w35:
            return 0
        ind = self.indicators
        pts = 0
        if ind.get("macd", 0) > 0:                                  pts += 1
        if ind.get("macd", 0) > ind.get("macd_signal", 0):          pts += 2
        if ind.get("macd_hist", 0) > ind.get("macd_hist_prev", 0):  pts += 1
        return min(pts, 4)

    def get_rsi_score(self) -> int:
        if len(self.df) < self._w15:
            return 0
        rsi = self.indicators.get("rsi14", 50.0)
        if 55 <= rsi <= 70:  return 10
        if 50 <= rsi <  55:  return 8
        if 70 <  rsi <= 75:  return 6
        if 45 <= rsi <  50:  return 5
        if 75 <  rsi <= 80:  return 3
        if 40 <= rsi <  45:  return 2
        return 0

    # ── Pattern detection (all 19 preserved — pruned ones score 0) ──────────

    def _detect(self):
        df = self.df
        if len(df) < self._w60:
            return

        close = df["Close"].squeeze().to_numpy(dtype=float)
        high  = df["High"].squeeze().to_numpy(dtype=float)
        low   = df["Low"].squeeze().to_numpy(dtype=float)
        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        n     = len(close)

        c20 = close[-self._w20:]; c10 = close[-self._w10:]; c5  = close[-self._w5:]
        v20 = vol[-self._w20:];   v10 = vol[-self._w10:];   v5  = vol[-self._w5:]

        price  = close[-1]
        avg20v = (float(np.mean(vol[-(self._w20+1):-1]))
                  if len(vol) > self._w20+1 and np.mean(vol[-(self._w20+1):-1]) > 0
                  else float(np.mean(v20)) if np.mean(v20) > 0 else 1)
        rvol   = float(vol[-2]) / avg20v if len(vol) >= 2 and avg20v > 0 else 0.8

        ema10_arr  = self._ema(close, self._w10)
        ema21_arr  = self._ema(close, self._w21)
        ema50_arr  = self._ema(close, self._w50)
        ema200_arr = self._ema(close, 200) if n >= 200 else np.zeros(n)

        above_50  = price > ema50_arr[-1]  if n >= self._w50  else False
        above_200 = price > ema200_arr[-1] if n >= 200 else False

        # w52_high/near_52w intentionally left unscaled (252-bar daily
        # window) -- only feeds the pruned "52W Momentum" pattern, see
        # scope note in __init__.
        w52_high = float(np.max(high[-252:])) if n >= 252 else float(np.max(high))
        near_52w = price >= 0.80 * w52_high

        detections = []

        # ── PRUNED as of 2026-07-27: no gate-cleared signal in 10yr replay,
        # untested rather than disproven — detection kept in case regime/
        # liquidity conditions change and it starts firing. See DEFAULT_WEIGHTS.

        # HIGH TIGHT FLAG — untested (0 gate-cleared signals, confirmed again 05 Aug)
        if n >= 40:
            pole_low   = float(np.min(low[-40:-15]))
            pole_high  = float(np.max(high[-40:-15]))
            flag_high  = float(np.max(high[-15:]))
            flag_low   = float(np.min(low[-15:]))
            flag_range_pct = (flag_high - flag_low) / flag_high if flag_high > 0 else 1
            pole_gain_pct  = (pole_high - pole_low) / pole_low  if pole_low  > 0 else 0
            vol_dry = float(np.mean(v5)) < 0.7 * avg20v
            if pole_gain_pct >= 0.80 and flag_range_pct <= 0.25 and vol_dry:
                detections.append(("High Tight Flag", flag_high * 1.003))

        # FLAT BASE — pruned 2026-07-27: net avg R=-0.07, N=219, p=0.9570 (05 Aug confirmation)
        if n >= 25:
            fb_range    = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                           if max(close[-25:]) > 0 else 1)
            fb_breakout = price >= 0.99 * float(np.max(high[-25:]))
            if fb_range <= 0.12 and fb_breakout and above_50:
                detections.append(("Flat Base", float(np.max(high[-25:]))))

        # ROUNDED BASE — pruned 2026-07-27: net avg R=-0.47, N=88, p=0.9974 (05 Aug confirmation)
        if n >= 60:
            mid        = len(close) // 2
            left_avg   = float(np.mean(close[:mid//2]))
            trough_avg = float(np.mean(close[mid//2:mid]))
            right_avg  = float(np.mean(close[mid:]))
            if (trough_avg < left_avg * 0.92
                    and right_avg > trough_avg * 1.05
                    and right_avg >= left_avg * 0.75):
                detections.append(("Rounded Base",
                                   float(np.max(high[-10:])) * 1.002))

        # HIGH BASE — pruned 2026-07-27: net avg R=-0.46, N=654, p=1.0000 (05 Aug confirmation)
        if n >= 25 and near_52w:
            hb_range = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                        if max(close[-25:]) > 0 else 1)
            if hb_range <= 0.15 and above_200:
                detections.append(("High Base",
                                   float(np.max(high[-25:])) * 1.002))

        # VOLUME EXPANSION — pruned 2026-07-27: net avg R=-0.65, N=45, p=0.9921 (05 Aug confirmation)
        if rvol >= 1.3 and price > close[-2]:
            resistance = float(np.max(high[-30:])) if n >= 30 else float(np.max(high))
            detections.append(("Volume Expansion", resistance))

        # ── REMAINING PATTERNS — mix of promoted (Cup & Handle, Swing High
        # Breakout) and pruned. VCP was promoted 2026-07-27 at reduced
        # weight, then PRUNED 2026-08-05 once a larger sample showed its
        # edge weakening rather than strengthening (see module docstring).
        # Everything else here stays pruned — detected for future
        # re-validation, weight=0.

        # VCP — pruned 2026-08-05: edge weakened with more data (N=50→70,
        # avg R 0.68→0.50, PF 2.16→1.78, p 0.1019→0.1525). See docstring
        # [PRUNE] section for full reasoning. Detection kept for future
        # re-testing if a larger/different sample becomes available.
        if n >= self._w60:
            ranges = []
            for w in [self._w20, self._w10, self._w5]:
                seg_h = float(np.max(high[-w:]))
                seg_l = float(np.min(low[-w:]))
                ranges.append((seg_h - seg_l) / seg_h if seg_h > 0 else 0)
            vcp_contracting = ranges[0] > ranges[1] > ranges[2]
            vcp_vol_dry     = float(np.mean(v5)) < 0.75 * avg20v
            vcp_breakout    = price >= 0.99 * float(np.max(high[-self._w20:]))
            if vcp_contracting and above_50 and vcp_vol_dry and vcp_breakout:
                detections.append(("VCP", float(np.max(high[-self._w20:]))))

        # SWING HIGH BREAKOUT — PROMOTED, weight 16, N=1456, p=0.0000 (05 Aug confirmation)
        if n >= self._w30:
            recent_swing = float(np.max(high[-self._w30:-self._w5]))
            if price >= recent_swing * 0.995 and (rvol >= 0.8 or price > recent_swing):
                detections.append(("Swing High Breakout", recent_swing))

        # BASE BREAKOUT — pruned: net avg R=-0.40, N=49, p=0.9652 (05 Aug confirmation)
        if n >= 20:
            base_top = float(np.max(high[-20:]))
            if price >= base_top * 0.99 and rvol >= 1.1 and above_50:
                detections.append(("Base Breakout", base_top))

        # BULL FLAG — pruned: net avg R=0.16, N=400, p=0.6642, not significant (05 Aug confirmation)
        if n >= 30:
            pole_gain  = (close[-10] - close[-25]) / close[-25] if close[-25] > 0 else 0
            flag_range = ((max(close[-10:]) - min(close[-10:])) / max(close[-10:])
                          if max(close[-10:]) > 0 else 1)
            flag_vol   = (float(np.mean(v5)) < 0.8 * float(np.mean(v20[-15:-5]))
                          if len(v20) >= 15 else False)
            if pole_gain >= 0.08 and flag_range <= 0.06 and (flag_vol or rvol >= 1.3):
                detections.append(("Bull Flag",
                                   float(max(close[-10:])) * 1.003))

        # CUP & HANDLE — PROMOTED, weight 20 (highest), N=1119, p=0.0000 (05 Aug confirmation)
        if n >= self._w60:
            cup_high     = float(np.max(high[-self._w60:-self._w30]))
            cup_low      = float(np.min(low[-self._w45:-self._w15]))
            recovery     = close[-self._w5] >= cup_high * 0.90
            handle_range = ((max(close[-self._w15:]) - min(close[-self._w15:])) / max(close[-self._w15:])
                            if max(close[-self._w15:]) > 0 else 1)
            handle_low_ok = float(min(low[-self._w15:])) >= cup_low
            if recovery and handle_range <= 0.08 and handle_low_ok:
                detections.append(("Cup & Handle", cup_high * 1.002))

        # DOUBLE BOTTOM — pruned: net avg R=-0.14, N=692, p=1.0000 (05 Aug confirmation)
        if n >= 40:
            lows40   = low[-40:]
            min1_idx = int(np.argmin(lows40[:20]))
            min2_idx = int(np.argmin(lows40[20:])) + 20
            bot1     = lows40[min1_idx]
            bot2     = lows40[min2_idx]
            similar_lows  = abs(bot1 - bot2) / bot1 <= 0.03 if bot1 > 0 else False
            neckline      = float(np.max(high[-40:])) * 0.99
            near_neckline = price >= neckline * 0.95
            if similar_lows and near_neckline:
                detections.append(("Double Bottom", neckline))

        # 52W MOMENTUM — pruned: net avg R=-0.48, N=715, p=1.0000 (05 Aug confirmation)
        if near_52w and above_50:
            detections.append(("52W Momentum", w52_high))

        # 3-WEEKS-TIGHT — pruned: net avg R=-0.35, N=103, p=0.9922 (05 Aug confirmation)
        if n >= 20:
            last3w    = close[-15:]
            range_pct = (max(last3w) - min(last3w)) / max(last3w) if max(last3w) > 0 else 1
            vol_dry3w = float(np.mean(v10)) < 0.8 * avg20v
            if range_pct <= 0.03 and vol_dry3w and above_50:
                detections.append(("3-Weeks-Tight",
                                   float(max(last3w)) * 1.002))

        # ASCENDING TRIANGLE — pruned: net avg R=-0.12, N=65, p=0.8699, inconclusive (05 Aug confirmation)
        if n >= 30:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            flat_top       = len(peaks) >= 2 and (max(peaks) - min(peaks)) / max(peaks) <= 0.02
            rising_bottoms = len(troughs) >= 2 and troughs[-1] > troughs[0]
            if flat_top and rising_bottoms:
                detections.append(("Ascending Triangle", max(peaks) * 1.002))

        # DESCENDING WEDGE — pruned: net avg R=-0.22, N=65, p=0.9368, inconclusive (05 Aug confirmation)
        if n >= 30:
            highs30 = [float(np.max(high[-30+i*5:-25+i*5])) for i in range(5)]
            lows30  = [float(np.min(low[-30+i*5:-25+i*5]))  for i in range(5)]
            dw_highs_falling = len(highs30) >= 2 and highs30[-1] < highs30[0]
            dw_lows_falling  = len(lows30)  >= 2 and lows30[-1]  < lows30[0]
            dw_narrowing     = (len(highs30) >= 2 and len(lows30) >= 2
                                and (highs30[-1] - lows30[-1]) < (highs30[0] - lows30[0]))
            if dw_highs_falling and dw_lows_falling and dw_narrowing and above_50:
                detections.append(("Descending Wedge",
                                   float(np.max(high[-5:])) * 1.003))

        # FALLING WEDGE — pruned: net avg R=0.22, N=102, p=0.4804, not significant (05 Aug confirmation)
        if n >= 30:
            h_slope = float(np.max(high[-5:])) - float(np.max(high[-30:-25]))
            l_slope = float(np.min(low[-5:]))  - float(np.min(low[-30:-25]))
            if h_slope < 0 and l_slope < 0 and l_slope < h_slope:
                detections.append(("Falling Wedge", price * 1.02))

        # SYMMETRICAL TRIANGLE — pruned: N=13, TOO THIN TO TEST (N<20), no verdict possible (05 Aug confirmation)
        if n >= 30:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            if len(peaks) >= 2 and len(troughs) >= 2:
                falling_peaks  = peaks[-1]   < peaks[0]
                rising_troughs = troughs[-1] > troughs[0]
                if falling_peaks and rising_troughs and rvol >= 1.3:
                    detections.append(("Symmetrical Triangle",
                                       float(np.max(high[-5:])) * 1.002))

        # DIAMOND BOTTOM — pruned: net avg R=-0.42, N=251, p=1.0000 (05 Aug confirmation)
        if n >= 40:
            phase1_range   = float(np.max(high[-40:-20])) - float(np.min(low[-40:-20]))
            phase2_range   = float(np.max(high[-20:]))    - float(np.min(low[-20:]))
            price_recovery = close[-1] > close[-10] and close[-1] > close[-20]
            if (phase1_range > 0
                    and phase2_range < phase1_range * 0.7
                    and price_recovery):
                detections.append(("Diamond Bottom",
                                   float(np.max(high[-20:])) * 1.002))

        # IPO BASE — untested (0 gate-cleared signals, confirmed again 05 Aug)
        if n < 300:
            base_high      = float(np.max(high))
            base_low       = float(np.min(low))
            base_range_pct = (base_high - base_low) / base_high if base_high > 0 else 1
            vol_dry        = float(np.mean(v10)) < 0.8 * avg20v
            if base_range_pct <= 0.20 and vol_dry and price >= 0.90 * base_high:
                detections.append(("IPO Base", base_high * 1.002))

        # v6.1: capture EVERY detection (weighted + pruned) before the
        # weight-based selection below discards all but one. This is what
        # lets the deep replay test all 19 patterns' real gate-adjusted
        # performance, instead of only the 2 that can win the max() below.
        self.all_detections = list(detections)

        # ── Select best detection by weight ──────────────────────────────────
        # Validated patterns (weight > 0) always beat pruned patterns (weight 0)
        if detections:
            best = max(detections, key=lambda x: ALL_WEIGHTS.get(x[0], 0))

            # Only assign pattern if it has positive weight
            if ALL_WEIGHTS.get(best[0], 0) > 0:
                self.pattern        = best[0]
                self.breakout_level = best[1]
                self.entry_low      = price * 0.995
                self.entry_high     = best[1] * 1.005
                self.raw_score      = DEFAULT_WEIGHTS.get(self.pattern, 0)

                res_6m = float(np.max(high[-self._w130:])) if n >= self._w130 else float(np.max(high))
                if best[1] >= res_6m * 0.98:
                    self.breakout_quality = "MAJOR"
                    self.raw_score = min(self.raw_score + 2, 20)
                else:
                    self.breakout_quality = "MINOR"

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> np.ndarray:
        # [FIX] Was a manual Python for-loop -- ~25x slower than pandas'
        # vectorized ewm(), verified bit-for-bit identical output before
        # this change. Called 4x per _detect() call (ema10/21/50/200) at
        # every sampled bar in a walk-forward replay, against windows
        # growing up to ~19,000 hourly bars deep -- this was a real,
        # measurable chunk of total replay runtime.
        return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()
