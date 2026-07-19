"""
NSE Momentum v5.3 — Pattern Agent (complete replacement)

CHANGES vs v5.2:
  [PRUNE] 14 patterns removed from DEFAULT_WEIGHTS after validation showed
          negative expectancy on 2yr NSE data (2023-2026).
  [KEEP]  5 patterns retained with weights proportional to validated edge:
          High Tight Flag  +0.69% expectancy → weight 20
          Flat Base        +0.51% expectancy → weight 18
          Rounded Base     +0.40% expectancy → weight 15
          High Base        +0.09% expectancy → weight 12
          Volume Expansion  0.00% expectancy → weight 8 (kept as filter signal)
  [DETECT] All 19 pattern detection blocks preserved in _detect() —
           pruned patterns still detected but score 0, so they never
           win the "best pattern" selection against a WATCH pattern.
           This means if a stock has both Rounded Base AND Swing High
           Breakout, Rounded Base wins. If it ONLY has Swing High
           Breakout, no pattern is assigned → stock rejected at G2.
  [API]  All public methods unchanged — orchestrator unaffected.
"""

import logging
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    _PANDAS_TA = True
except ImportError:
    _PANDAS_TA = False
    logging.getLogger(__name__).warning(
        "pandas-ta not installed — falling back to manual calculations. "
        "Run: pip install pandas-ta --no-deps"
    )

log = logging.getLogger(__name__)

# ── Validated pattern weights (v5.3 — post-validation pruning) ───────────────
# Only patterns with positive expectancy in 2023-2026 NSE backtest.
# 14 patterns removed: VCP, Bull Flag, Base Breakout, 52W Momentum,
# Double Bottom, Cup & Handle, Ascending Triangle, Symmetrical Triangle,
# Descending Wedge, Falling Wedge, 3-Weeks-Tight, Swing High Breakout,
# Diamond Bottom, IPO Base.
DEFAULT_WEIGHTS = {
    "High Tight Flag":   20,   # +0.69% exp, 163 signals, PF 1.21x
    "Flat Base":         18,   # +0.51% exp,  40 signals, PF 1.24x
    "Rounded Base":      15,   # +0.40% exp, 7216 signals, PF 1.17x
    "High Base":         12,   # +0.09% exp, 20916 signals, PF 1.05x
    "Volume Expansion":   8,   #  0.00% exp, 6632 signals, PF 1.00x — filter only
}

# Pruned patterns — detected but weight=0 so never selected as best pattern
PRUNED_PATTERNS = {
    "VCP":                  0,
    "Bull Flag":            0,
    "Base Breakout":        0,
    "52W Momentum":         0,
    "Double Bottom":        0,
    "Cup & Handle":         0,
    "Ascending Triangle":   0,
    "Symmetrical Triangle": 0,
    "Descending Wedge":     0,
    "Falling Wedge":        0,
    "3-Weeks-Tight":        0,
    "Swing High Breakout":  0,
    "Diamond Bottom":       0,
    "IPO Base":             0,
}

ALL_WEIGHTS = {**DEFAULT_WEIGHTS, **PRUNED_PATTERNS}

# ── Backtested expectancy per validated pattern (from the same 2023-2026 ──
# NSE backtest that produced DEFAULT_WEIGHTS above). Used to flag patterns
# whose historical edge rounds to statistical noise, even though they still
# score high enough to clear the Tier 1 gate on RS/volume/EMA/etc alone.
PATTERN_EXPECTANCY = {
    "High Tight Flag":   0.69,
    "Flat Base":         0.51,
    "Rounded Base":      0.40,
    "High Base":         0.09,
    "Volume Expansion":  0.00,
}
LOW_EDGE_EXPECTANCY_THRESHOLD = 0.15   # below this, don't badge the pick as high-conviction Tier 1


def is_low_edge_pattern(pattern_name: str) -> bool:
    """
    True if this pattern's own backtested expectancy is below the threshold
    for a high-conviction label — e.g. High Base (+0.09%) and Volume
    Expansion (0.00%) can still legitimately clear the Tier 1 SCORE gate on
    RS/volume/EMA strength alone, but their own historical data says the
    pattern itself doesn't carry real edge. That distinction should be
    visible to the trader, not flattened into the same 'MAJOR' badge as
    High Tight Flag (+0.69%).
    """
    return PATTERN_EXPECTANCY.get(pattern_name, 1.0) < LOW_EDGE_EXPECTANCY_THRESHOLD


class PatternAgent:
    def __init__(self, df: pd.DataFrame):
        self.df               = df.copy()
        self.pattern          = ""
        self.breakout_level   = 0.0
        self.entry_low        = 0.0
        self.entry_high       = 0.0
        self.raw_score        = 0
        self.breakout_quality = "MINOR"
        self.indicators       = {}
        self._compute_indicators()
        self._detect()

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self):
        df    = self.df
        n     = len(df)
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        ind   = {}

        if _PANDAS_TA and n >= 26:
            try:
                ind["ema10"]  = float(ta.ema(close, length=10).iloc[-1])
                ind["ema21"]  = float(ta.ema(close, length=21).iloc[-1])
                ind["ema50"]  = float(ta.ema(close, length=50).iloc[-1]) if n >= 50 else 0.0
                ind["ema200"] = float(ta.ema(close, length=200).iloc[-1]) if n >= 200 else 0.0

                rsi_s = ta.rsi(close, length=14)
                ind["rsi14"] = float(rsi_s.iloc[-1]) if rsi_s is not None else 50.0

                macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty:
                    cols = macd_df.columns.tolist()
                    ind["macd"]           = float(macd_df[cols[0]].iloc[-1])
                    ind["macd_signal"]    = float(macd_df[cols[1]].iloc[-1])
                    ind["macd_hist"]      = float(macd_df[cols[2]].iloc[-1])
                    ind["macd_hist_prev"] = float(macd_df[cols[2]].iloc[-2]) if n >= 27 else 0.0
                else:
                    ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = ind["macd_hist_prev"] = 0.0

                if n >= 15:
                    atr_s = ta.atr(high, low, close, length=14)
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

        ind["ema10"]  = self._ema(close, 10)[-1]  if n >= 10  else 0.0
        ind["ema21"]  = self._ema(close, 21)[-1]  if n >= 21  else 0.0
        ind["ema50"]  = self._ema(close, 50)[-1]  if n >= 50  else 0.0
        ind["ema200"] = self._ema(close, 200)[-1] if n >= 200 else 0.0

        if n >= 15:
            delta  = np.diff(close[-15:])
            gain   = np.where(delta > 0, delta, 0)
            loss   = np.where(delta < 0, -delta, 0)
            avg_g  = np.mean(gain) if gain.any() else 1e-9
            avg_l  = np.mean(loss) if loss.any() else 1e-9
            rs     = avg_g / avg_l if avg_l > 0 else 99
            ind["rsi14"] = 100 - 100 / (1 + rs)
        else:
            ind["rsi14"] = 50.0

        if n >= 26:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd  = ema12 - ema26
            sig   = self._ema(macd, 9)
            hist  = macd - sig
            ind["macd"]           = macd[-1]
            ind["macd_signal"]    = sig[-1]
            ind["macd_hist"]      = hist[-1]
            ind["macd_hist_prev"] = hist[-2] if n >= 27 else 0.0
        else:
            ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = ind["macd_hist_prev"] = 0.0

        if n >= 15:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:]  - close[:-1]))
            )
            ind["atr14"] = float(np.mean(tr[-14:]))
        else:
            ind["atr14"] = 0.0

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_atr_pct(self) -> float:
        price = float(self.df["Close"].squeeze().iloc[-1])
        atr   = self.indicators.get("atr14", 0.0)
        return round(atr / price * 100, 2) if price > 0 else 0.0

    def score(self) -> int:
        return self.raw_score

    def get_ema_score(self) -> int:
        if len(self.df) < 50:
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
        if len(self.df) < 35:
            return 0
        ind = self.indicators
        pts = 0
        if ind.get("macd", 0) > 0:                                  pts += 1
        if ind.get("macd", 0) > ind.get("macd_signal", 0):          pts += 2
        if ind.get("macd_hist", 0) > ind.get("macd_hist_prev", 0):  pts += 1
        return min(pts, 4)

    def get_rsi_score(self) -> int:
        if len(self.df) < 15:
            return 0
        rsi = self.indicators.get("rsi14", 50.0)
        if 55 <= rsi <= 70:  return 10
        if 50 <= rsi <  55:  return 8
        if 70 <  rsi <= 75:  return 6
        if 45 <= rsi <  50:  return 5
        if 75 <  rsi <= 80:  return 3
        if 40 <= rsi <  45:  return 2
        return 0

    # ── Pattern detection (all 19 preserved — pruned ones score 0) ───────────

    def _detect(self):
        df = self.df
        if len(df) < 60:
            return

        close = df["Close"].squeeze().to_numpy(dtype=float)
        high  = df["High"].squeeze().to_numpy(dtype=float)
        low   = df["Low"].squeeze().to_numpy(dtype=float)
        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        n     = len(close)

        c20 = close[-20:]; c10 = close[-10:]; c5  = close[-5:]
        v20 = vol[-20:];   v10 = vol[-10:];   v5  = vol[-5:]

        price  = close[-1]
        avg20v = (float(np.mean(vol[-21:-1]))
                  if len(vol) > 21 and np.mean(vol[-21:-1]) > 0
                  else float(np.mean(v20)) if np.mean(v20) > 0 else 1)
        rvol   = float(vol[-2]) / avg20v if len(vol) >= 2 and avg20v > 0 else 0.8

        ema10_arr  = self._ema(close, 10)
        ema21_arr  = self._ema(close, 21)
        ema50_arr  = self._ema(close, 50)
        ema200_arr = self._ema(close, 200) if n >= 200 else np.zeros(n)

        above_50  = price > ema50_arr[-1]  if n >= 50  else False
        above_200 = price > ema200_arr[-1] if n >= 200 else False

        w52_high = float(np.max(high[-252:])) if n >= 252 else float(np.max(high))
        near_52w = price >= 0.80 * w52_high

        detections = []

        # ── VALIDATED PATTERNS (positive expectancy — will be selected) ───────

        # 1. HIGH TIGHT FLAG — best edge (+0.69%)
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

        # 2. FLAT BASE — good edge (+0.51%)
        if n >= 25:
            fb_range    = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                           if max(close[-25:]) > 0 else 1)
            fb_breakout = price >= 0.99 * float(np.max(high[-25:]))
            if fb_range <= 0.12 and fb_breakout and above_50:
                detections.append(("Flat Base", float(np.max(high[-25:]))))

        # 3. ROUNDED BASE — solid edge (+0.40%, large sample)
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

        # 4. HIGH BASE — marginal edge (+0.09%, very large sample confirms it)
        if n >= 25 and near_52w:
            hb_range = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                        if max(close[-25:]) > 0 else 1)
            if hb_range <= 0.15 and above_200:
                detections.append(("High Base",
                                   float(np.max(high[-25:])) * 1.002))

        # 5. VOLUME EXPANSION — break-even, kept as confirming signal
        if rvol >= 1.3 and price > close[-2]:
            resistance = float(np.max(high[-30:])) if n >= 30 else float(np.max(high))
            detections.append(("Volume Expansion", resistance))

        # ── PRUNED PATTERNS (detected but weight=0 — only win if no WATCH pattern) ──
        # Detection logic preserved exactly — useful for future re-validation
        # after adding regime filter.

        # VCP
        if n >= 60:
            ranges = []
            for w in [20, 10, 5]:
                seg_h = float(np.max(high[-w:]))
                seg_l = float(np.min(low[-w:]))
                ranges.append((seg_h - seg_l) / seg_h if seg_h > 0 else 0)
            vcp_contracting = ranges[0] > ranges[1] > ranges[2]
            vcp_vol_dry     = float(np.mean(v5)) < 0.75 * avg20v
            vcp_breakout    = price >= 0.99 * float(np.max(high[-20:]))
            if vcp_contracting and above_50 and vcp_vol_dry and vcp_breakout:
                detections.append(("VCP", float(np.max(high[-20:]))))

        # SWING HIGH BREAKOUT
        if n >= 30:
            recent_swing = float(np.max(high[-30:-5]))
            if price >= recent_swing * 0.995 and (rvol >= 0.8 or price > recent_swing):
                detections.append(("Swing High Breakout", recent_swing))

        # BASE BREAKOUT
        if n >= 20:
            base_top = float(np.max(high[-20:]))
            if price >= base_top * 0.99 and rvol >= 1.1 and above_50:
                detections.append(("Base Breakout", base_top))

        # BULL FLAG
        if n >= 30:
            pole_gain  = (close[-10] - close[-25]) / close[-25] if close[-25] > 0 else 0
            flag_range = ((max(close[-10:]) - min(close[-10:])) / max(close[-10:])
                          if max(close[-10:]) > 0 else 1)
            flag_vol   = (float(np.mean(v5)) < 0.8 * float(np.mean(v20[-15:-5]))
                          if len(v20) >= 15 else False)
            if pole_gain >= 0.08 and flag_range <= 0.06 and (flag_vol or rvol >= 1.3):
                detections.append(("Bull Flag",
                                   float(max(close[-10:])) * 1.003))

        # CUP & HANDLE
        if n >= 60:
            cup_high     = float(np.max(high[-60:-30]))
            cup_low      = float(np.min(low[-45:-15]))
            recovery     = close[-5] >= cup_high * 0.90
            handle_range = ((max(close[-15:]) - min(close[-15:])) / max(close[-15:])
                            if max(close[-15:]) > 0 else 1)
            handle_low_ok = float(min(low[-15:])) >= cup_low
            if recovery and handle_range <= 0.08 and handle_low_ok:
                detections.append(("Cup & Handle", cup_high * 1.002))

        # DOUBLE BOTTOM
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

        # 52W MOMENTUM
        if near_52w and above_50:
            detections.append(("52W Momentum", w52_high))

        # 3-WEEKS-TIGHT
        if n >= 20:
            last3w    = close[-15:]
            range_pct = (max(last3w) - min(last3w)) / max(last3w) if max(last3w) > 0 else 1
            vol_dry3w = float(np.mean(v10)) < 0.8 * avg20v
            if range_pct <= 0.03 and vol_dry3w and above_50:
                detections.append(("3-Weeks-Tight",
                                   float(max(last3w)) * 1.002))

        # ASCENDING TRIANGLE
        if n >= 30:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            flat_top       = len(peaks) >= 2 and (max(peaks) - min(peaks)) / max(peaks) <= 0.02
            rising_bottoms = len(troughs) >= 2 and troughs[-1] > troughs[0]
            if flat_top and rising_bottoms:
                detections.append(("Ascending Triangle", max(peaks) * 1.002))

        # DESCENDING WEDGE
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

        # FALLING WEDGE
        if n >= 30:
            h_slope = float(np.max(high[-5:])) - float(np.max(high[-30:-25]))
            l_slope = float(np.min(low[-5:]))  - float(np.min(low[-30:-25]))
            if h_slope < 0 and l_slope < 0 and l_slope < h_slope:
                detections.append(("Falling Wedge", price * 1.02))

        # SYMMETRICAL TRIANGLE
        if n >= 30:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            if len(peaks) >= 2 and len(troughs) >= 2:
                falling_peaks  = peaks[-1]   < peaks[0]
                rising_troughs = troughs[-1] > troughs[0]
                if falling_peaks and rising_troughs and rvol >= 1.3:
                    detections.append(("Symmetrical Triangle",
                                       float(np.max(high[-5:])) * 1.002))

        # DIAMOND BOTTOM
        if n >= 40:
            phase1_range   = float(np.max(high[-40:-20])) - float(np.min(low[-40:-20]))
            phase2_range   = float(np.max(high[-20:]))    - float(np.min(low[-20:]))
            price_recovery = close[-1] > close[-10] and close[-1] > close[-20]
            if (phase1_range > 0
                    and phase2_range < phase1_range * 0.7
                    and price_recovery):
                detections.append(("Diamond Bottom",
                                   float(np.max(high[-20:])) * 1.002))

        # IPO BASE
        if n < 300:
            base_high      = float(np.max(high))
            base_low       = float(np.min(low))
            base_range_pct = (base_high - base_low) / base_high if base_high > 0 else 1
            vol_dry        = float(np.mean(v10)) < 0.8 * avg20v
            if base_range_pct <= 0.20 and vol_dry and price >= 0.90 * base_high:
                detections.append(("IPO Base", base_high * 1.002))

        # ── Select best detection by weight ───────────────────────────────────
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

                res_6m = float(np.max(high[-130:])) if n >= 130 else float(np.max(high))
                if best[1] >= res_6m * 0.98:
                    self.breakout_quality = "MAJOR"
                    self.raw_score = min(self.raw_score + 2, 20)
                else:
                    self.breakout_quality = "MINOR"

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        ema   = np.empty(len(values))
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema
