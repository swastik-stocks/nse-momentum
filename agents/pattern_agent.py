"""
NSE Momentum v5.2 — Pattern Agent (complete replacement)

19 patterns: VCP, Bull Flag, Flat Base, Base Breakout, Volume Expansion,
52W Momentum, Double Bottom, Cup & Handle, Ascending Triangle, Symmetrical Triangle,
Descending Wedge, Falling Wedge, Rounded Base, High Base, 3-Weeks-Tight (3WT),
Swing High Breakout, Diamond Bottom, High Tight Flag (v4.0), IPO Base (v4.0)

Plus: EMA score (8pts), MACD score (4pts), RSI score (10pts)
Dynamic weights applied from trade_logger.get_dynamic_weight()

CHANGES vs v4.3:
  [LIB] pandas-ta replaces ALL manual indicator calculations:
        _ema() loop  → pandas_ta.ema()
        manual RSI   → pandas_ta.rsi()
        manual MACD  → pandas_ta.macd()
        ATR          → pandas_ta.atr() (now stored, available to orchestrator)
  [SAFE] _ema() static method kept as internal fallback — pattern detection
         numpy paths still use it for speed (no Series overhead in tight loops)
  [API]  All public methods and attributes unchanged — orchestrator unaffected
         PatternAgent(df).pattern / .breakout_level / .entry_low / .entry_high
         .score() / .get_ema_score() / .get_macd_score() / .get_rsi_score()
  [NEW]  .get_atr_pct() — returns ATR as % of price, used by AsymmetryGate
         .indicators   — dict of pre-computed indicator values for inspection
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
        "pandas-ta not installed — falling back to manual indicator calculations. "
        "Run: pip install pandas-ta --no-deps"
    )

log = logging.getLogger(__name__)

# ── Default pattern weights ────────────────────────────────────────────────────
# Overridden at runtime by trade_logger.get_dynamic_weight()
DEFAULT_WEIGHTS = {
    "VCP":                  16,
    "Bull Flag":            12,
    "Flat Base":            10,
    "Base Breakout":        14,
    "Volume Expansion":     12,
    "52W Momentum":         13,
    "Double Bottom":        15,
    "Cup & Handle":         11,
    "Ascending Triangle":   13,
    "Symmetrical Triangle": 11,
    "Descending Wedge":     12,
    "Falling Wedge":         8,
    "Rounded Base":         13,
    "High Base":            14,
    "3-Weeks-Tight":        15,
    "Swing High Breakout":  17,
    "Diamond Bottom":       14,
    "High Tight Flag":      17,
    "IPO Base":             15,
}


class PatternAgent:
    def __init__(self, df: pd.DataFrame):
        self.df             = df.copy()
        self.pattern        = ""
        self.breakout_level = 0.0
        self.entry_low      = 0.0
        self.entry_high     = 0.0
        self.raw_score      = 0
        self.breakout_quality = "MINOR"
        self.indicators     = {}   # pre-computed, available for inspection
        self._compute_indicators()
        self._detect()

    # ── Indicator computation (pandas-ta with fallback) ───────────────────────

    def _compute_indicators(self):
        """
        Pre-compute all indicators once via pandas-ta.
        Results stored in self.indicators dict.
        Falls back to manual calculations if pandas-ta unavailable.
        """
        df   = self.df
        n    = len(df)
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()

        ind = {}

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
                    ind["macd"]        = float(macd_df[cols[0]].iloc[-1])   # MACD line
                    ind["macd_signal"] = float(macd_df[cols[1]].iloc[-1])   # Signal line
                    ind["macd_hist"]   = float(macd_df[cols[2]].iloc[-1])   # Histogram
                    ind["macd_hist_prev"] = float(macd_df[cols[2]].iloc[-2]) if n >= 27 else 0.0
                else:
                    ind["macd"] = ind["macd_signal"] = ind["macd_hist"] = ind["macd_hist_prev"] = 0.0

                if n >= 15:
                    atr_s = ta.atr(high, low, close, length=14)
                    ind["atr14"] = float(atr_s.iloc[-1]) if atr_s is not None else 0.0
                else:
                    ind["atr14"] = 0.0

            except Exception as e:
                log.debug(f"pandas-ta compute failed, using fallback: {e}")
                self._compute_indicators_fallback(ind)
        else:
            self._compute_indicators_fallback(ind)

        self.indicators = ind

    def _compute_indicators_fallback(self, ind: dict):
        """Manual numpy fallback — identical to v4.3 logic."""
        df    = self.df
        n     = len(df)
        close = df["Close"].squeeze().to_numpy(dtype=float)
        high  = df["High"].squeeze().to_numpy(dtype=float)
        low   = df["Low"].squeeze().to_numpy(dtype=float)

        ind["ema10"]  = self._ema(close, 10)[-1]  if n >= 10  else 0.0
        ind["ema21"]  = self._ema(close, 21)[-1]  if n >= 21  else 0.0
        ind["ema50"]  = self._ema(close, 50)[-1]  if n >= 50  else 0.0
        ind["ema200"] = self._ema(close, 200)[-1] if n >= 200 else 0.0

        # RSI-14
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

        # MACD
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

        # ATR-14
        if n >= 15:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]),
                           np.abs(low[1:]  - close[:-1]))
            )
            ind["atr14"] = float(np.mean(tr[-14:]))
        else:
            ind["atr14"] = 0.0

    # ── Public helper — new in v5.2 ───────────────────────────────────────────

    def get_atr_pct(self) -> float:
        """ATR-14 as % of last close price. Used by AsymmetryGate."""
        price = float(self.df["Close"].squeeze().iloc[-1])
        atr   = self.indicators.get("atr14", 0.0)
        return round(atr / price * 100, 2) if price > 0 else 0.0

    # ── Pattern detection (identical to v4.3 — no changes) ───────────────────

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

        # Use pre-computed EMAs from pandas-ta where possible
        ind       = self.indicators
        ema10_arr = self._ema(close, 10)
        ema21_arr = self._ema(close, 21)
        ema50_arr = self._ema(close, 50)
        ema200_arr = self._ema(close, 200) if n >= 200 else np.zeros(n)

        above_50  = price > ema50_arr[-1]  if n >= 50  else False
        above_200 = price > ema200_arr[-1] if n >= 200 else False

        w52_high = float(np.max(high[-252:])) if n >= 252 else float(np.max(high))
        w52_low  = float(np.min(low[-252:]))  if n >= 252 else float(np.min(low))
        near_52w = price >= 0.80 * w52_high

        detections = []

        # 1. HIGH TIGHT FLAG
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

        # 2. IPO BASE
        if n < 300:
            base_high = float(np.max(high))
            base_low  = float(np.min(low))
            base_range_pct = (base_high - base_low) / base_high if base_high > 0 else 1
            vol_dry = float(np.mean(v10)) < 0.8 * avg20v
            if base_range_pct <= 0.20 and vol_dry and price >= 0.90 * base_high:
                detections.append(("IPO Base", base_high * 1.002))

        # 3. VCP
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

        # 4. SWING HIGH BREAKOUT
        if n >= 30:
            recent_swing = float(np.max(high[-30:-5]))
            if price >= recent_swing * 0.995 and (rvol >= 0.8 or price > recent_swing):
                detections.append(("Swing High Breakout", recent_swing))

        # 4b. RECOVERY BREAKOUT
        if n >= 65 and not near_52w:
            recovery_resistance = float(np.max(high[-60:-5]))
            if price >= recovery_resistance * 0.99 and above_50 and rvol >= 0.8:
                detections.append(("Base Breakout", recovery_resistance))
            elif (price >= recovery_resistance * 0.92 and above_50
                  and close[-1] > close[-5] > close[-20]):
                detections.append(("Rounded Base", recovery_resistance))

        # 5. 3-WEEKS-TIGHT
        if n >= 20:
            last3w    = close[-15:]
            range_pct = (max(last3w) - min(last3w)) / max(last3w) if max(last3w) > 0 else 1
            vol_dry3w = float(np.mean(v10)) < 0.8 * avg20v
            if range_pct <= 0.03 and vol_dry3w and above_50:
                detections.append(("3-Weeks-Tight", float(max(last3w)) * 1.002))

        # 6. BULL FLAG
        if n >= 30:
            pole_gain  = (close[-10] - close[-25]) / close[-25] if close[-25] > 0 else 0
            flag_range = ((max(close[-10:]) - min(close[-10:])) / max(close[-10:])
                          if max(close[-10:]) > 0 else 1)
            flag_vol   = (float(np.mean(v5)) < 0.8 * float(np.mean(v20[-15:-5]))
                          if len(v20) >= 15 else False)
            breakout_vol = rvol >= 1.3
            if pole_gain >= 0.08 and flag_range <= 0.06 and (flag_vol or breakout_vol):
                detections.append(("Bull Flag", float(max(close[-10:])) * 1.003))

        # 7. FLAT BASE
        if n >= 25:
            fb_range   = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                          if max(close[-25:]) > 0 else 1)
            fb_breakout = price >= 0.99 * float(np.max(high[-25:]))
            if fb_range <= 0.12 and fb_breakout and above_50:
                detections.append(("Flat Base", float(np.max(high[-25:]))))

        # 8. CUP & HANDLE
        if n >= 60:
            cup_high    = float(np.max(high[-60:-30]))
            cup_low     = float(np.min(low[-45:-15]))
            recovery    = close[-5] >= cup_high * 0.90
            handle_range = ((max(close[-15:]) - min(close[-15:])) / max(close[-15:])
                            if max(close[-15:]) > 0 else 1)
            handle_low_ok = float(min(low[-15:])) >= cup_low
            if recovery and handle_range <= 0.08 and handle_low_ok:
                detections.append(("Cup & Handle", cup_high * 1.002))

        # 9. DOUBLE BOTTOM
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

        # 10. BASE BREAKOUT
        if n >= 20:
            base_top = float(np.max(high[-20:]))
            if price >= base_top * 0.99 and rvol >= 1.1 and above_50:
                detections.append(("Base Breakout", base_top))

        # 11. VOLUME EXPANSION
        if rvol >= 1.3 and price > close[-2]:
            resistance = float(np.max(high[-30:])) if n >= 30 else float(np.max(high))
            detections.append(("Volume Expansion", resistance))

        # 12. 52W MOMENTUM
        if near_52w and above_50 and (n < 252 or price >= 0.85 * w52_low + 0.15 * w52_high):
            detections.append(("52W Momentum", w52_high))

        # 12b. MOMENTUM RISING
        if n >= 60 and not detections:
            low_60        = float(np.min(low[-60:]))
            gain_from_low = (price - low_60) / low_60 if low_60 > 0 else 0
            higher_highs  = float(np.max(high[-10:])) > float(np.max(high[-30:-10]))
            if gain_from_low >= 0.12 and above_50 and higher_highs and rvol >= 0.8:
                detections.append(("Swing High Breakout", float(np.max(high[-10:])) * 1.002))

        # 13. ASCENDING TRIANGLE
        if n >= 30:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            flat_top       = len(peaks) >= 2 and (max(peaks) - min(peaks)) / max(peaks) <= 0.02
            rising_bottoms = len(troughs) >= 2 and troughs[-1] > troughs[0]
            if flat_top and rising_bottoms:
                detections.append(("Ascending Triangle", max(peaks) * 1.002))

        # 14. DESCENDING WEDGE
        if n >= 30:
            highs30 = [float(np.max(high[-30+i*5:-25+i*5])) for i in range(5)]
            lows30  = [float(np.min(low[-30+i*5:-25+i*5]))  for i in range(5)]
            dw_highs_falling = len(highs30) >= 2 and highs30[-1] < highs30[0]
            dw_lows_falling  = len(lows30)  >= 2 and lows30[-1]  < lows30[0]
            dw_narrowing     = (len(highs30) >= 2 and len(lows30) >= 2
                                and (highs30[-1] - lows30[-1]) < (highs30[0] - lows30[0]))
            if dw_highs_falling and dw_lows_falling and dw_narrowing and above_50:
                detections.append(("Descending Wedge", float(np.max(high[-5:])) * 1.003))

        # 15. FALLING WEDGE
        if n >= 30 and not [d for d in detections if "Descending" in d[0]]:
            h_slope = float(np.max(high[-5:])) - float(np.max(high[-30:-25]))
            l_slope = float(np.min(low[-5:]))  - float(np.min(low[-30:-25]))
            if h_slope < 0 and l_slope < 0 and l_slope < h_slope:
                detections.append(("Falling Wedge", price * 1.02))

        # 16. ROUNDED BASE
        if n >= 60:
            mid        = len(close) // 2
            left_avg   = float(np.mean(close[:mid//2]))
            trough_avg = float(np.mean(close[mid//2:mid]))
            right_avg  = float(np.mean(close[mid:]))
            if (trough_avg < left_avg * 0.92
                    and right_avg > trough_avg * 1.05
                    and right_avg >= left_avg * 0.75):
                detections.append(("Rounded Base", float(np.max(high[-10:])) * 1.002))

        # 17. HIGH BASE
        if n >= 25 and near_52w:
            hb_range = ((max(close[-25:]) - min(close[-25:])) / max(close[-25:])
                        if max(close[-25:]) > 0 else 1)
            if hb_range <= 0.15 and above_200:
                detections.append(("High Base", float(np.max(high[-25:])) * 1.002))

        # 18. SYMMETRICAL TRIANGLE
        if n >= 30 and not detections:
            peaks   = [float(np.max(high[i:i+5])) for i in range(-30, -5, 5)]
            troughs = [float(np.min(low[i:i+5]))  for i in range(-30, -5, 5)]
            if len(peaks) >= 2 and len(troughs) >= 2:
                falling_peaks  = peaks[-1]   < peaks[0]
                rising_troughs = troughs[-1] > troughs[0]
                if falling_peaks and rising_troughs and rvol >= 1.3:
                    detections.append(("Symmetrical Triangle",
                                       float(np.max(high[-5:])) * 1.002))

        # 19. DIAMOND BOTTOM
        if n >= 40 and not detections:
            phase1_range   = float(np.max(high[-40:-20])) - float(np.min(low[-40:-20]))
            phase2_range   = float(np.max(high[-20:]))    - float(np.min(low[-20:]))
            price_recovery = close[-1] > close[-10] and close[-1] > close[-20]
            if (phase1_range > 0
                    and phase2_range < phase1_range * 0.7
                    and price_recovery):
                detections.append(("Diamond Bottom", float(np.max(high[-20:])) * 1.002))

        # ── Select best detection (highest weight) ────────────────────────────
        if detections:
            best = max(detections, key=lambda x: DEFAULT_WEIGHTS.get(x[0], 10))
            self.pattern        = best[0]
            self.breakout_level = best[1]
            self.entry_low      = price * 0.995
            self.entry_high     = best[1] * 1.005
            self.raw_score      = DEFAULT_WEIGHTS.get(self.pattern, 10)

            res_6m = float(np.max(high[-130:])) if n >= 130 else float(np.max(high))
            if best[1] >= res_6m * 0.98:
                self.breakout_quality = "MAJOR"
                self.raw_score = min(self.raw_score + 2, 18)
            else:
                self.breakout_quality = "MINOR"

    # ── Public scoring methods ────────────────────────────────────────────────

    def score(self) -> int:
        return self.raw_score

    def get_ema_score(self) -> int:
        """EMA alignment score (0-8). Uses pre-computed pandas-ta values."""
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
        """MACD bullishness score (0-4). Uses pre-computed pandas-ta values."""
        if len(self.df) < 35:
            return 0
        ind = self.indicators
        pts = 0
        if ind.get("macd", 0) > 0:                                   pts += 1
        if ind.get("macd", 0) > ind.get("macd_signal", 0):           pts += 2
        if ind.get("macd_hist", 0) > ind.get("macd_hist_prev", 0):   pts += 1
        return min(pts, 4)

    def get_rsi_score(self) -> int:
        """RSI score (0-10). Sweet spot 55-70. Uses pre-computed pandas-ta value."""
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> np.ndarray:
        """
        Vectorised EMA fallback used in pattern detection numpy paths.
        Kept for speed — avoids Series overhead inside tight detection loops.
        """
        alpha = 2 / (span + 1)
        ema   = np.empty(len(values))
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema
