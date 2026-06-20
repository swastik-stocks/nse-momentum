"""
NSE Momentum v5.0 - Institutional Proxy Agent v5
10A: Delivery % accumulation score
10B: RVOL quality (up-volume vs down-volume)
10C: Volume contraction/expansion pattern
10D: Shareholding change proxy (NSE API)
10E: Bulk/block deal detection

v5 additions:
  - Rolling delivery trend: rewards sustained rising delivery over 5 days
  - Distribution risk flag: price rising + delivery falling = warning
  - Score: 0-20 pts (unchanged ceiling, better internals)
"""

import logging
import requests
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class InstitutionalProxyAgent:
    def __init__(self, ticker: str, df: pd.DataFrame,
                 delivery_pct: float = 0.0):
        self.ticker       = ticker
        self.df           = df
        self.delivery_pct = delivery_pct

    def evaluate(self) -> dict:
        s10a                = self._10a_delivery()
        s10b                = self._10b_rvol_quality()
        s10c                = self._10c_volume_pattern()
        s10d                = self._10d_shareholding()
        s10e                = self._10e_bulk_deals()
        rolling_bonus       = self._rolling_delivery_trend()
        distribution_penalty = self._distribution_risk()

        total = min(
            s10a + s10b + s10c + s10d + s10e
            + rolling_bonus - distribution_penalty,
            20
        )
        total = max(total, 0)

        return {
            "institutional_proxy_score": total,
            "10a_delivery":      s10a,
            "10b_rvol_quality":  s10b,
            "10c_volume_pattern": s10c,
            "10d_shareholding":  s10d,
            "10e_bulk_deals":    s10e,
            "rolling_delivery_bonus":    rolling_bonus,
            "distribution_risk_penalty": distribution_penalty,
        }

    def _10a_delivery(self) -> int:
        """Delivery % as institutional accumulation signal. 0-5 pts."""
        d = self.delivery_pct
        if d >= 70: return 5
        if d >= 55: return 4
        if d >= 40: return 3
        if d >= 25: return 2
        if d >= 10: return 1
        return 0

    def _10b_rvol_quality(self) -> int:
        """Up-volume vs down-volume quality. 0-5 pts."""
        df = self.df
        if len(df) < 20:
            return 2
        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        close = df["Close"].squeeze().to_numpy(dtype=float)

        up_vol   = [vol[i] for i in range(-20, 0) if close[i] > close[i-1]]
        down_vol = [vol[i] for i in range(-20, 0) if close[i] < close[i-1]]

        avg_up   = float(np.mean(up_vol))   if up_vol   else 0
        avg_down = float(np.mean(down_vol)) if down_vol else avg_up + 1
        ratio    = avg_up / avg_down if avg_down > 0 else 1.0

        if ratio >= 2.0: return 5
        if ratio >= 1.5: return 4
        if ratio >= 1.2: return 3
        if ratio >= 0.9: return 2
        return 1

    def _10c_volume_pattern(self) -> int:
        """Volume Dry-Up and Expansion Breakout detection. 0-4 pts."""
        df = self.df
        if len(df) < 20:
            return 1
        vol    = df["Volume"].squeeze().to_numpy(dtype=float)
        avg20v = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1

        vdu       = float(np.mean(vol[-5:])) < 0.60 * avg20v
        expansion = float(vol[-1]) > 1.5 * avg20v

        if vdu and expansion: return 4
        if expansion:         return 3
        if vdu:               return 3
        return 1

    def _rolling_delivery_trend(self) -> int:
        """
        v5 NEW: Reward rising delivery % across 5-day rolling window.
        Uses bhavcopy delivery data stored in factor_store if available,
        otherwise uses single-day delivery_pct as proxy.
        Returns 0-2 bonus pts.
        """
        df = self.df
        if len(df) < 10 or self.delivery_pct <= 0:
            return 0

        # Proxy: use volume trend as delivery trend signal
        # (real delivery history requires bhavcopy multi-day store)
        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        close = df["Close"].squeeze().to_numpy(dtype=float)

        # Rising volume on up days in last 5 sessions = accumulation trend
        recent_up_vol = []
        for i in range(-5, 0):
            if close[i] > close[i-1]:
                recent_up_vol.append(vol[i])

        avg5_up = float(np.mean(recent_up_vol)) if recent_up_vol else 0
        avg20   = float(np.mean(vol[-20:])) if len(vol) >= 20 else avg5_up

        # Rising up-volume trend + high delivery = sustained accumulation
        if avg5_up > avg20 * 1.3 and self.delivery_pct >= 50:
            return 2
        if avg5_up > avg20 * 1.1 and self.delivery_pct >= 35:
            return 1
        return 0

    def _distribution_risk(self) -> int:
        """
        v5 NEW: Detect distribution — price rising while volume
        and delivery quality deteriorate over several sessions.
        Returns 0-2 penalty pts.
        """
        df = self.df
        if len(df) < 10:
            return 0

        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        close = df["Close"].squeeze().to_numpy(dtype=float)

        price_rising  = close[-1] > close[-5]
        vol_declining = float(np.mean(vol[-5:])) < float(np.mean(vol[-10:-5])) * 0.80

        # Classic distribution: price up, volume collapsing, low delivery
        if price_rising and vol_declining and self.delivery_pct < 25:
            return 2
        if price_rising and vol_declining:
            return 1
        return 0

    def _10d_shareholding(self) -> int:
        """Shareholding change proxy from NSE quarterly data. 0-3 pts."""
        try:
            symbol  = self.ticker.replace(".NS", "")
            url     = f"https://www.nseindia.com/api/corporates-shp?symbol={symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer":    "https://www.nseindia.com",
                "Accept":     "application/json",
            }
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                return 1
            data    = r.json()
            records = data.get("data", []) if isinstance(data, dict) else data
            if not records or len(records) < 2:
                return 1
            latest_fii = float(records[0].get("fii", 0) or 0)
            prior_fii  = float(records[1].get("fii", 0) or 0)
            change     = latest_fii - prior_fii
            if change >= 2.0:  return 3
            if change >= 0.5:  return 2
            if change >= -0.5: return 1
            return 0
        except Exception:
            return 1

    def _10e_bulk_deals(self) -> int:
        """NSE bulk/block deal detection. 0-3 pts."""
        try:
            symbol  = self.ticker.replace(".NS", "")
            url     = f"https://www.nseindia.com/api/bulk-deals?symbol={symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer":    "https://www.nseindia.com",
                "Accept":     "application/json",
            }
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                return 0
            data  = r.json()
            deals = data.get("data", []) if isinstance(data, dict) else []
            if not deals:
                return 0
            today       = datetime.today()
            recent_buys = 0
            for d in deals:
                date_str   = d.get("BD_DT_DATE", "") or d.get("date", "")
                order_type = (d.get("BD_TP_ATCHMT_SLTP", "") or "").upper()
                try:
                    deal_dt = datetime.strptime(date_str, "%d-%b-%Y")
                except Exception:
                    continue
                if (today - deal_dt).days <= 30 and "BUY" in order_type:
                    recent_buys += 1
            if recent_buys >= 3: return 3
            if recent_buys >= 1: return 2
            return 0
        except Exception:
            return 0
