"""
NSE Momentum v6.0 - Institutional Proxy Agent v6
10A: Delivery % accumulation score
10B: RVOL quality (up-volume vs down-volume)
10C: Volume contraction/expansion pattern
10D: Shareholding change proxy (NSE API)
10E: Bulk/block deal detection
10F: Promoter momentum-confirmation bonus (v6, factor-library item 4)

v5 additions:
  - Rolling delivery trend: rewards sustained rising delivery over 5 days
  - Distribution risk flag: price rising + delivery falling = warning
  - Score: 0-20 pts (unchanged ceiling, better internals)

v6 (2026-08-13): 10F added -- see _10f_promoter_momentum()'s own docstring.
Validated via validation/promoter_feedback_validate.py against 1,021
gate-cleared signals (2007-2026): promoters buying into an already-strong
RS position ("buying_winner") scored Avg R 1.54 vs pool 0.68, p~0.0 --
the single strongest result of any factor-library item tested in this
codebase so far. This INVERTS Shu (2009)'s original direction (which
predicted reversal, not confirmation) -- see the agent's docstring for
why that's an expected, explainable consequence of substituting promoter
data for the FII data Shu's paper used, not a red flag.
"""

import logging
import requests
from datetime import datetime

import numpy as np
import pandas as pd

from agents.promoter_feedback_agent import PromoterFeedbackAgent, MIN_QUARTERS_FOR_SIGNAL

log = logging.getLogger(__name__)

SHAREHOLDING_URL = "https://www.nseindia.com/api/corporate-share-holdings-master"
SHAREHOLDING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer":    "https://www.nseindia.com",
    "Accept":     "application/json",
}
QUARTERS_BACK = 8   # Shu's own 8-quarter lookback


class InstitutionalProxyAgent:
    def __init__(self, ticker: str, df: pd.DataFrame,
                 delivery_pct: float = 0.0, rs_percentile: float = 50.0):
        self.ticker        = ticker
        self.df            = df
        self.delivery_pct  = delivery_pct
        # v6 -- factor-library item 4. Defaults to the neutral midpoint
        # (50.0) for any caller that doesn't pass this (e.g. older tests),
        # in which case _10f_promoter_momentum() simply never fires (its
        # gate requires rs_percentile > 50 AND real promoter buying).
        self.rs_percentile = rs_percentile
        self._shareholding_records = None   # cached, fetched at most once per instance

    def evaluate(self) -> dict:
        s10a                = self._10a_delivery()
        s10b                = self._10b_rvol_quality()
        s10c                = self._10c_volume_pattern()
        s10d                = self._10d_shareholding()
        s10e                = self._10e_bulk_deals()
        s10f                = self._10f_promoter_momentum()
        rolling_bonus       = self._rolling_delivery_trend()
        distribution_penalty = self._distribution_risk()

        total = min(
            s10a + s10b + s10c + s10d + s10e + s10f
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
            "10f_promoter_momentum": s10f,
            "rolling_delivery_bonus":    rolling_bonus,
            "distribution_risk_penalty": distribution_penalty,
        }

    def _fetch_shareholding_records(self) -> list:
        """
        Fetched ONCE per instance, shared by _10d_shareholding (1-quarter
        delta) and _10f_promoter_momentum (8-quarter delta) -- avoids
        hitting NSE twice per stock per scan for the same underlying data.
        Returns [] on any failure; callers treat that as "no signal",
        never a crash.
        """
        if self._shareholding_records is not None:
            return self._shareholding_records
        try:
            symbol = self.ticker.replace(".NS", "")
            url    = f"{SHAREHOLDING_URL}?index=equities&symbol={symbol}"
            r = requests.get(url, headers=SHAREHOLDING_HEADERS, timeout=8)
            if r.status_code != 200:
                self._shareholding_records = []
                return []
            data = r.json()
            records = data.get("data", []) if isinstance(data, dict) else data
            self._shareholding_records = records or []
        except Exception:
            self._shareholding_records = []
        return self._shareholding_records

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
        """
        Shareholding change proxy from NSE quarterly data. 0-3 pts.

        [FIX 2026-08-13] The old endpoint (corporates-shp) is DEAD --
        confirmed 404 live. This silently broke the whole method (every
        call fell through to `except: return 1`, a flat neutral score,
        for an unknown period). Replaced with the real, live endpoint
        (corporate-share-holdings-master), confirmed working and returning
        22 quarters of history for RELIANCE in one call.

        DATA-FIT NOTE: the real endpoint has no FII/DII field under any
        name -- it only reports promoter% (pr_and_prgrp) and public%
        (public_val). The original code's "fii" key was reading a field
        that doesn't exist in this response shape either (it would have
        silently defaulted to 0 - 0 = no change even if the URL had
        worked). This fix uses PROMOTER holding delta as the accumulation
        proxy instead of FII delta -- a different signal (insider
        conviction, not foreign institutional flow), interim pending the
        proper factor-library item 4 validation (see
        FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md) which will test whether
        this proxy actually predicts anything before it's trusted beyond
        "at least it's live data now, not a silent flat 1".
        """
        try:
            records = self._fetch_shareholding_records()
            if not records or len(records) < 2:
                return 1
            # records[0] is the latest quarter (confirmed: NSE returns
            # these sorted descending by date).
            latest_promoter = float(records[0].get("pr_and_prgrp", 0) or 0)
            prior_promoter  = float(records[1].get("pr_and_prgrp", 0) or 0)
            change          = latest_promoter - prior_promoter
            if change >= 2.0:  return 3
            if change >= 0.5:  return 2
            if change >= -0.5: return 1
            return 0
        except Exception:
            return 1

    def _10f_promoter_momentum(self) -> int:
        """
        [v6, factor-library item 4] Promoter momentum-confirmation bonus.
        0-3 pts, ADDITIVE ONLY -- never a penalty, because the validation
        only found a significant edge in ONE quadrant (promoters buying
        into an already-strong RS position; see
        validation/promoter_feedback_validate.py's item4_quadrant_*
        results). The other three quadrants (buying a laggard, selling a
        winner, selling a laggard) were all statistically indistinguishable
        from the pool baseline (p=0.92-0.98) -- NOT evidence they're bad,
        just evidence we don't have grounds to score them differently from
        neutral. Scoring this as a bonus-only, single-quadrant signal is a
        deliberately conservative read of what was actually validated,
        not the full continuous MT formula PromoterFeedbackAgent exposes
        (that formula's generic +/-mapping across all four quadrants was
        an explicit "starting hypothesis, not yet validated" per its own
        docstring -- only the buying-into-strength quadrant cleared that
        bar here).
        """
        try:
            records = self._fetch_shareholding_records()
            if not records or len(records) < MIN_QUARTERS_FOR_SIGNAL + 1:
                return 0
            idx_trailing = min(QUARTERS_BACK, len(records) - 1)
            promoter_now      = float(records[0].get("pr_and_prgrp", 0) or 0)
            promoter_trailing = float(records[idx_trailing].get("pr_and_prgrp", 0) or 0)

            pf = PromoterFeedbackAgent(
                promoter_pct_now=promoter_now, promoter_pct_trailing=promoter_trailing,
                quarters_available=idx_trailing, rs_percentile=self.rs_percentile
            )
            if not pf.is_valid():
                return 0

            # The validated pattern specifically: promoters buying
            # (delta > 0) into a stock already in the top half of the RS
            # universe (rs_percentile > 50) -- N=130, Avg R 1.54, p~0.0.
            if pf.get_delta() > 0 and self.rs_percentile > 50:
                if pf.get_delta() >= 2.0 and self.rs_percentile >= 70:
                    return 3
                if pf.get_delta() >= 1.0:
                    return 2
                return 1
            return 0
        except Exception:
            return 0

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
