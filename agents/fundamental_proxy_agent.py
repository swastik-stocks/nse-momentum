"""
NSE Momentum v4.3 — Fundamental Proxy Agent
Agent 9A: NSE announcements keyword scoring (corporate catalyst)
Agent 9B: Earnings calendar proximity (pre-result risk/opportunity)
Agent 9C: Price-led fundamental proxy (RS + delivery + sector RS)
Score: 0-20 pts combined.
"""

import logging
import requests
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Positive catalyst keywords
POSITIVE_KEYWORDS = [
    "order", "contract", "win", "award", "capacity", "expansion", "merger",
    "acquisition", "demerger", "buyback", "dividend", "bonus", "capex",
    "partnership", "joint venture", "approval", "license", "fda", "anda",
    "launch", "record date", "rights issue", "results beat", "upgrade",
]
NEGATIVE_KEYWORDS = [
    "penalty", "fraud", "scam", "default", "recall", "ban", "warning letter",
    "sebi notice", "court", "arbitration", "downgrade", "loss",
]


class FundamentalProxyAgent:
    def __init__(self, ticker: str, df: pd.DataFrame,
                 delivery_pct: float = 0.0,
                 rs_percentile: float = 50.0,
                 sector_rank: int = 7):
        self.ticker       = ticker
        self.df           = df
        self.delivery_pct = delivery_pct
        self.rs_pct       = rs_percentile
        self.sector_rank  = sector_rank

    def evaluate(self) -> dict:
        score_9a = self._agent_9a()
        score_9b = self._agent_9b()
        score_9c = self._agent_9c()
        total = min(score_9a + score_9b + score_9c, 20)
        return {
            "fundamental_proxy_score": total,
            "9a_catalyst": score_9a,
            "9b_earnings": score_9b,
            "9c_price_led": score_9c,
        }

    def _agent_9a(self) -> int:
        """NSE corporate announcements → catalyst score 0-8."""
        symbol = self.ticker.replace(".NS", "")
        try:
            url = f"https://www.nseindia.com/api/corp-info?symbol={symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com",
            }
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                return 3  # neutral default
            data = r.json()
            text = json.dumps(data).lower()
            pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
            neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
            score = max(0, min(8, 4 + pos - neg * 2))
            return score
        except Exception:
            return 3  # neutral on timeout

    def _agent_9b(self) -> int:
        """Earnings calendar proximity → 0-4 pts. Penalty if within 7 days."""
        try:
            symbol = self.ticker.replace(".NS", "")
            url = f"https://www.nseindia.com/api/event-calendar?symbol={symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.nseindia.com",
            }
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                return 2
            data = r.json()
            today = datetime.today()
            for event in (data if isinstance(data, list) else []):
                date_str = event.get("date", "") or event.get("bm_date", "")
                desc = (event.get("purpose", "") or "").lower()
                if not date_str:
                    continue
                try:
                    ev_date = datetime.strptime(date_str, "%d-%b-%Y")
                except Exception:
                    continue
                days_away = (ev_date - today).days
                if "quarterly" in desc or "result" in desc or "q1" in desc or "q2" in desc or "q3" in desc or "q4" in desc:
                    if 0 <= days_away <= 7:
                        return 0  # pre-result risk: skip
                    elif 8 <= days_away <= 30:
                        return 2  # approaching — use caution
                    return 4  # safe window
        except Exception:
            pass
        return 2

    def _agent_9c(self) -> int:
        """Price-led proxy: RS + delivery + sector rank → 0-8 pts."""
        pts = 0
        # RS percentile proxy (higher = better fundamentals likely)
        if self.rs_pct >= 80: pts += 4
        elif self.rs_pct >= 70: pts += 3
        elif self.rs_pct >= 60: pts += 2
        elif self.rs_pct >= 50: pts += 1

        # Delivery % as institutional accumulation proxy
        if self.delivery_pct >= 60: pts += 2
        elif self.delivery_pct >= 40: pts += 1

        # Sector rank (top sectors tend to have better earnings momentum)
        if self.sector_rank <= 3: pts += 2
        elif self.sector_rank <= 6: pts += 1

        return min(pts, 8)
