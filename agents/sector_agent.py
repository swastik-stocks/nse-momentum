"""
NSE Momentum v5.0 - Sector Rotation Agent v5
Ranks 13 sectors by 3-month RS vs Nifty 500.
Score: 0-8 pts (increased from 7 in v5 rebalance).

v5 additions:
  - sector_breadth: % of sector stocks above EMA50
  - sector_failure_rate: % of recent T1 picks from sector that failed
  - Penalty applied when failure rate > 40%
  - Top sector identification updated for email display
"""
import logging
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "momentum_v5.db"

SECTORS = [
    "Financials", "IT", "Energy", "FMCG", "Auto",
    "Pharma", "Healthcare", "Metals", "Cement", "Telecom",
    "ConsumerDisc", "Industrials", "Chemicals",
]


class SectorAgent:
    def __init__(self, data_dict: dict):
        self.data           = data_dict
        self.sector_scores:     Dict[str, float] = {}
        self.sector_ranks:      Dict[str, int]   = {}
        self.sector_breadth:    Dict[str, float] = {}
        self.sector_failure_rate: Dict[str, float] = {}
        self._compute()
        self._compute_breadth()
        self._compute_failure_rates()

    def _compute(self):
        """Compute 13-week RS for each sector vs benchmark."""
        nifty500 = self.data.get("nifty500_data", pd.DataFrame())
        if nifty500.empty:
            nifty500 = self.data.get("nifty50_data", pd.DataFrame())

        benchmark_ret = 0.0
        if not nifty500.empty and len(nifty500) >= 65:
            b = nifty500["Close"].squeeze().to_numpy(dtype=float)
            benchmark_ret = float(b[-1] / b[-65] - 1) if b[-65] > 0 else 0.0

        stock_data    = self.data.get("stock_data", {})
        universe_meta = self.data.get("universe_meta", {})

        sector_rets: Dict[str, list] = {s: [] for s in SECTORS}
        for ticker, df in stock_data.items():
            sector = universe_meta.get(ticker, "Unknown")
            if sector not in sector_rets or df.empty or len(df) < 65:
                continue
            c      = df["Close"].squeeze().to_numpy(dtype=float)
            ret13w = float(c[-1] / c[-65] - 1) if c[-65] > 0 else 0.0
            sector_rets[sector].append(ret13w - benchmark_ret)

        sector_avg: Dict[str, float] = {}
        for s, rets in sector_rets.items():
            sector_avg[s] = float(np.mean(rets)) if rets else 0.0

        ranked = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
        self.sector_scores = dict(sector_avg)
        self.sector_ranks  = {s: rank + 1 for rank, (s, _) in enumerate(ranked)}

    def _compute_breadth(self):
        """
        v5: sector breadth = % of sector stocks above EMA50.
        Prevents high-ranking sectors carried by 1-2 heavyweights.
        """
        stock_data    = self.data.get("stock_data", {})
        universe_meta = self.data.get("universe_meta", {})

        sector_counts: Dict[str, list] = {s: [] for s in SECTORS}
        for ticker, df in stock_data.items():
            sector = universe_meta.get(ticker, "Unknown")
            if sector not in sector_counts or df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            price = float(close.iloc[-1])
            sector_counts[sector].append(1 if price > ema50 else 0)

        for s, vals in sector_counts.items():
            self.sector_breadth[s] = round(
                float(np.mean(vals)) * 100, 1
            ) if vals else 50.0

    def _compute_failure_rates(self):
        """
        v5: sector failure rate from trades_v4.
        Penalises sectors with recent high breakout failure rates.
        """
        try:
            if not DB_PATH.exists():
                return
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute("""
                SELECT sector,
                       COUNT(*) as total,
                       SUM(CASE WHEN r_multiple IS NOT NULL AND r_multiple < 0
                               THEN 1 ELSE 0 END) as failures
                FROM trades_v4
                WHERE entry_date >= date('now', '-60 days')
                  AND status IN ('CLOSED', 'open')
                GROUP BY sector
                HAVING total >= 3
            """)
            for row in cur.fetchall():
                sector, total, failures = row
                if sector and total > 0:
                    self.sector_failure_rate[sector] = round(
                        failures / total * 100, 1
                    )
            conn.close()
        except Exception as e:
            log.debug("SectorAgent failure rate error: %s", e)

    def score_for_sector(self, sector: str,
                         sector_ranks: Dict[str, int] = None) -> int:
        """
        Score 0-8 pts based on sector rank + breadth adjustment.
        v5: penalty if failure rate > 40%.
        """
        ranks = sector_ranks or self.sector_ranks
        rank  = ranks.get(sector, 7)

        # Base score from rank
        if rank == 1:   base = 8
        elif rank == 2: base = 7
        elif rank == 3: base = 6
        elif rank <= 5: base = 5
        elif rank <= 8: base = 3
        elif rank <= 10: base = 2
        elif rank <= 12: base = 1
        else:            base = 0

        # v5: breadth adjustment
        breadth = self.sector_breadth.get(sector, 50.0)
        if breadth >= 70:
            base = min(base + 1, 8)
        elif breadth < 35:
            base = max(base - 1, 0)

        # v5: failure rate penalty
        fail_rate = self.sector_failure_rate.get(sector, 0.0)
        if fail_rate > 40:
            base = max(base - 2, 0)

        return base

    def get_ranks(self) -> Dict[str, int]:
        return self.sector_ranks

    def get_top_sectors(self, n: int = 3) -> list:
        return sorted(self.sector_ranks.items(), key=lambda x: x[1])[:n]

    def get_sector_health(self, sector: str) -> dict:
        """Returns sector health summary for email display."""
        return {
            "rank":         self.sector_ranks.get(sector, 13),
            "breadth_pct":  self.sector_breadth.get(sector, 50.0),
            "failure_rate": self.sector_failure_rate.get(sector, 0.0),
        }
