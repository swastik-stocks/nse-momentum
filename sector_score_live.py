"""
sector_score_live.py — P3-07 Stage 4

Live-scan version of the LOO sector_score validated in Stage 2/3
(breadth_edge_test.py / breadth_significance_test.py). Deliberately
matches that formula exactly:
    sector_score = leave-one-out mean of the sector's 5 price flags
    (above_sma20/50/100, above_rsi50, above_rs55), excluding the stock
    being scored.
The only differences from the backtest version are structural, not
formula changes: reads from price_history (live 2yr table, refreshed
daily by price_collector.py) instead of price_history_deep (Kaggle-
capped at 23 Feb 2026), and scores the CURRENT universe/sector
composition instead of a historical point-in-time one -- there is no
"point-in-time universe" concept for a live scan, today's universe IS
the live universe.

Any other deviation from the Stage 2 formula would create a train/live
mismatch -- the backtest proved THIS SPECIFIC formula has edge on
6,011 real trades; a live gate computing something merely similar would
not carry that same evidence.

Usage (as a module, called once per scan by orchestrator.py):
    from sector_score_live import LiveSectorBreadth
    lsb = LiveSectorBreadth()          # one DB pass, computes all sectors
    score = lsb.score("RELIANCE.NS")   # LOO sector_score for this ticker
    tier  = lsb.tier("RELIANCE.NS")    # "TOP" / "MID" / "BOTTOM" / None
"""

import logging
from collections import defaultdict

import numpy as np

from database.schema import get_connection
from sector_breadth import build_ticker_sector_mapping, _compute_stock_flags

log = logging.getLogger(__name__)

FLAG_KEYS = ("above_sma20", "above_sma50", "above_sma100", "above_rsi50", "above_rs55")

# Fixed thresholds from compute_sector_score_thresholds.py, run 05 Aug 2026
# against the validated breadth_tagged_trades dataset (33rd/66th percentile
# of sector_score across 6,011 backtested trades: N=6011, p33=63.50,
# p66=80.00). Baked in as constants rather than recomputed per-scan, so
# live gate behavior matches exactly what was backtested rather than
# drifting with whatever's in today's scan.
SECTOR_SCORE_BOTTOM_THRESHOLD = 63.5
SECTOR_SCORE_TOP_THRESHOLD = 80.0

# Bonus/penalty applied to StockResult.raw_score per tier (orchestrator.py
# integration -- see r.sector_breadth_bonus). NOTE: only the DIRECTION and
# RANKING of this effect was backtested (TOP vs BOTTOM tertile, p=0.0000,
# matched-sample-confirmed 05 Aug 2026) -- this specific +/-2 MAGNITUDE is
# a deliberately conservative starting choice (same scale as sector_agent.
# py's own +/-1 breadth adjustment and macd_score's 0-4 range), NOT itself
# backtested. Revisit once live results accumulate.
SECTOR_BREADTH_BONUS_MAGNITUDE = 2


class LiveSectorBreadth:
    """
    One DB pass at construction computes every sector's aggregate flag
    counts for the current universe. score()/tier() calls after that are
    pure in-memory LOO arithmetic -- cheap to call once per stock during
    a scan without re-querying the DB per ticker.
    """

    def __init__(self):
        self.ticker_to_sector, n_official, n_fallback = build_ticker_sector_mapping()
        if n_fallback:
            log.warning(f"  [LiveSectorBreadth] {n_fallback} ticker(s) using fallback sector "
                        f"(not in official NSE list)")

        conn = get_connection()
        self._ticker_flags = {}
        n_skipped = 0
        for ticker in self.ticker_to_sector:
            rows = conn.execute(
                "SELECT close FROM price_history WHERE ticker = ? ORDER BY date",
                (ticker,),
            ).fetchall()
            if not rows:
                n_skipped += 1
                continue
            close = np.array([r[0] for r in rows], dtype=float)
            flags = _compute_stock_flags(close)
            if flags is not None:
                self._ticker_flags[ticker] = flags
            else:
                n_skipped += 1
        conn.close()

        self._sector_sum = defaultdict(lambda: {k: 0 for k in FLAG_KEYS})
        self._sector_count = defaultdict(int)
        for ticker, flags in self._ticker_flags.items():
            sector = self.ticker_to_sector.get(ticker, "Unknown")
            self._sector_count[sector] += 1
            for k in FLAG_KEYS:
                if flags[k]:
                    self._sector_sum[sector][k] += 1

        # P3-07: load MCap weights for composite score
        self._mcap_weights = self._load_mcap_weights()

        log.info(f"  [LiveSectorBreadth] {len(self._ticker_flags)} tickers scored across "
                 f"{len(self._sector_count)} sectors ({n_skipped} skipped -- no/short history)")

    def score(self, ticker: str) -> float:
        """LOO sector_score for this ticker, or None if unavailable
        (no price history, or sector has 0 other members -- essentially
        never happens given the smallest observed sector has 3 members)."""
        flags = self._ticker_flags.get(ticker)
        if flags is None:
            return None
        sector = self.ticker_to_sector.get(ticker, "Unknown")
        c = self._sector_count.get(sector, 0)
        remaining = c - 1
        if remaining < 1:
            return None
        pct_sum = 0.0
        for k in FLAG_KEYS:
            own = 1 if flags[k] else 0
            loo_sum = self._sector_sum[sector][k] - own
            pct_sum += loo_sum / remaining * 100
        return round(pct_sum / 5, 1)

    def tier(self, ticker: str) -> str:
        """'TOP' / 'MID' / 'BOTTOM' per the backtested tertile thresholds,
        or None if score() itself returned None."""
        if SECTOR_SCORE_BOTTOM_THRESHOLD is None or SECTOR_SCORE_TOP_THRESHOLD is None:
            raise RuntimeError(
                "SECTOR_SCORE_BOTTOM_THRESHOLD / SECTOR_SCORE_TOP_THRESHOLD are still None -- "
                "run compute_sector_score_thresholds.py and fill these in before live use."
            )
        s = self.score(ticker)
        if s is None:
            return None
        if s >= SECTOR_SCORE_TOP_THRESHOLD:
            return "TOP"
        if s < SECTOR_SCORE_BOTTOM_THRESHOLD:
            return "BOTTOM"
        return "MID"

    def bonus(self, ticker: str) -> int:
        """raw_score contribution: +SECTOR_BREADTH_BONUS_MAGNITUDE for TOP,
        -SECTOR_BREADTH_BONUS_MAGNITUDE for BOTTOM, 0 for MID or unavailable."""
        t = self.tier(ticker)
        if t == "TOP":
            return SECTOR_BREADTH_BONUS_MAGNITUDE
        if t == "BOTTOM":
            return -SECTOR_BREADTH_BONUS_MAGNITUDE
        return 0


    def _load_mcap_weights(self) -> dict:
        """
        P3-07: {ticker_with_NS: mcap_cr} from Turso ticker_mcap table.
        Returns {} on failure — composite_score degrades gracefully to LOO score.
        """
        try:
            from turso_sync import get_client
            client = get_client()
            try:
                rs = client.execute("SELECT ticker, mcap_cr FROM ticker_mcap")
                result = {row[0]: float(row[1]) for row in rs.rows if row[1]}
                log.info(f"  [LiveSectorBreadth] MCap weights: {len(result)} tickers (P3-07)")
                return result
            except Exception as e:
                log.warning(f"  [LiveSectorBreadth] ticker_mcap unavailable ({e}) — "
                            f"composite_score = count-based LOO score")
                return {}
            finally:
                client.close()
        except Exception as e:
            log.warning(f"  [LiveSectorBreadth] Turso unreachable ({e})")
            return {}

    def composite_score(self, ticker: str) -> float:
        """
        P3-07 composite rotation score.
        Formula: 0.6 * count_loo_score + 0.4 * sector_mcap_sma50_pct

        0.6/0.4 weights: count-based LOO is validated (p=0.0000, 6011 trades);
        MCap weighting is new and unvalidated. Conservative split prevents MCap
        from overriding validated signal. Do not change without backtesting.

        Falls back to count-based LOO if MCap data is unavailable.
        Returns None if count-based score is also unavailable.

        IMPORTANT: Do not wire composite_bonus into raw_score until this has
        been backtested against breadth_tagged_trades.
        """
        loo = self.score(ticker)
        if loo is None:
            return None

        if not self._mcap_weights:
            return loo  # graceful fallback

        sector = self.ticker_to_sector.get(ticker, "Unknown")

        # LOO MCap-weighted SMA50 for this sector (exclude the stock being scored)
        mcap_total = 0.0
        mcap_above = 0.0
        for t, flags in self._ticker_flags.items():
            if self.ticker_to_sector.get(t) == sector and t != ticker:
                mcap = self._mcap_weights.get(t, 0.0)
                if mcap > 0:
                    mcap_total += mcap
                    if flags["above_sma50"]:
                        mcap_above += mcap

        if mcap_total > 0:
            mcap_pct = mcap_above / mcap_total * 100
            return round(0.6 * loo + 0.4 * mcap_pct, 1)

        return loo  # fallback: no MCap data for this sector

    def composite_tier(self, ticker: str) -> str:
        """
        P3-07 composite tier: TOP/MID/BOTTOM based on composite_score().
        Uses same thresholds as tier() — thresholds were derived on
        count-only scores and may need recalibration once composite scores
        have been backtested. Returns None if composite_score() is None.
        """
        s = self.composite_score(ticker)
        if s is None:
            return None
        if s >= SECTOR_SCORE_TOP_THRESHOLD:
            return "TOP"
        if s < SECTOR_SCORE_BOTTOM_THRESHOLD:
            return "BOTTOM"
        return "MID"


if __name__ == "__main__":
    # Standalone smoke test -- run this BEFORE wiring into orchestrator.py.
    # Cross-check a handful of known tickers' sector_breadth_score against
    # sector_breadth.py's already-trusted, already-published output for
    # the same sectors as a sanity check (won't match exactly -- that
    # script uses a straight, non-LOO mean; this uses LOO -- but should be
    # in the same ballpark, same direction).
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    lsb = LiveSectorBreadth()
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS", "INFY.NS"]
    print(f"\n{'TICKER':<15}{'SECTOR':<28}{'SCORE':>8}{'TIER':>8}{'BONUS':>7}")
    print("-" * 68)
    for t in sample_tickers:
        sector = lsb.ticker_to_sector.get(t, "?")
        score = lsb.score(t)
        tier = lsb.tier(t)
        bonus = lsb.bonus(t)
        score_str = f"{score:.1f}" if score is not None else "N/A"
        print(f"{t:<15}{sector:<28}{score_str:>8}{tier or 'N/A':>8}{bonus:>7}")
