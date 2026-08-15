"""
NSE Momentum vNext - Insider Silence/Traded Agent   [PROPOSED -- NOT WIRED
INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase — must clear
validation/insider_silence_validate.py's permutation-significance test
before this touches live scoring.

WHY THIS EXISTS
    Factor-library item 5 (Ma 2013, "Momentum and Insider Trading") — see
    FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md. Ma's core finding: among
    momentum winners, stocks with ANY insider trading activity (buy OR
    sell — direction didn't matter in the original paper) in the trailing
    6 months kept earning positive returns, while "silent" winners (zero
    insider activity) reversed hard. The mechanism: insiders avoid
    trading (especially selling) when they're sitting on undisclosed bad
    news, due to litigation exposure — so silence itself is the tell, not
    the direction of any trade that does happen.

DATA SOURCE: insider_transactions table, bulk-loaded once via
    load_shareholding_insider_history.py from NSE's corporates-pit
    endpoint (Regulation 7(2) SAST/PIT disclosures) — 416/500 tickers,
    2015-2026. See that script's docstring for why this was a one-time
    bulk pull, not ongoing infrastructure.

HONEST SCOPE NOTE: Ma's paper found this effect specifically at a
    multi-year horizon (winners diverged sharply in "year 2+" of the
    holding period). This codebase's standard validation window
    (FORWARD_BARS=20, ~1 month) is much shorter than that. The validation
    harness for this agent tests BOTH the standard ~1-month window (for
    comparability with items 1-4) AND a ~1-year window (closer to what Ma
    actually found) — results should be read with that horizon mismatch
    in mind, not assumed to transfer from one window to the other.

INTERFACE
    Mirrors every other prototype in this repo: dual .passes_gate() /
    .score_bonus() interface, fails open on insufficient/no data.
"""

import logging
from datetime import datetime, timedelta

import numpy as np

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 180   # Ma's own trailing 6-month window


class InsiderSilenceAgent:
    """
    Given a ticker's known insider transactions and an as-of date, answers
    Ma (2013)'s core question: was there ANY insider trading activity
    (buy or sell) in the trailing 6 months, or silence?
    """

    def __init__(self, transactions: list, as_of: str, lookback_days: int = LOOKBACK_DAYS):
        """
        transactions: list of dicts with at least 'disclosure_dt' (ISO
            'YYYY-MM-DD' string) and 'transaction_type' ('Buy'/'Sell'),
            optionally 'buy_value'/'sell_value' for the net-flow diagnostic.
            Does NOT need to be pre-filtered to the window -- this class
            does that itself, same convention as
            FiftyTwoWeekHighAgent/RandomnessAgent taking raw windows.
        as_of: 'YYYY-MM-DD' string, the signal/checkpoint date.
        """
        self.transactions = transactions or []
        self.as_of = as_of
        self.lookback_days = lookback_days

        self._n_buys = 0
        self._n_sells = 0
        self._net_value = 0.0
        self._traded = False

        self._compute()

    def _compute(self):
        try:
            as_of_dt = datetime.strptime(self.as_of, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return
        window_start = as_of_dt - timedelta(days=self.lookback_days)

        for txn in self.transactions:
            try:
                dt = datetime.strptime(txn["disclosure_dt"], "%Y-%m-%d").date()
            except (KeyError, ValueError, TypeError):
                continue
            if not (window_start <= dt <= as_of_dt):
                continue

            ttype = (txn.get("transaction_type") or "").strip().lower()
            if ttype == "buy":
                self._n_buys += 1
                self._net_value += float(txn.get("buy_value", 0) or 0)
            elif ttype == "sell":
                self._n_sells += 1
                self._net_value -= float(txn.get("sell_value", 0) or 0)

        self._traded = (self._n_buys + self._n_sells) > 0

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC -- integration style not yet confirmed.]
        True (pass) when there WAS insider activity in the trailing 6
        months ("traded" -- Ma's non-reversal-prone case), False on
        silence. Deliberately does NOT fail open on zero transactions --
        that IS the signal this agent measures, not a data-quality gap.
        Only fails open (True) if `as_of` itself couldn't be parsed
        (a genuine data problem, distinct from "no transactions found").
        """
        try:
            datetime.strptime(self.as_of, "%Y-%m-%d")
        except (ValueError, TypeError):
            return True
        return self._traded

    def score_bonus(self) -> float:
        """
        Continuous alternative to the hard gate, for the validation run to
        compare. Silence maps to a penalty (Ma's flagged reversal risk);
        traded maps to a small bonus, scaled modestly by net insider flow
        direction/magnitude where available. Clamped to the same
        +/-2..+3 range every other prototype in this repo uses.
        """
        try:
            datetime.strptime(self.as_of, "%Y-%m-%d")
        except (ValueError, TypeError):
            return 0.0
        if not self._traded:
            return -1.5   # Ma's flagged case: silence among winners
        raw = 1.0 + np.sign(self._net_value) * min(abs(self._net_value) / 1e7, 1.0)
        return round(float(np.clip(raw, -2.0, 3.0)), 2)

    def get_traded(self) -> bool:
        return self._traded

    def get_n_buys(self) -> int:
        return self._n_buys

    def get_n_sells(self) -> int:
        return self._n_sells

    def get_net_value(self) -> float:
        return self._net_value
