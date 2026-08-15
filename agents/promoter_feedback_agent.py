"""
NSE Momentum vNext - Promoter Feedback-Trading Agent   [PROPOSED -- NOT
WIRED INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase — must clear
validation/promoter_feedback_validate.py's permutation-significance test
before this touches live scoring.

WHY THIS EXISTS
    Factor-library item 4 (Shu 2009, MT-measure exit-timing overlay) —
    see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md. Shu's original measure:
    institutional-ownership delta over trailing 8 quarters, weighted by
    the stock's past-return decile rank. The thesis is "positive feedback
    trading" — institutions piling into a stock BECAUSE it's already a
    momentum winner (not because of independent fundamental conviction)
    historically precedes sharper reversals than ownership growth that's
    NOT correlated with recent momentum.

DATA-FIT DEVIATION FROM THE ORIGINAL PAPER (read before trusting this)
    Shu's measure used FII/institutional ownership. NSE's live shareholding
    endpoint (see agents/institutional_proxy_agent.py::_10d_shareholding's
    2026-08-13 fix docstring) has no FII field at all — only promoter% and
    public%. This agent uses PROMOTER holding delta instead.

    This is not a neutral substitution. Shu's mechanism specifically
    requires the buyer to lack an information edge (institutions chasing
    performance without inside knowledge is what makes their buying an
    uninformative, reversal-prone signal). Promoters are the opposite of
    that — they're insiders, presumed to HAVE an information edge. It is
    entirely possible promoter buying into strength means something
    different (or even opposite) from what Shu found for institutions.
    This agent and its validation harness are built to test that
    empirically, not to assume Shu's direction transfers. Given 2 of the
    3 factor-library items already validated in this codebase (Randomness
    gate, 52-week-high proximity) inverted their source papers' direction
    once tested here, no particular outcome should be assumed going in.

FORMULA
    MT = promoter_delta_pct * ((rs_percentile - 50) / 50)

    promoter_delta_pct: change in promoter holding % over the trailing
        window (default ~8 quarters / 2 years, matching Shu's own
        lookback, now feasible since NSE's endpoint returns ~5 years of
        quarterly history in one call — see load_shareholding_insider_history.py).
    rs_percentile: the stock's existing universe-wide RS percentile
        (0-100, same figure RSAgent already computes) as the past-return-
        decile proxy, rescaled to -1..+1 around the 50th-percentile
        midpoint.

    Positive MT: promoter buying (delta > 0) concentrated in an
        already-strong momentum name (rs_percentile > 50) — Shu's
        "feedback trading" signature — OR promoter selling (delta < 0) in
        an already-weak name. Negative MT: promoter buying a laggard, or
        selling a leader — the "contrarian to recent price action" cases.
        Which direction (if either) actually predicts worse/better forward
        returns is exactly what the validation harness tests; this agent
        does not presuppose an answer.

INTERFACE
    Mirrors every other prototype in this repo: dual .passes_gate() /
    .score_bonus() interface, fails open on insufficient data.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)

# Minimum number of quarters of promoter-holding history required before
# trusting a delta at all -- a 1-quarter delta is noisy; Shu's own paper
# used 8 quarters, so anything meaningfully short of that is a materially
# different (weaker) measure, not just "less precise."
MIN_QUARTERS_FOR_SIGNAL = 4

# Gate threshold on |MT| -- starting hypothesis for the validation harness
# to confirm/recalibrate, not backtested yet.
GATE_MIN_ABS_MT = 1.0


class PromoterFeedbackAgent:
    """
    Combines promoter-holding delta with the stock's existing RS
    percentile into Shu (2009)'s MT-measure, adapted to promoter data
    (see module docstring for why that's a real deviation, not a free
    substitution).
    """

    def __init__(self, promoter_pct_now: float, promoter_pct_trailing: float,
                 quarters_available: int, rs_percentile: float):
        """
        promoter_pct_now: most recent promoter holding %.
        promoter_pct_trailing: promoter holding % ~8 quarters (or
            whatever `quarters_available` reflects) earlier.
        quarters_available: how many quarters actually separate the two
            readings -- used to fail open when this is too short to trust.
        rs_percentile: 0-100, same convention as RSAgent.get_percentile().
        """
        self.promoter_pct_now = promoter_pct_now
        self.promoter_pct_trailing = promoter_pct_trailing
        self.quarters_available = quarters_available
        self.rs_percentile = rs_percentile

        self._delta = 0.0
        self._mt = 0.0
        self._valid = False

        self._compute()

    def _compute(self):
        if self.quarters_available < MIN_QUARTERS_FOR_SIGNAL:
            return
        if self.promoter_pct_now is None or self.promoter_pct_trailing is None:
            return
        if self.rs_percentile is None:
            return

        self._delta = round(self.promoter_pct_now - self.promoter_pct_trailing, 3)
        momentum_weight = (self.rs_percentile - 50.0) / 50.0   # -1..+1
        self._mt = round(self._delta * momentum_weight, 4)
        self._valid = True

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC -- integration style AND direction not yet
        confirmed.] True (pass) when |MT| is below the gate threshold
        (i.e. NOT showing Shu's feedback-trading signature) OR when there
        isn't enough quarterly history -- fails open. NOTE: unlike other
        prototypes in this repo, whether "pass" should mean high or low
        MT is itself an open question the validation harness answers;
        this default assumes Shu's original direction (high |MT| = bad)
        purely as a starting convention to test, not a conclusion.
        """
        if not self._valid:
            return True
        return abs(self._mt) < GATE_MIN_ABS_MT

    def score_bonus(self) -> float:
        """
        Continuous alternative to the hard gate, for the validation run to
        compare against passes_gate(). Negative MT (buying-a-laggard or
        selling-a-leader -- the "not chasing" case under Shu's framing)
        maps to a small positive bonus; positive MT maps to a small
        penalty -- again, a starting convention for the validation harness
        to confirm, invert, or reject, not an assumed-correct mapping.
        Clamped to the same +/-2..+3 range every other prototype uses.
        """
        if not self._valid:
            return 0.0
        raw = -1.0 * self._mt
        return round(float(np.clip(raw, -2.0, 3.0)), 2)

    def get_delta(self) -> float:
        return self._delta

    def get_mt(self) -> float:
        return self._mt

    def is_valid(self) -> bool:
        return self._valid
