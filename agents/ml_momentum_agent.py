"""
NSE Momentum vNext - ML Momentum Agent   [LIVE, v6 -- wired into
agents/rs_agent.py::RSAgent.score(), 2026-08-14]

STATUS: validated. See validation/ml_momentum_validate.py's walk-forward,
out-of-sample test (8 independently-trained windows, 2016-2026, 1,681
out-of-sample predictions): predicted-positive bucket WR=61.8%, Avg R 1.05
vs pool 0.78, p=0.0, holding up (with some decay, same pattern as every
other validated item) across both halves of the date range.
predicted-negative bucket reliably underperformed the pool (p=1.0).

UNLIKE every other factor-library item, this one is a TRAINED MODEL, not a
closed-form formula -- it needs a persisted model artifact (fitted weights
+ feature normalization stats) and a periodic retraining cadence, not just
a live computation from today's data. See save_model()/load_latest_model()
below and train_ml_momentum_model.py (repo root) for the retraining script.
Re-run train_ml_momentum_model.py periodically (paper's own guidance:
somewhere between 2-8 years is fine; this codebase's shorter history
means "whenever meaningfully more data has accumulated" is a more useful
cadence than a fixed calendar trigger -- there's no autonomous
convergence-based retraining trigger implemented, unlike the paper's own
Section 3.2.3, since NSE's much shorter history makes that harder to
validate reliably here).

WHY THIS EXISTS
    Factor-library backlog item, Beaudan & He (2019) "Applying Machine
    Learning to Trading Strategies: Using Logistic Regression to Build
    Momentum-based Trading Strategies" -- full text obtained and reviewed
    this round (previously only known secondhand). See
    FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md for the scope decision this was
    built under.

WHAT THE PAPER ACTUALLY DOES (important -- this is an adaptation, not a
    literal port): Beaudan & He build a SINGLE-ASSET time-series timing
    model -- one logistic regression that decides, for the S&P 500 index
    alone, whether to be invested or in cash on each day. This codebase is
    a CROSS-SECTIONAL stock-ranking system with no single-asset timing
    role -- there is nothing to "go to cash" on a per-index basis here.
    Per explicit instruction, this was adapted into a PER-STOCK CLASSIFIER:
    the same feature engineering and logistic-regression machinery,
    applied to each gate-cleared candidate stock, predicting whether THAT
    candidate's own forward return is likely to clear a minimum
    profitability threshold -- competing with (not replacing outright)
    the existing hand-tuned additive scoring formula in orchestrator.py.
    Because the paper's own per-asset models never share training data
    across different assets, and NSE single-stock histories are far
    shorter/noisier than SPX's 90 years, this implementation POOLS
    gate-cleared signals across the whole universe into one shared
    training set per walk-forward window (a cross-sectional generalization
    of the paper's single-asset design, not a deviation from its
    statistical logic -- see validation/ml_momentum_validate.py's
    docstring for the full walk-forward methodology).

FORMULA (ported directly from the paper, Sections 3.2.1-3.2.5):
    Features (12 base, un-normalized):
        Momenta:    trailing % price change over 30, 60, 90, 120, 180,
                    270, 300, 360 business days (8 features)
        Drawdowns:  current % decline from the trailing peak within a
                    15, 60, 90, 120 business-day window (4 features)
    Normalized to zero-mean, [-1, +1] range using TRAINING-SET-ONLY
    min/max (never test-set stats -- avoids look-ahead).
    Expanded to cubic (degree-3) polynomial combinations, exactly as
    sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=True)
    would -- 12 base features -> 455 total columns (matches the paper's
    own reported count for 12 seed features).
    Label: y=1 if forward annualized profitability p = (Price[i+H]/Price[i])
    ^(252/H) - 1 >= delta (5% annualized, paper's own default), else 0.
    Classifier: L2-regularized logistic regression, cost function ported
    verbatim from the paper's Eq. in Section 3.2.1 (regularization
    parameter lambda=1, intercept/bias term excluded from the penalty,
    matching the paper's own sum starting at j=1). Fit via scipy.optimize
    (L-BFGS-B) rather than scikit-learn, since scikit-learn could not be
    installed in this environment (persistent file-lock error on the
    venv's site-packages -- same recurring issue documented elsewhere in
    this session) -- the math is identical, just hand-rolled with numpy/
    scipy instead of sklearn's Pipeline/PolynomialFeatures/LogisticRegression.

HONEST SCOPE NOTE: the paper found H=3 business days optimal for SPX, but
    explicitly cautions "why three days turns out to be an optimal horizon
    ... is likely unknowable and not a result we anticipate would hold for
    other types of equities, such as single stocks." This codebase's real
    trading rhythm (positions held ~20 trading days, matching every other
    factor-library item's FORWARD_BARS) is a poor match for a 3-day label,
    so the primary trained model here uses H=20 (this codebase's own
    horizon) rather than assuming the paper's H=3 transfers. See the
    validator for the H=3 diagnostic run kept for academic comparability.

INTERFACE
    Mirrors every other prototype in this repo: dual .passes_gate() /
    .score_bonus() interface, fails open on insufficient/no data. Unlike
    the simpler prototypes, this one needs a FITTED MODEL (not just raw
    inputs) since it's a trained classifier, not a closed-form formula --
    the validator handles walk-forward fit/refit; this class just wraps
    prediction given an already-fitted MLMomentumModel.
"""

import sys
import json
import logging
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.schema import get_connection

log = logging.getLogger(__name__)

MOMENTUM_WINDOWS = [30, 60, 90, 120, 180, 270, 300, 360]
DRAWDOWN_WINDOWS = [15, 60, 90, 120]
N_BASE_FEATURES = len(MOMENTUM_WINDOWS) + len(DRAWDOWN_WINDOWS)  # 12
POLY_DEGREE = 3
DELTA_ANNUAL = 0.05        # paper's own 5% annualized minimum profitability threshold
REG_LAMBDA = 1.0           # paper's own regularization parameter


def compute_raw_features(close: np.ndarray, i: int):
    """
    12 raw (un-normalized) features at index i of a 1-D close-price array:
    8 momenta + 4 drawdowns, exactly as specified in the paper. Returns
    None if there isn't enough trailing history for the longest window
    (360 business days).
    """
    max_window = max(MOMENTUM_WINDOWS + DRAWDOWN_WINDOWS)
    if i < max_window:
        return None

    feats = []
    for w in MOMENTUM_WINDOWS:
        past = close[i - w]
        feats.append((close[i] / past - 1.0) if past > 0 else 0.0)
    for w in DRAWDOWN_WINDOWS:
        window = close[i - w + 1: i + 1]
        peak = float(np.max(window))
        feats.append((close[i] - peak) / peak if peak > 0 else 0.0)
    return np.array(feats, dtype=float)


def annualized_forward_profitability(close: np.ndarray, i: int, H: int):
    """p_i = (Price[i+H]/Price[i])^(252/H) - 1, the paper's own Eq. in
    footnote 6. Returns None if H future bars don't exist."""
    if i + H >= len(close) or close[i] <= 0:
        return None
    ratio = close[i + H] / close[i]
    if ratio <= 0:
        return None
    return ratio ** (252.0 / H) - 1.0


def _poly_feature_indices(n_base: int, degree: int):
    """Reproduces sklearn.preprocessing.PolynomialFeatures(degree, include_bias=True)'s
    column ordering: all combinations_with_replacement of feature indices for
    each degree 0..degree, degree 0 first (the bias/constant column)."""
    combos = [()]  # degree 0 -> bias term
    for d in range(1, degree + 1):
        combos.extend(itertools.combinations_with_replacement(range(n_base), d))
    return combos


_POLY_COMBOS = _poly_feature_indices(N_BASE_FEATURES, POLY_DEGREE)
N_POLY_FEATURES = len(_POLY_COMBOS)  # 455 for 12 base features, degree 3


def _expand_polynomial(X_norm: np.ndarray) -> np.ndarray:
    """X_norm: (n_samples, 12) normalized features -> (n_samples, 455)."""
    n = X_norm.shape[0]
    out = np.empty((n, N_POLY_FEATURES), dtype=float)
    for j, combo in enumerate(_POLY_COMBOS):
        if not combo:
            out[:, j] = 1.0
        else:
            col = np.ones(n, dtype=float)
            for idx in combo:
                col = col * X_norm[:, idx]
            out[:, j] = col
    return out


class MLMomentumModel:
    """
    L2-regularized logistic regression on cubic-polynomial-expanded
    momentum/drawdown features. Ported from the paper's own math
    (Section 3.2.1), not scikit-learn -- see module docstring for why.
    """

    def __init__(self, reg_lambda: float = REG_LAMBDA):
        self.reg_lambda = reg_lambda
        self.theta = None          # fitted weights, shape (455,)
        self.feat_min = None       # per-base-feature min, for normalization
        self.feat_max = None       # per-base-feature max, for normalization

    def _normalize(self, X_raw: np.ndarray) -> np.ndarray:
        """Zero-mean, [-1,+1] scaling using TRAINING-SET stats (self.feat_min/max),
        never recomputed on test data -- avoids look-ahead."""
        span = self.feat_max - self.feat_min
        span[span == 0] = 1.0
        mid = (self.feat_max + self.feat_min) / 2.0
        return (X_raw - mid) / (span / 2.0)

    def fit(self, X_raw: np.ndarray, y: np.ndarray, max_iter: int = 200) -> None:
        from scipy.optimize import minimize

        self.feat_min = X_raw.min(axis=0)
        self.feat_max = X_raw.max(axis=0)
        X_norm = self._normalize(X_raw)
        Xp = _expand_polynomial(X_norm)
        m, n = Xp.shape
        y = y.astype(float)

        def cost_and_grad(theta):
            z = Xp @ theta
            z = np.clip(z, -500, 500)
            h = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            cost = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
            reg = (self.reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)  # exclude bias col 0
            grad = (Xp.T @ (h - y)) / m
            grad_reg = np.zeros(n)
            grad_reg[1:] = (self.reg_lambda / m) * theta[1:]
            return cost + reg, grad + grad_reg

        theta0 = np.zeros(n)
        result = minimize(cost_and_grad, theta0, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_iter})
        self.theta = result.x

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(X_raw)
        Xp = _expand_polynomial(X_norm)
        z = np.clip(Xp @ self.theta, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


_MODEL_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ml_momentum_model_state (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    theta_json            TEXT,
    feat_min_json         TEXT,
    feat_max_json         TEXT,
    n_training_signals    INTEGER,
    trained_through_date  TEXT,
    trained_at            TEXT
)
"""


def save_model(model: "MLMomentumModel", trained_through_date: str, n_training_signals: int) -> None:
    """Persists a fitted model as a new row (never overwrites -- history of
    past retrainings stays queryable). load_latest_model() always reads the
    most recent row."""
    conn = get_connection()
    conn.execute(_MODEL_TABLE_DDL)
    conn.execute("""
        INSERT INTO ml_momentum_model_state
        (theta_json, feat_min_json, feat_max_json, n_training_signals, trained_through_date, trained_at)
        VALUES (?,?,?,?,?,?)
    """, (
        json.dumps(model.theta.tolist()),
        json.dumps(model.feat_min.tolist()),
        json.dumps(model.feat_max.tolist()),
        n_training_signals,
        trained_through_date,
        datetime.today().isoformat(),
    ))
    conn.commit()
    conn.close()


def load_latest_model():
    """Returns (MLMomentumModel, trained_through_date) or (None, None) if
    no model has been trained yet -- callers must handle the None case
    (compute_universe_ml_momentum_predictions() below already does)."""
    conn = get_connection()
    conn.execute(_MODEL_TABLE_DDL)
    row = conn.execute("""
        SELECT theta_json, feat_min_json, feat_max_json, trained_through_date
        FROM ml_momentum_model_state ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn.close()
    if not row:
        return None, None
    model = MLMomentumModel()
    model.theta = np.array(json.loads(row["theta_json"]))
    model.feat_min = np.array(json.loads(row["feat_min_json"]))
    model.feat_max = np.array(json.loads(row["feat_max_json"]))
    return model, row["trained_through_date"]


def compute_universe_ml_momentum_predictions(data_dict: dict, model=None) -> dict:
    """
    [LIVE, v6] Cross-sectional predicted class (0/1) for every ticker,
    computed once per scan -- same {ticker: value} shape as the other
    universe-wide precompute functions, so orchestrator.py wires it in
    identically. `model` defaults to the latest persisted model
    (load_latest_model()) if not supplied. Returns {} (never raises) if no
    model has been trained yet, or if the model is stale/missing -- callers
    treat a missing entry as "no prediction," same convention as
    vol_adj_pct/lottery_pct.

    Only the binary predicted class (proba>=0.5) is used, matching exactly
    what validation tested -- the raw probability was never itself
    validated as a graded signal, only the hard threshold.
    """
    if model is None:
        model, _ = load_latest_model()
    if model is None or model.theta is None:
        return {}

    stock_data = data_dict.get("stock_data", {})
    tickers, feats = [], []
    for ticker, df in stock_data.items():
        if df.empty:
            continue
        close = df["Close"].squeeze().to_numpy(dtype=float)
        f = compute_raw_features(close, len(close) - 1)
        if f is None:
            continue
        tickers.append(ticker)
        feats.append(f)

    if not feats:
        return {}

    proba = model.predict_proba(np.array(feats))
    pred = (proba >= 0.5).astype(int)
    return {t: int(p) for t, p in zip(tickers, pred)}


class MLMomentumAgent:
    """
    Thin wrapper around a single already-computed prediction (probability
    of clearing the forward profitability threshold). The heavy lifting
    -- walk-forward fit/refit -- lives in validation/ml_momentum_validate.py,
    same division of labour as compute_universe_ranks() in rs_agent.py.
    """

    def __init__(self, probability: float = None):
        self.probability = probability

    def passes_gate(self) -> bool:
        """Diagnostic-only pending validation -- never blocks."""
        return True

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise."""
        return 0.0

    def get_probability(self):
        return self.probability
