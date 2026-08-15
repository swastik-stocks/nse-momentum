"""
NSE Momentum v6 — ML Momentum Model Training/Retraining

Fits the production MLMomentumModel (agents/ml_momentum_agent.py, factor-
library backlog item, Beaudan & He 2019) on ALL gate-cleared signals
currently in ml_momentum_validate_progress (built by
validation/ml_momentum_validate.py --generate) and persists the fitted
weights via save_model() for orchestrator.py's live scan to load.

WHEN TO RUN
    - Once now, to seed the first production model (this script's own
      first run).
    - Periodically thereafter, whenever validation/ml_momentum_validate.py
      --generate has been re-run against a meaningfully larger/more recent
      signal set. There's no fixed calendar cadence enforced here -- see
      agents/ml_momentum_agent.py's module docstring for why an autonomous
      retraining trigger (the paper's own Section 3.2.3 approach) wasn't
      implemented for this codebase's shorter history.
    - This is a SEPARATE step from validation/ml_momentum_validate.py
      --evaluate on purpose: --evaluate measures out-of-sample performance
      via walk-forward splits (never lets a window's model see its own
      test data); this script fits ONE final model on ALL available data
      for live use, which is the correct thing to do for a production
      model (use every available data point) but would be look-ahead bias
      if used for validation, hence the separation.

Usage:
    python train_ml_momentum_model.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from agents.ml_momentum_agent import MLMomentumModel, save_model
from validation.ml_momentum_validate import _load_all_progress

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def main():
    log.info("Loading gate-cleared signals from ml_momentum_validate_progress...")
    all_progress = _load_all_progress()
    rows = []
    for res in all_progress:
        for o in res["results"]:
            if o.get("features") is not None:
                rows.append(o)

    if len(rows) < 200:
        raise RuntimeError(
            f"Only {len(rows)} usable signals — run "
            f"'python validation/ml_momentum_validate.py --generate --fresh' first."
        )

    rows.sort(key=lambda o: o["date"])
    X = np.array([o["features"] for o in rows])
    y = np.array([o["y_h20"] for o in rows])
    trained_through_date = rows[-1]["date"]

    log.info(f"Fitting on {len(rows)} signals, {trained_through_date} most recent date...")
    model = MLMomentumModel()
    model.fit(X, y)

    save_model(model, trained_through_date, n_training_signals=len(rows))
    log.info(f"Model persisted: trained on {len(rows)} signals through {trained_through_date}.")
    log.info(f"theta[:5] = {model.theta[:5]}")


if __name__ == "__main__":
    main()
