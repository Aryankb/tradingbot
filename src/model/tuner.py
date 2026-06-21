"""Optuna hyperparameter search for the XGBoost classifier.

Objective: maximise mean precision of BUY and SELL *after applying the
ENTRY_PROB_THRESHOLD*.  This directly optimises for the precision the bot
will see at runtime, not the precision across all low-confidence predictions
that the bot would never act on.

A minimum-signal guard (at least 0.5% of validation rows must be actionable)
prevents the trivial solution of never predicting BUY/SELL.
"""

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import precision_score
import logging

import config
from src.model.trainer import _sample_weights, apply_threshold

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_CANDLES_PER_DAY = int(24 * 60 / config.CANDLE_INTERVAL_MINUTES)

# Minimum fraction of validation rows that must be actionable signals.
# 0.005 (0.5%) was too permissive — Optuna found near-trivial solutions with
# ~20 signals on 3600 rows giving inflated precision estimates.
# 0.05 (5%) = 180 signals minimum on the val set, forcing reliable estimates.
_MIN_SIGNAL_FRAC = 0.05


def _objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    """Return mean(precision_BUY, precision_SELL) at ENTRY_PROB_THRESHOLD."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "max_depth":          trial.suggest_int("max_depth", 3, 10),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 2.0),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=_sample_weights(y_tr),
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    proba  = model.predict_proba(X_va)
    y_pred = apply_threshold(proba, config.ENTRY_PROB_THRESHOLD)

    n_signals = int(np.sum(y_pred > 0))
    if n_signals < max(1, int(_MIN_SIGNAL_FRAC * len(y_va))):
        return 0.0

    prec_buy  = precision_score(y_va, y_pred, labels=[1], average="macro", zero_division=0)
    prec_sell = precision_score(y_va, y_pred, labels=[2], average="macro", zero_division=0)
    return (prec_buy + prec_sell) / 2.0


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Run an Optuna study to find the best XGBoost hyperparameters.

    Uses the same walk-forward train/valid split as the training pipeline
    so there is zero look-ahead leakage during tuning.

    Args:
        X:        Feature matrix aligned to labels.
        y:        Label array {0, 1, 2}.
        n_trials: Number of Optuna trials (higher = better, slower).

    Returns:
        Dict of best hyperparameters ready to pass directly to train_model().
    """
    train_end = min(config.TRAIN_END_DAY * _CANDLES_PER_DAY, len(X))
    valid_end = min(config.VALID_END_DAY * _CANDLES_PER_DAY, len(X))

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_va, y_va = X[train_end:valid_end], y[train_end:valid_end]

    logger.info(
        "Optuna search: %d trials | train=%d val=%d | threshold=%.2f",
        n_trials, len(X_tr), len(X_va), config.ENTRY_PROB_THRESHOLD,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_thresholded_precision",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: _objective(trial, X_tr, y_tr, X_va, y_va),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    best = study.best_params
    logger.info(
        "Best params (thresholded val precision=%.4f):\n  %s",
        study.best_value,
        "\n  ".join(f"{k}: {v}" for k, v in best.items()),
    )
    return best
