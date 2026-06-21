"""XGBoost classifier training with walk-forward time-series validation."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, classification_report
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}

_CANDLES_PER_DAY = int(24 * 60 / config.CANDLE_INTERVAL_MINUTES)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Moderate upweight for BUY/SELL (1.5x) vs HOLD (1.0x).

    'balanced' gives BUY/SELL ~2.7x weight which boosts recall but hurts
    precision.  1.5x keeps the minority classes from being ignored while
    staying conservative enough to support high-threshold precision trading.
    """
    w = np.where(y == 0, 1.0, 1.5).astype(np.float32)
    return w


def apply_threshold(
    proba: np.ndarray,
    threshold: float = config.ENTRY_PROB_THRESHOLD,
) -> np.ndarray:
    """Return class predictions with a confidence gate.

    Predicts BUY (1) or SELL (2) only when P(class) >= threshold.
    Everything else is HOLD (0).  When both BUY and SELL exceed the
    threshold, the higher-confidence class wins.
    """
    y_pred = np.zeros(len(proba), dtype=int)
    buy_p  = proba[:, 1]
    sell_p = proba[:, 2]

    y_pred[buy_p  >= threshold] = 1
    y_pred[sell_p >= threshold] = 2

    both = (buy_p >= threshold) & (sell_p >= threshold)
    y_pred[both] = np.where(buy_p[both] >= sell_p[both], 1, 2)
    return y_pred


def _day_split(n_total: int) -> tuple[int, int]:
    """Return (train_end, valid_end) sample indices from day-based config."""
    train_end = min(config.TRAIN_END_DAY * _CANDLES_PER_DAY, n_total)
    valid_end = min(config.VALID_END_DAY * _CANDLES_PER_DAY, n_total)
    return train_end, valid_end


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    symbol: str = "BTC/USDT",
) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier using a strict walk-forward split.

    Training uses rows 0..TRAIN_END_DAY days, validation uses the next
    TRAIN_END_DAY..VALID_END_DAY slice (no shuffle), and the held-out test
    set is everything after VALID_END_DAY.

    Args:
        X:      Feature matrix (n_samples, TOTAL_FEATURES).
        y:      Integer label array {0, 1, 2}.
        params: Override XGBoost hyperparameters. Merged on top of defaults.
        symbol: Used to derive the model save path.

    Returns:
        Fitted XGBClassifier, also persisted to models/{symbol}_xgb.json.
    """
    merged = {**_DEFAULT_PARAMS, **(params or {})}
    train_end, valid_end = _day_split(len(X))

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_va, y_va = X[train_end:valid_end], y[train_end:valid_end]
    X_te, y_te = X[valid_end:], y[valid_end:]

    logger.info(
        "Walk-forward split -- train: %d  valid: %d  test: %d",
        len(X_tr), len(X_va), len(X_te),
    )

    classes, counts = np.unique(y_tr, return_counts=True)
    logger.info(
        "Training class counts -- %s",
        {int(c): int(n) for c, n in zip(classes, counts)},
    )

    model = xgb.XGBClassifier(**merged)
    model.fit(
        X_tr, y_tr,
        sample_weight=_sample_weights(y_tr),
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    if len(X_te) > 0:
        proba  = model.predict_proba(X_te)
        y_raw  = model.predict(X_te)

        # Show probability distribution so threshold can be set sensibly.
        # max_trade_p = highest probability for BUY or SELL on each row.
        max_trade_p = np.maximum(proba[:, 1], proba[:, 2])
        pcts = np.percentile(max_trade_p, [50, 75, 90, 95, 99])
        logger.info(
            "P(BUY|SELL) distribution on test set -- "
            "p50=%.3f  p75=%.3f  p90=%.3f  p95=%.3f  p99=%.3f  max=%.3f",
            *pcts, max_trade_p.max(),
        )
        for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            n = int(np.sum(max_trade_p >= thr))
            b = precision_score(y_te, apply_threshold(proba, thr), labels=[1], average="macro", zero_division=0)
            s = precision_score(y_te, apply_threshold(proba, thr), labels=[2], average="macro", zero_division=0)
            logger.info(
                "  threshold=%.2f  signals=%4d  BUY_prec=%.3f  SELL_prec=%.3f",
                thr, n, b, s,
            )

        y_thr = apply_threshold(proba)
        n_signals = int(np.sum(y_thr > 0))
        buy_prec_thr  = precision_score(y_te, y_thr, labels=[1], average="macro", zero_division=0)
        sell_prec_thr = precision_score(y_te, y_thr, labels=[2], average="macro", zero_division=0)

        logger.info(
            "Test set (threshold=%.2f) -- signals: %d/%d  "
            "BUY precision: %.4f  SELL precision: %.4f",
            config.ENTRY_PROB_THRESHOLD, n_signals, len(y_te),
            buy_prec_thr, sell_prec_thr,
        )
        logger.info(
            "--- Thresholded report (what the bot actually does) ---\n%s",
            classification_report(y_te, y_thr, target_names=["HOLD", "BUY", "SELL"]),
        )
        logger.info(
            "--- Raw model report (all predictions, for reference) ---\n%s",
            classification_report(y_te, y_raw, target_names=["HOLD", "BUY", "SELL"]),
        )

    model_path = config.MODEL_PATH_TEMPLATE.format(symbol=symbol.replace("/", "_"))
    model.save_model(model_path)
    logger.info("Model saved -> %s", model_path)
    return model


def load_model(symbol: str) -> xgb.XGBClassifier:
    """Load a persisted XGBoost model for `symbol`.

    Raises:
        FileNotFoundError: if the model file does not exist.
    """
    model_path = config.MODEL_PATH_TEMPLATE.format(symbol=symbol.replace("/", "_"))
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info("Model loaded <- %s", model_path)
    return model
