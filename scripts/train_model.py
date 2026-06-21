"""Step 2 — Feature engineering -> Triple Barrier labeling -> Optuna tuning -> XGBoost fit.

Usage
-----
    python scripts/train_model.py                       # train with defaults
    python scripts/train_model.py --tune                # run Optuna first
    python scripts/train_model.py --tune --trials 100   # 100 Optuna trials
    python scripts/train_model.py --symbols ETH/USDT    # single symbol
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config
from src.data.storage import init_db, load_candles
from src.features.engineering import compute_features
from src.features.sliding_window import build_feature_matrix
from src.labeling.triple_barrier import label_triple_barrier
from src.model.tuner import tune_hyperparameters
from src.model.trainer import train_model
from src.risk.sizing import compute_atr_series

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline(symbol: str, tune: bool, n_trials: int) -> None:
    """End-to-end training pipeline for one symbol."""
    logger.info("=== Training pipeline: %s ===", symbol)

    # 1. Load raw candles
    df_raw = load_candles(symbol)
    if df_raw.empty:
        raise RuntimeError(
            f"No candles in DB for {symbol}. "
            "Run 'python scripts/download_historical.py' first."
        )
    logger.info("Loaded %d raw candles.", len(df_raw))

    # 2. Feature engineering
    df_feat = compute_features(df_raw)
    logger.info("Feature engineering complete: %d rows.", len(df_feat))

    # 3. Compute per-candle ATR series (same formula used at inference time)
    #    Then label using ATR-based barriers so training fully matches execution.
    atr_series = compute_atr_series(df_feat)
    valid_atr = atr_series.dropna()
    logger.info(
        "ATR range across dataset — min: %.4f  max: %.4f  mean: %.4f",
        float(valid_atr.min()), float(valid_atr.max()), float(valid_atr.mean()),
    )

    # 4. Triple Barrier labeling with ATR-based dynamic barriers
    labels = label_triple_barrier(df_feat, atr_series=atr_series)

    # 5. Build sliding-window feature matrix
    X, row_indices = build_feature_matrix(df_feat)
    y = labels.iloc[row_indices].values.astype(np.int32)

    # Drop the last FORWARD_WINDOW samples (insufficient future data for labeling)
    valid_mask = row_indices < (len(df_feat) - config.FORWARD_WINDOW)
    X = X[valid_mask]
    y = y[valid_mask]
    logger.info("Feature matrix: %s  (labels trimmed for look-ahead safety)", X.shape)

    # 6. Optuna hyperparameter search (optional)
    params = None
    if tune:
        logger.info("Starting Optuna search (%d trials)…", n_trials)
        params = tune_hyperparameters(X, y, n_trials=n_trials)

    # 7. Fit and save
    train_model(X, y, params=params, symbol=symbol)
    logger.info("Pipeline complete for %s.", symbol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost classification model."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=config.SYMBOLS,
        help="Symbols to train (default: all configured)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter search before training",
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials when --tune is set (default: 100)",
    )
    args = parser.parse_args()

    init_db()
    for symbol in args.symbols:
        run_pipeline(symbol, tune=args.tune, n_trials=args.trials)

    print(
        "\nTraining complete.  Run  python run_bot.py  to start the simulation."
    )


if __name__ == "__main__":
    main()
