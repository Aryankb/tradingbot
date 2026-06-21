"""Sliding-window feature matrix construction from engineered candle data."""

import numpy as np
import pandas as pd
import logging

import config
from src.features.engineering import FEATURE_COLS

logger = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build a flattened sliding-window feature matrix from all candles.

    Each output row represents the last WINDOW_SIZE candles concatenated
    into a 1-D vector of shape (TOTAL_FEATURES,).

    Args:
        df: DataFrame produced by compute_features() — must contain FEATURE_COLS.

    Returns:
        (X, df_indices) where
          X.shape == (n_samples, TOTAL_FEATURES) with dtype float32,
          df_indices[i] is the integer position in `df` of the *final* candle
          of window i (i.e., the prediction point).
    """
    arr = df[FEATURE_COLS].values.astype(np.float32)
    n = len(arr)
    ws = config.WINDOW_SIZE

    if n < ws:
        logger.warning(
            "build_feature_matrix: %d rows < window size %d; returning empty.", n, ws
        )
        return (
            np.empty((0, config.TOTAL_FEATURES), dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    # sliding_window_view: shape (n - ws + 1, ws, n_features)
    view = np.lib.stride_tricks.sliding_window_view(
        arr, window_shape=(ws, len(FEATURE_COLS))
    )
    n_samples = view.shape[0]
    X = view.reshape(n_samples, -1).copy()
    df_indices = np.arange(ws - 1, n, dtype=np.int64)
    return X, df_indices


def build_live_vector(df: pd.DataFrame) -> np.ndarray:
    """Build a single inference vector from the last WINDOW_SIZE rows of df.

    Args:
        df: DataFrame produced by compute_features(), at least WINDOW_SIZE rows long.

    Returns:
        1-D np.ndarray of shape (TOTAL_FEATURES,) ready for model.predict_proba().

    Raises:
        ValueError: if df has fewer than WINDOW_SIZE rows.
    """
    ws = config.WINDOW_SIZE
    if len(df) < ws:
        raise ValueError(
            f"build_live_vector requires >= {ws} rows, got {len(df)}."
        )
    window = df[FEATURE_COLS].iloc[-ws:].values.astype(np.float32)
    return window.flatten()
