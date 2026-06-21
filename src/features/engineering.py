"""Per-candle stationary feature engineering on raw OHLCV DataFrames."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

_EMA9_SPAN   = 9
_EMA21_SPAN  = 21
_EMA12_SPAN  = 12
_EMA26_SPAN  = 26
_VOL_MA_PERIOD = 20
_RSI_PERIOD    = 14
_BB_PERIOD     = 20

# The ordered feature column names used throughout the pipeline.
# sliding_window.py and build_live_vector() read this list directly,
# so adding a name here automatically extends the feature vector.
FEATURE_COLS: list[str] = [
    "log_return",    # log(close_t / close_{t-1})
    "volatility",    # (high - low) / close
    "vol_change",    # volume / rolling_mean(volume, 20)
    "ema9_ratio",    # close / EMA(9) - 1
    "ema21_ratio",   # close / EMA(21) - 1
    "rsi14",         # Wilder RSI(14) normalised to [-1, 1]
    "macd_norm",     # (EMA12 - EMA26) / close
    "bb_pos",        # (close - lower_band) / band_width, clipped [0, 1]
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi_series(close: pd.Series) -> pd.Series:
    """Wilder-smoothed RSI(14), normalised to [-1, 1]."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / _RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / _RSI_PERIOD, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi - 50.0) / 50.0   # [-1, 1]; NaN when avg_loss==0 handled below


def _macd_norm_series(close: pd.Series) -> pd.Series:
    """(EMA12 - EMA26) / close — dimensionless momentum."""
    ema12 = close.ewm(span=_EMA12_SPAN, adjust=False).mean()
    ema26 = close.ewm(span=_EMA26_SPAN, adjust=False).mean()
    return (ema12 - ema26) / close


def _bb_pos_series(close: pd.Series) -> pd.Series:
    """Position within Bollinger Bands (20-period, 2 std-dev), clipped [0, 1]."""
    mean   = close.rolling(_BB_PERIOD, min_periods=_BB_PERIOD).mean()
    std    = close.rolling(_BB_PERIOD, min_periods=_BB_PERIOD).std(ddof=0)
    upper  = mean + 2.0 * std
    lower  = mean - 2.0 * std
    width  = (upper - lower).replace(0.0, np.nan)
    pos    = (close - lower) / width
    return pos.clip(0.0, 1.0).fillna(0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment an OHLCV DataFrame with stationary engineered features.

    Returns a copy with feature columns appended and NaN rows removed.
    """
    df = df.copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = (df["high"] - df["low"]) / df["close"]

    vol_ma = df["volume"].rolling(_VOL_MA_PERIOD, min_periods=1).mean()
    df["vol_change"] = df["volume"] / vol_ma.replace(0, np.nan)

    ema9  = df["close"].ewm(span=_EMA9_SPAN,  adjust=False).mean()
    ema21 = df["close"].ewm(span=_EMA21_SPAN, adjust=False).mean()
    df["ema9_ratio"]  = df["close"] / ema9  - 1.0
    df["ema21_ratio"] = df["close"] / ema21 - 1.0

    df["rsi14"]     = _rsi_series(df["close"])
    df["macd_norm"] = _macd_norm_series(df["close"])
    df["bb_pos"]    = _bb_pos_series(df["close"])

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def patch_last_close(df: pd.DataFrame, live_price: float) -> pd.DataFrame:
    """Replace the last row's close with the current live market price
    and recompute all close-derived features for that row.

    `volatility` and `vol_change` are left unchanged because the live
    ticker does not supply the current candle's high/low or volume.
    """
    if df.empty or live_price <= 0:
        return df

    df  = df.copy()
    idx = df.index[-1]

    df.at[idx, "close"] = live_price

    # log_return
    prev_close = float(df.at[idx - 1, "close"]) if idx > 0 else live_price
    df.at[idx, "log_return"] = float(np.log(live_price / prev_close))

    # EMA-based features — recompute full series, read last value
    ema9  = df["close"].ewm(span=_EMA9_SPAN,  adjust=False).mean()
    ema21 = df["close"].ewm(span=_EMA21_SPAN, adjust=False).mean()
    df.at[idx, "ema9_ratio"]  = live_price / float(ema9.iloc[-1])  - 1.0
    df.at[idx, "ema21_ratio"] = live_price / float(ema21.iloc[-1]) - 1.0

    rsi_s  = _rsi_series(df["close"])
    df.at[idx, "rsi14"] = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 0.0

    macd_s = _macd_norm_series(df["close"])
    df.at[idx, "macd_norm"] = float(macd_s.iloc[-1])

    bb_s = _bb_pos_series(df["close"])
    df.at[idx, "bb_pos"] = float(bb_s.iloc[-1])

    return df
