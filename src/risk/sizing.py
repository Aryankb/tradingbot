"""ATR calculation and dynamic position sizing with fixed-risk leverage."""

import numpy as np
import pandas as pd
import logging

import config

logger = logging.getLogger(__name__)


def _wilder_atr_array(df: pd.DataFrame, period: int) -> np.ndarray:
    """Return full Wilder ATR array aligned to df rows. NaN for first period-1 rows."""
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    atr = np.full(len(tr), np.nan)
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> float:
    """Return the most recent Wilder ATR(period) value as a scalar."""
    return float(_wilder_atr_array(df, period)[-1])


def compute_atr_series(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """Return per-row ATR values as a pd.Series aligned to df.index.

    Used by the labeling pipeline so training barriers match inference barriers.
    First (period-1) rows are NaN and must be dropped before labeling.
    """
    arr = _wilder_atr_array(df, period)
    return pd.Series(arr, index=df.index, name="atr")


def compute_position_size(
    entry_price: float,
    atr: float,
    side: int,
    usd_inr_rate: float = config.USD_INR_FALLBACK,
    fixed_risk_inr: float = config.FIXED_RISK_INR,
    margin_inr: float = config.MARGIN_INR,
    sl_mult: float = config.SL_ATR_MULT,
    tp_mult: float = config.TP_ATR_MULT,
) -> dict:
    """Compute SL, TP, quantity, and leverage for a new position.

    Leverage formula
    ----------------
        sl_pct   = sl_mult × ATR / entry_price
        leverage = fixed_risk_inr / (margin_inr × sl_pct)

    Leverage is clamped to [MIN_LEVERAGE, MAX_LEVERAGE] from config.

    Args:
        entry_price:    Current close price of the signal candle.
        atr:            14-period ATR value.
        side:           1 for BUY (long), 2 for SELL (short).
        fixed_risk_inr: Maximum acceptable loss per trade in INR (₹500).
        margin_inr:     Total margin deployed in INR (₹10 000).
        sl_mult:        ATR multiplier for stop-loss distance.
        tp_mult:        ATR multiplier for take-profit distance.

    Returns:
        Dict with keys:
            sl_price    : stop-loss price level
            tp_price    : take-profit price level
            sl_pct      : stop-loss as fraction of entry price
            leverage    : computed and clamped leverage
            quantity    : position size in asset units (USD notional / price)
    """
    sl_distance = sl_mult * atr
    tp_distance = tp_mult * atr

    if side == 1:  # Long
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:           # Short
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    sl_pct = sl_distance / entry_price
    if sl_pct == 0:
        sl_pct = 1e-6  # guard against zero ATR

    raw_leverage = fixed_risk_inr / (margin_inr * sl_pct)
    leverage = float(np.clip(raw_leverage, config.MIN_LEVERAGE, config.MAX_LEVERAGE))

    # Convert INR margin to USD before computing quantity.
    # entry_price is in USD (USDT), so quantity must be derived in USD.
    # Without this conversion, quantity would be inflated ~84x (INR/USD ratio).
    margin_usd = margin_inr / usd_inr_rate
    quantity = (margin_usd * leverage) / entry_price

    logger.debug(
        "Sizing [%s] entry=%.4f ATR=%.6f SL=%.4f TP=%.4f lev=%.2fx "
        "margin_usd=%.2f qty=%.6f (USD/INR=%.2f)",
        "BUY" if side == 1 else "SELL",
        entry_price, atr, sl_price, tp_price, leverage,
        margin_usd, quantity, usd_inr_rate,
    )

    return {
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pct": sl_pct,
        "leverage": leverage,
        "quantity": quantity,
    }
