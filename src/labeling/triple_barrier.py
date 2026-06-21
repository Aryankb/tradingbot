"""Triple Barrier Method labeling using intracandle HIGHs and LOWs.

Design rationale
----------------
Labeling checks the HIGH of each forward candle for the BUY barrier and
the LOW for the SELL barrier — not the close price.

This matters because in live trading, a TP limit order at (entry + 3xATR)
gets filled the moment price TOUCHES that level intracandle, regardless of
where the candle eventually closes.  Using close prices for labeling would
systematically under-count valid BUY/SELL events (a candle whose high
touched the TP but closed lower would be mislabelled HOLD).

Label-execution alignment
--------------------------
  Labeling : BUY  if any HIGH in next FORWARD_WINDOW candles >= ref + LABEL_TP_ATR_MULT x ATR
             SELL if any LOW  in next FORWARD_WINDOW candles <= ref - LABEL_TP_ATR_MULT x ATR
             HOLD otherwise

  Execution: TP = entry + TP_ATR_MULT x ATR   (== LABEL_TP_ATR_MULT x ATR)
             SL = entry - SL_ATR_MULT x ATR   (for BUY; mirrored for SELL)

LABEL_TP_ATR_MULT == TP_ATR_MULT == 3.0 so the model learns which candles
will reach the take-profit level within the forward window.

Ambiguous candle (high >= upper AND low <= lower)
--------------------------------------------------
When a single very volatile candle simultaneously touches both barriers we
cannot know intracandle order.  We resolve it by checking which barrier the
candle OPEN is closer to: if open is closer to the upper barrier we assign
BUY (momentum was upward), otherwise SELL.  This is rare in practice.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def label_triple_barrier(
    df: pd.DataFrame,
    atr_series: Optional[pd.Series] = None,
    tp_mult: float = config.LABEL_TP_ATR_MULT,
    forward_window: int = config.FORWARD_WINDOW,
) -> pd.Series:
    """Assign a directional label to every candle using intracandle HIGHs/LOWs.

    For candle i, scan candles i+1 .. i+forward_window:
      - Label 1 (BUY)  : first candle whose HIGH >= entry + tp_mult x ATR_i
      - Label 2 (SELL) : first candle whose LOW  <= entry - tp_mult x ATR_i
      - Label 0 (HOLD) : neither barrier touched within the window

    Args:
        df:             DataFrame from compute_features() with close/high/low/open.
        atr_series:     Per-candle ATR aligned to df.index.
                        Falls back to a fixed 0.5% threshold when None.
        tp_mult:        ATR multiplier for both barriers (default LABEL_TP_ATR_MULT).
        forward_window: Candles to scan ahead (default config.FORWARD_WINDOW).

    Returns:
        pd.Series of int8 labels {0, 1, 2} aligned to df.index.
        Last forward_window rows are always 0 (no future data).
    """
    closes = df["close"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    opens  = df["open"].values.astype(np.float64)
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    use_atr = atr_series is not None
    if use_atr:
        atrs = atr_series.values.astype(np.float64)
    else:
        logger.warning(
            "label_triple_barrier: no ATR series provided -- "
            "falling back to fixed 0.5%% TP fraction."
        )

    for i in range(n - forward_window):
        ref = closes[i]

        if use_atr:
            tp_dist = tp_mult * atrs[i]
        else:
            tp_dist = ref * 0.005  # 0.5% fallback when no ATR provided

        upper = ref + tp_dist   # BUY  barrier: HIGH of any forward candle must reach here
        lower = ref - tp_dist   # SELL barrier: LOW  of any forward candle must reach here

        label = 0
        for j in range(i + 1, i + 1 + forward_window):
            hit_upper = highs[j] >= upper
            hit_lower = lows[j]  <= lower

            if hit_upper and hit_lower:
                # Both barriers touched in the same candle — resolve by open direction
                label = 1 if opens[j] < (upper + lower) / 2 else 2
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = 2
                break
        # Loop exhausted without break -> HOLD

        labels[i] = label

    total  = n - forward_window
    n_buy  = int((labels[:total] == 1).sum())
    n_sell = int((labels[:total] == 2).sum())
    n_hold = total - n_buy - n_sell
    logger.info(
        "Labels (n=%d) -- HOLD: %d (%.1f%%)  BUY: %d (%.1f%%)  SELL: %d (%.1f%%)",
        total,
        n_hold, 100 * n_hold / max(total, 1),
        n_buy,  100 * n_buy  / max(total, 1),
        n_sell, 100 * n_sell / max(total, 1),
    )
    return pd.Series(labels, index=df.index, name="label", dtype="int8")
