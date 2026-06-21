"""Core async execution engine — the while-True trading loop.

Every 60-second tick
--------------------
  ① Fetch completed candles + live ticker price (one network round-trip each)
  ② Patch the last candle's close with the live price before feature engineering
     so log_return and EMA ratios reflect the current market price, not the
     stale candle close.
  ③ In simulation: check hard SL/TP against live price for any open trade.

  On a candle boundary (every CANDLE_INTERVAL_MINUTES)
  ④ Run full inference.  If P(BUY|SELL) >= ENTRY_PROB_THRESHOLD -> open position
     using the live price (not candle close) for entry, SL, and TP.

  Hourly
  ⑤ Log Sharpe Ratio, Max Drawdown, win rate.
"""

import asyncio
import aiohttp
import logging
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional

import xgboost as xgb

import config
from src.data.coindcx_feed import fetch_all_symbols, get_live_prices, get_usd_inr_rate
from src.data.storage import upsert_candles
from src.features.engineering import compute_features, patch_last_close
from src.features.sliding_window import build_live_vector
from src.model.trainer import load_model
from src.risk.sizing import compute_atr, compute_position_size
from src.execution.trade_manager import TradeManager, ActiveTrade
from src.execution.coindcx_trader import place_market_order

logger = logging.getLogger(__name__)

_FETCH_LIMIT = config.WINDOW_SIZE + 60   # extra 60: BB warm-up (20) + ATR warm-up (14) + spare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sl_tp(trade: ActiveTrade, price: float) -> Optional[str]:
    """Return 'sl_hit', 'tp_hit', or None."""
    if trade.side == 1:
        if price <= trade.sl_price:
            return "sl_hit"
        if price >= trade.tp_price:
            return "tp_hit"
    else:
        if price >= trade.sl_price:
            return "sl_hit"
        if price <= trade.tp_price:
            return "tp_hit"
    return None


# ---------------------------------------------------------------------------
# Inference (one fetch, one patch, one predict_proba call)
# ---------------------------------------------------------------------------

def _build_inference(
    df_raw,
    live_price: Optional[float],
    model: xgb.XGBClassifier,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Build patched feature vector and return (proba_array, atr).

    Returns (None, None) if there is not enough data.
    proba_array has shape (3,): [P(HOLD), P(BUY), P(SELL)].
    """
    if df_raw is None or df_raw.empty:
        return None, None

    df_feat = compute_features(df_raw)
    if len(df_feat) < config.WINDOW_SIZE:
        return None, None

    # Patch last candle close with live price so features reflect current market
    if live_price and live_price > 0:
        df_feat = patch_last_close(df_feat, live_price)

    x = build_live_vector(df_feat).reshape(1, -1)
    proba: np.ndarray = model.predict_proba(x)[0]   # shape (3,)
    atr = compute_atr(df_feat)
    return proba, atr


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_bot(simulation: bool = config.SIMULATION_MODE) -> None:
    """Start the trading bot. Runs indefinitely; stop with KeyboardInterrupt."""
    models: dict[str, xgb.XGBClassifier] = {}
    for sym in config.SYMBOLS:
        try:
            models[sym] = load_model(sym)
        except Exception as exc:
            logger.error("Could not load model for %s: %s — skipped.", sym, exc)

    if not models:
        raise RuntimeError("No models loaded. Run 'python scripts/train_model.py' first.")

    manager = TradeManager()
    mode_tag = "SIMULATION" if simulation else "LIVE"
    logger.info("=== %s mode ===  symbols: %s", mode_tag, list(models.keys()))

    last_boundary: Optional[datetime] = None

    connector = aiohttp.TCPConnector(use_dns_cache=False, family=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            tick_start = time.monotonic()
            now = datetime.now(timezone.utc)
            # Current candle open time (floor to the nearest interval)
            current_boundary = now.replace(
                minute=0, second=0, microsecond=0
            ) if config.CANDLE_INTERVAL_MINUTES >= 60 else now.replace(
                minute=(now.minute // config.CANDLE_INTERVAL_MINUTES) * config.CANDLE_INTERVAL_MINUTES,
                second=0, microsecond=0,
            )
            on_boundary = last_boundary is None or current_boundary > last_boundary
            if on_boundary:
                last_boundary = current_boundary

            # ── Single network round-trip per tick for all symbols ────────────
            candle_feeds, live_prices, usd_inr = await asyncio.gather(
                fetch_all_symbols(session, limit=_FETCH_LIMIT),
                get_live_prices(session),
                get_usd_inr_rate(session),
            )

            logger.info(
                "-- tick %s  boundary=%s  usd_inr=%.2f --",
                now.strftime("%H:%M:%S"), on_boundary, usd_inr or 0,
            )

            for symbol, model in models.items():
                df_raw    = candle_feeds.get(symbol)
                live_px   = live_prices.get(symbol)
                trade     = manager.get_open(symbol)

                # Persist candles so DB is always current
                if df_raw is not None and not df_raw.empty:
                    upsert_candles(symbol, df_raw)

                proba, atr = _build_inference(df_raw, live_px, model)
                if proba is None:
                    logger.warning("[%s] Skipping tick — insufficient data.", symbol)
                    continue

                pred_class  = int(proba.argmax())
                max_prob    = float(proba[pred_class])
                entry_price = live_px if (live_px and live_px > 0) else None

                open_info = (
                    f"  OPEN {['HOLD','BUY','SELL'][trade.side]} "
                    f"entry={trade.entry_price:.2f} "
                    f"SL={trade.sl_price:.2f} TP={trade.tp_price:.2f}"
                    if trade else "  no open trade"
                )
                logger.info(
                    "  [%s] price=%-10.4f  HOLD=%.3f BUY=%.3f SELL=%.3f  "
                    "-> %-4s(%.3f)  ATR=%.4f%s",
                    symbol, entry_price or 0,
                    proba[0], proba[1], proba[2],
                    ["HOLD", "BUY", "SELL"][pred_class], max_prob,
                    atr or 0, open_info,
                )

                # ── ③ SL/TP + time-limit check (simulation only, every tick) ──
                if simulation and trade is not None and entry_price is not None:
                    hit = _check_sl_tp(trade, entry_price)
                    if hit:
                        pnl = manager.close_trade(symbol, entry_price, hit)
                        logger.info(
                            "%s hit [%s] @ %.4f  PnL=Rs%+.2f",
                            hit.upper(), symbol, entry_price, pnl or 0,
                        )
                        trade = None
                    else:
                        # Time-based exit: close after FORWARD_WINDOW candles
                        # aligns with labeling horizon (HOLD = "not resolved in time")
                        max_hold_ms = (
                            config.FORWARD_WINDOW
                            * config.CANDLE_INTERVAL_MINUTES
                            * 60 * 1000
                        )
                        if (int(time.time() * 1000) - trade.entry_time) >= max_hold_ms:
                            pnl = manager.close_trade(symbol, entry_price, "time_limit")
                            logger.info(
                                "TIME LIMIT [%s] @ %.4f  PnL=Rs%+.2f",
                                symbol, entry_price, pnl or 0,
                            )
                            trade = None

                # ── ④ Candle boundary: new entry check ───────────────────────
                if on_boundary:
                    logger.info(
                        "  *** CANDLE BOUNDARY [%s] — checking entry ***",
                        symbol,
                    )

                    if (
                        pred_class in config.TRADE_SIDES
                        and max_prob >= config.ENTRY_PROB_THRESHOLD
                        and not manager.has_open(symbol)
                        and entry_price is not None
                        and atr is not None
                    ):
                        sizing = compute_position_size(
                            entry_price, atr, side=pred_class, usd_inr_rate=usd_inr
                        )
                        manager.open_trade(
                            symbol=symbol,
                            side=pred_class,
                            entry_price=entry_price,      # live price, not stale candle close
                            sl_price=sizing["sl_price"],
                            tp_price=sizing["tp_price"],
                            quantity=sizing["quantity"],
                            leverage=sizing["leverage"],
                            entry_prob=max_prob,
                            atr=atr,
                            sl_pct=sizing["sl_pct"],
                            notional_inr=config.FIXED_RISK_INR / sizing["sl_pct"],
                        )
                        if not simulation:
                            order_side = "buy" if pred_class == 1 else "sell"
                            await place_market_order(
                                symbol, order_side, sizing["quantity"], session
                            )

            # ── ⑤ Hourly metrics ─────────────────────────────────────────────
            if now.minute == 0 and now.second < 5:
                metrics = TradeManager.compute_metrics()
                logger.info("HOURLY METRICS: %s", metrics)

            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, 60.0 - elapsed))
