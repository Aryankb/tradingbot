"""In-memory active trade state + historical performance metrics."""

import time
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import config
from src.data.storage import insert_trade, update_trade, load_trades, load_open_trades

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    symbol: str
    side: int           # 1 = BUY (long)  |  2 = SELL (short)
    entry_time: int     # Unix ms
    entry_price: float
    sl_price: float
    tp_price: float
    quantity: float
    leverage: float
    entry_prob: float
    atr: float = 0.0
    sl_pct: float = 0.0
    notional_inr: float = 0.0
    db_id: Optional[int] = None  # row id in trades table


class TradeManager:
    """Thread-safe (single-event-loop) manager for open positions."""

    def __init__(self) -> None:
        self._open: dict[str, ActiveTrade] = {}
        self._recover_open_trades()

    def _recover_open_trades(self) -> None:
        """Reload any unclosed trades from DB into memory after a restart."""
        df = load_open_trades()
        for _, row in df.iterrows():
            trade = ActiveTrade(
                symbol=row["symbol"],
                side=int(row["side"]),
                entry_time=int(row["entry_time"]),
                entry_price=float(row["entry_price"]),
                sl_price=float(row["sl_price"]),
                tp_price=float(row["tp_price"]),
                quantity=float(row["quantity"]),
                leverage=float(row["leverage"]),
                entry_prob=float(row["entry_prob"]) if row["entry_prob"] else 0.0,
                atr=float(row["atr"]) if row["atr"] else 0.0,
                sl_pct=float(row["sl_pct"]) if row["sl_pct"] else 0.0,
                notional_inr=float(row["notional_inr"]) if row["notional_inr"] else 0.0,
                db_id=int(row["id"]),
            )
            self._open[row["symbol"]] = trade
            logger.info(
                "RECOVERED open trade [%s] %s @ %.4f  SL=%.4f  TP=%.4f  [id=%d]",
                row["symbol"], ["HOLD", "BUY", "SELL"][int(row["side"])],
                float(row["entry_price"]), float(row["sl_price"]),
                float(row["tp_price"]), int(row["id"]),
            )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has_open(self, symbol: str) -> bool:
        """Return True if a position is currently open for `symbol`."""
        return symbol in self._open

    def get_open(self, symbol: str) -> Optional[ActiveTrade]:
        """Return the active trade for `symbol`, or None."""
        return self._open.get(symbol)

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        side: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        quantity: float,
        leverage: float,
        entry_prob: float,
        atr: float = 0.0,
        sl_pct: float = 0.0,
        notional_inr: float = 0.0,
    ) -> ActiveTrade:
        """Record a new position entry and persist it to the database.

        Returns the created ActiveTrade dataclass.
        """
        now_ms = int(time.time() * 1000)
        trade = ActiveTrade(
            symbol=symbol,
            side=side,
            entry_time=now_ms,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            quantity=quantity,
            leverage=leverage,
            entry_prob=entry_prob,
            atr=atr,
            sl_pct=sl_pct,
            notional_inr=notional_inr,
        )
        # Entry fee: taker fee on notional at entry
        entry_fee_inr = notional_inr * config.TAKER_FEE_WITH_GST

        db_record = {
            "symbol": symbol,
            "side": side,
            "entry_time": now_ms,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "quantity": quantity,
            "leverage": leverage,
            "entry_prob": entry_prob,
            "atr": atr,
            "sl_pct": sl_pct,
            "notional_inr": notional_inr,
            "entry_fee_inr": round(entry_fee_inr, 2),
        }
        trade.db_id = insert_trade(db_record)
        self._open[symbol] = trade

        logger.info(
            "OPEN  %-8s %-4s @ %10.4f  SL=%.4f  TP=%.4f  qty=%.6f  "
            "lev=%.2fx  prob=%.3f  ATR=%.2f  sl_pct=%.3f%%  notional=INR%.0f  entry_fee=INR%.0f  [id=%s]",
            symbol,
            "BUY" if side == 1 else "SELL",
            entry_price, sl_price, tp_price, quantity, leverage, entry_prob,
            atr, sl_pct * 100, notional_inr, entry_fee_inr, trade.db_id,
        )
        return trade

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> Optional[float]:
        """Close a position, compute INR P&L, update the database row.

        Args:
            symbol:     Trading pair.
            exit_price: Market price at exit.
            reason:     One of 'sl_hit', 'tp_hit', 'soft_exit', or 'manual'.

        Returns:
            Realised P&L in INR, or None if no open trade exists.
        """
        trade = self._open.pop(symbol, None)
        if trade is None:
            logger.warning("close_trade called for %s but no open position found.", symbol)
            return None

        price_delta = exit_price - trade.entry_price
        if trade.side == 2:          # short: profit when price falls
            price_delta = -price_delta

        pnl_pct = price_delta / trade.entry_price
        # Gross INR P&L = margin × leverage × price_change_fraction
        pnl_inr = config.MARGIN_INR * trade.leverage * pnl_pct

        # Exit fee: taker fee on exit notional (notional shifts slightly with price)
        exit_notional_inr = trade.notional_inr * (exit_price / trade.entry_price)
        exit_fee_inr = exit_notional_inr * config.TAKER_FEE_WITH_GST

        # Recover entry fee from DB record (stored at open time)
        entry_fee_inr = trade.notional_inr * config.TAKER_FEE_WITH_GST
        net_pnl_inr = pnl_inr - entry_fee_inr - exit_fee_inr

        now_ms = int(time.time() * 1000)
        if trade.db_id is not None:
            update_trade(trade.db_id, {
                "exit_time": now_ms,
                "exit_price": exit_price,
                "pnl_inr": round(pnl_inr, 2),
                "exit_reason": reason,
                "exit_fee_inr": round(exit_fee_inr, 2),
                "net_pnl_inr": round(net_pnl_inr, 2),
            })

        tag = "WIN " if net_pnl_inr >= 0 else "LOSS"
        logger.info(
            "CLOSE [%s] %-8s %-4s exit=%.4f entry=%.4f  "
            "gross=INR%+.0f  fees=INR%.0f  net=INR%+.0f  reason=%s",
            tag, symbol,
            "BUY" if trade.side == 1 else "SELL",
            exit_price, trade.entry_price,
            pnl_inr, entry_fee_inr + exit_fee_inr, net_pnl_inr, reason,
        )
        return net_pnl_inr

    # ------------------------------------------------------------------
    # Performance analytics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics() -> dict:
        """Compute Sharpe Ratio, Max Drawdown, win rate from closed trades.

        Returns a dict with keys:
            sharpe, max_drawdown_inr, total_trades, win_rate, total_pnl_inr
        """
        trades = load_trades().dropna(subset=["net_pnl_inr"])
        if trades.empty:
            return {
                "sharpe": 0.0,
                "max_drawdown_inr": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_inr": 0.0,
            }

        pnl = trades["net_pnl_inr"].values.astype(np.float64)

        # Max drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float((running_max - cumulative).max())

        # Trade-level Sharpe annualised: ~360 trades/year (30 trades/month × 12)
        mean_p = pnl.mean()
        std_p = pnl.std(ddof=1) if len(pnl) > 1 else 1e-9
        sharpe = (mean_p / std_p) * np.sqrt(360)

        wins = int((pnl > 0).sum())
        win_rate = wins / len(pnl)

        return {
            "sharpe": round(float(sharpe), 4),
            "max_drawdown_inr": round(max_dd, 2),
            "total_trades": len(pnl),
            "win_rate": round(win_rate, 4),
            "total_pnl_inr": round(float(cumulative[-1]), 2),
        }
