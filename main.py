"""FastAPI server exposing past trade data from the simulation DB.

Run with:
    uv run uvicorn main:app --reload --port 8000

Endpoints
---------
GET /trades              — all closed trades, newest first
GET /trades/open         — currently open trades
GET /trades/summary      — win rate, total net P&L, Sharpe, drawdown
GET /trades/{id}         — single trade by DB id
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.data.storage import load_trades, load_open_trades
from src.execution.trade_manager import TradeManager

app = FastAPI(title="Trading Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-serialisable list, replacing NaN with None."""
    return df.where(pd.notna(df), other=None).to_dict(orient="records")


@app.get("/trades")
def get_all_trades(symbol: str = None, limit: int = 100):
    """Return closed trades, newest first. Optionally filter by symbol."""
    df = load_trades().dropna(subset=["exit_time"])
    if symbol:
        df = df[df["symbol"] == symbol]
    df = df.sort_values("entry_time", ascending=False).head(limit)
    return {"count": len(df), "trades": _df_to_records(df)}


@app.get("/trades/open")
def get_open_trades():
    """Return all currently open (unclosed) trades."""
    df = load_open_trades()
    return {"count": len(df), "trades": _df_to_records(df)}


@app.get("/trades/summary")
def get_summary(symbol: str = None):
    """Return performance summary: win rate, net P&L, Sharpe, max drawdown."""
    df = load_trades().dropna(subset=["net_pnl_inr"])
    if symbol:
        df = df[df["symbol"] == symbol]

    if df.empty:
        return {"total_trades": 0, "message": "No closed trades yet."}

    wins = int((df["net_pnl_inr"] > 0).sum())
    losses = int((df["net_pnl_inr"] <= 0).sum())
    total = len(df)

    by_reason = df["exit_reason"].value_counts().to_dict()

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4),
        "total_gross_pnl_inr": round(float(df["pnl_inr"].sum()), 2),
        "total_fees_inr": round(float((df["entry_fee_inr"].fillna(0) + df["exit_fee_inr"].fillna(0)).sum()), 2),
        "total_net_pnl_inr": round(float(df["net_pnl_inr"].sum()), 2),
        "avg_net_pnl_per_trade": round(float(df["net_pnl_inr"].mean()), 2),
        "exits": by_reason,
        "metrics": TradeManager.compute_metrics(),
    }


@app.get("/trades/{trade_id}")
def get_trade(trade_id: int):
    """Return a single trade by its DB id."""
    df = load_trades()
    row = df[df["id"] == trade_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found.")
    return _df_to_records(row)[0]


def main():
    print("Hello from trading-bot!")


if __name__ == "__main__":
    main()


"""
2026-06-21 21:28:16,020 [INFO] src.execution.main_loop: -- tick 15:58:15  boundary=False  usd_inr=100.56 --
2026-06-21 21:28:16,039 [INFO] src.execution.main_loop:   [BTC/USDT] price=64233.3000  HOLD=0.212 BUY=0.447 SELL=0.341  -> BUY (0.447)  ATR=236.2715  no open trade
2026-06-21 21:28:16,066 [INFO] src.execution.main_loop:   [ETH/USDT] price=1730.7600   HOLD=0.176 BUY=0.496 SELL=0.328  -> BUY (0.496)  ATR=8.3114  no open trade
2026-06-21 21:29:16,000 [INFO] src.execution.main_loop: -- tick 15:59:15  boundary=False  usd_inr=100.65 --
2026-06-21 21:29:16,056 [INFO] src.execution.main_loop:   [BTC/USDT] price=64239.9700  HOLD=0.212 BUY=0.448 SELL=0.340  -> BUY (0.448)  ATR=236.2715  no open trade
2026-06-21 21:29:16,078 [INFO] src.execution.main_loop:   [ETH/USDT] price=1730.8200   HOLD=0.176 BUY=0.496 SELL=0.328  -> BUY (0.496)  ATR=8.3114  no open trade
2026-06-21 21:30:18,244 [INFO] src.execution.main_loop: -- tick 16:00:15  boundary=True  usd_inr=100.57 --
2026-06-21 21:30:18,294 [INFO] src.execution.main_loop:   [BTC/USDT] price=64208.0500  HOLD=0.303 BUY=0.390 SELL=0.307  -> BUY (0.390)  ATR=220.5572  no open trade
2026-06-21 21:30:18,296 [INFO] src.execution.main_loop:   *** CANDLE BOUNDARY [BTC/USDT] — checking entry ***
2026-06-21 21:30:18,408 [INFO] src.execution.main_loop:   [ETH/USDT] price=1730.2700   HOLD=0.253 BUY=0.435 SELL=0.313  -> BUY (0.435)  ATR=7.7513  no open trade
2026-06-21 21:30:18,408 [INFO] src.execution.main_loop:   *** CANDLE BOUNDARY [ETH/USDT] — checking entry ***
2026-06-21 21:31:16,327 [INFO] src.execution.main_loop: -- tick 16:01:15  boundary=False  usd_inr=100.58 --
2026-06-21 21:31:16,352 [INFO] src.execution.main_loop:   [BTC/USDT] price=64198.0100  HOLD=0.302 BUY=0.390 SELL=0.308  -> BUY (0.390)  ATR=221.5408  no open trade
2026-06-21 21:31:16,378 [INFO] src.execution.main_loop:   [ETH/USDT] price=1728.9500   HOLD=0.254 BUY=0.430 SELL=0.316  -> BUY (0.430)  ATR=7.8541  no open trade
2026-06-21 21:32:16,030 [INFO] src.execution.main_loop: -- tick 16:02:15  boundary=False  usd_inr=100.58 --
2026-06-21 21:32:16,045 [INFO] src.execution.main_loop:   [BTC/USDT] price=64232.5600  HOLD=0.301 BUY=0.391 SELL=0.308  -> BUY (0.391)  ATR=222.7479  no open trade
2026-06-21 21:32:16,077 [INFO] src.execution.main_loop:   [ETH/USDT] price=1729.7000   HOLD=0.254 BUY=0.431 SELL=0.315  -> BUY (0.431)  ATR=7.8570  no open trade
2026-06-21 21:33:16,084 [INFO] src.execution.main_loop: -- tick 16:03:15  boundary=False  usd_inr=100.59 --
2026-06-21 21:33:16,103 [INFO] src.execution.main_loop:   [BTC/USDT] price=64248.0200  HOLD=0.301 BUY=0.391 SELL=0.308  -> BUY (0.391)  ATR=223.5593  no open trade
2026-06-21 21:33:16,148 [INFO] src.execution.main_loop:   [ETH/USDT] price=1730.4900   HOLD=0.248 BUY=0.444 SELL=0.308  -> BUY (0.444)  ATR=7.8570  no open trade




when boundra true, why probability changed significantly ? - because ohlc data is of last candle and current price is of current candle. So when boundary is true, the model sees a new candle and the features change significantly. This is expected behavior.




In live mode the bot places the entry order via place_market_order and then never closes the position. This is a gap that needs to be fixed before you go live — it would need to either place SL/TP stop orders on CoinDCX at entry time, or execute a closing market order at the 24-hour limit.
For now since you're in simulation this doesn't matter, but flag it before switching SIMULATION_MODE=false

in future, for BTC selling, use previous model if probab is between 0.55 and 0.6

do something like - if goes minus 1.5*ATR but never reaches - 3*ATR then hold
BUTTTT! if doing above thing, it must increase precision accuracy, and should have enough signals too in high volatility scenerio

All done. Now when you run the simulation, every trade saved to the DB will have atr, sl_pct, and notional_inr. After a few weeks of simulation you can query:


SELECT AVG(sl_pct), AVG(notional_inr), AVG(atr)
FROM trades WHERE sl_pct > 0.015  -- high-vol trades only
to see how often high-volatility conditions occur and whether their P&L is better.


exactly where you are doing inr to usd or usd to inr? is it only while placing order? coindcx does have inr wallet in futures which i used in manual trading. i doubt there still might be some assumptions . should i fetch the latest coindcx docs for you?


are you really very sure that download_history.py downloads latest data and overrides in db, then train_model.py takes same recently downloaded data for the correct coin?

"""
