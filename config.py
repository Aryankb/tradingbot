"""Central configuration — all tuneable constants live here."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
COINDCX_API_KEY: str = os.getenv("COINDCX_API_KEY", "")
COINDCX_SECRET: str = os.getenv("COINDCX_SECRET", "")

# ---------------------------------------------------------------------------
# Trading universe
# ---------------------------------------------------------------------------
SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT"]

# Binance symbol names for the live candle feed (public, no auth required)
BINANCE_SYMBOLS: dict[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
}

# CoinDCX pair identifiers kept for reference; candles now come from Binance
COINDCX_PAIRS: dict[str, str] = {
    "BTC/USDT": "KC-BTC_USDT",
    "ETH/USDT": "KC-ETH_USDT",
}

# CoinDCX market names for the private order endpoint
COINDCX_MARKETS: dict[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
}

# ---------------------------------------------------------------------------
# Candle / window settings
# ---------------------------------------------------------------------------
CANDLE_INTERVAL = "1h"
CANDLE_INTERVAL_MINUTES = 60

WINDOW_SIZE = 48          # 48 x 1hr = 2 days of context
FORWARD_WINDOW = 24       # labeling look-ahead: 24 candles = 1 day

FEATURES_PER_CANDLE = 8   # log_return, volatility, vol_change, ema9_ratio, ema21_ratio, rsi14, macd_norm, bb_pos
TOTAL_FEATURES = WINDOW_SIZE * FEATURES_PER_CANDLE  # 384  (48 x 8)

# ---------------------------------------------------------------------------
# Triple barrier labeling
# ---------------------------------------------------------------------------
# Labeling checks candle HIGHs for BUY and LOWs for SELL (not closes),
# mirroring how TP limit orders actually fill intracandle in live trading.
#
# LABEL_TP_ATR_MULT = 3.0: label BUY  if any next-24-candle HIGH  >= ref + 3xATR
#                          label SELL if any next-24-candle LOW   <= ref - 3xATR
# Matches TP_ATR_MULT exactly — the model learns which candles will reach the TP.
LABEL_TP_ATR_MULT: float = 3.0

# ---------------------------------------------------------------------------
# Inference thresholds
# ---------------------------------------------------------------------------
# For a 3-class softmax, P(class)=0.33 is baseline (random).
# 0.55 means the model is ~1.65x more confident than baseline — roughly
# equivalent to 0.75 in a binary classifier.  Raise this if you want
# fewer but higher-conviction trades.
ENTRY_PROB_THRESHOLD = 0.50   # minimum P(class) to open a trade
# Set to [1, 2] for both directions, [1] for BUY-only, [2] for SELL-only.
# Use BUY-only when SELL precision is below break-even on recent test data.
TRADE_SIDES: list = [1]     # BUY-only: SELL precision below break-even on current test data

# ---------------------------------------------------------------------------
# Risk / position sizing
# ---------------------------------------------------------------------------
# MARGIN_INR    : total capital you are deploying (in INR — your actual balance)
# FIXED_RISK_INR: max loss per trade in INR.  Defaults to 5% of MARGIN_INR.
# USD_INR_RATE  : fallback exchange rate used to convert INR margin to USD for
#                 quantity calculation (prices are USD-denominated USDT pairs).
#                 The bot fetches the live USDT/INR rate from CoinDCX at runtime
#                 and only falls back to this value if the fetch fails.
MARGIN_INR: float = float(os.getenv("MARGIN_INR", "10000"))
FIXED_RISK_INR: float = float(os.getenv("FIXED_RISK_INR", str(MARGIN_INR * 0.05)))
USD_INR_FALLBACK: float = float(os.getenv("USD_INR_RATE", "99.0"))

ATR_PERIOD = 14
# SL = 1.5 x ATR, TP = 3.0 x ATR  ->  1:2 R:R
# Labels use 2xATR (HIGH/LOW), execution targets 3xATR — model detects momentum
# onset at 2xATR; wider TP rides the move further with noise-tolerant SL.
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0

MAX_LEVERAGE = 20.0        # hard cap on computed leverage
MIN_LEVERAGE = 1.0

# ---------------------------------------------------------------------------
# Walk-forward split (days from start of history)
# ---------------------------------------------------------------------------
# Proportions: 70% train / 15% valid / 15% test (no shuffle, strict time order)
HISTORICAL_DAYS = 1500
TRAIN_END_DAY = 1050  # days 0-1050  (~70%)
VALID_END_DAY = 1275  # days 1050-1275 (~15%);  test = days 1275-1500 (~15%)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
DB_PATH = str(DATA_DIR / "trading_bot.db")
MODEL_PATH_TEMPLATE = str(MODEL_DIR / "{symbol}_xgb.json")

# ---------------------------------------------------------------------------
# Fee structure (CoinDCX Futures USDT-M, Regular 1 tier, market orders)
# ---------------------------------------------------------------------------
# Taker fee = 0.05% per side + 18% GST = 0.059% per side
# Round-trip (entry + exit) = 0.118%
TAKER_FEE_RATE: float = 0.0005   # 0.05% per side before GST
GST_RATE: float = 0.18            # 18% GST on fee
TAKER_FEE_WITH_GST: float = TAKER_FEE_RATE * (1 + GST_RATE)  # 0.059% per side

# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
SIMULATION_MODE: bool = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = str(LOG_DIR / "trading_bot.log")
