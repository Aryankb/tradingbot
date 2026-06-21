"""Step 1 — Download and persist historical candles at the configured interval.

Usage
-----
    python scripts/download_historical.py
    python scripts/download_historical.py --days 1000 --symbols BTC/USDT
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.storage import init_db, clear_candles
from src.data.binance_downloader import download_historical

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from Binance and store in SQLite."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=config.HISTORICAL_DAYS,
        help=f"Calendar days to download (default: {config.HISTORICAL_DAYS})",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=config.SYMBOLS,
        help="Trading pairs to download (default: all configured symbols)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing candles before downloading (use when switching intervals)",
    )
    args = parser.parse_args()

    init_db()
    for symbol in args.symbols:
        if args.reset:
            clear_candles(symbol)
        download_historical(symbol, days=args.days)

    print("\nDownload complete.  Run  python scripts/train_model.py  next.")


if __name__ == "__main__":
    main()
