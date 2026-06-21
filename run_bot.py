"""Entry point — start the scalping bot in simulation or live mode.

Usage
-----
    python run_bot.py                   # honours SIMULATION_MODE in .env
    SIMULATION_MODE=false python run_bot.py   # override to live (dangerous!)
"""

import asyncio
import logging
import logging.handlers
import sys

import config
from src.data.storage import init_db
from src.execution.main_loop import run_bot


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    mode = "SIMULATION (paper trading)" if config.SIMULATION_MODE else "LIVE TRADING"
    logger.info("==========================================")
    logger.info("   BTC / ETH  %s Candle Trading Bot", config.CANDLE_INTERVAL)
    logger.info("   Mode : %s", mode)
    logger.info("==========================================")

    if not config.SIMULATION_MODE:
        logger.warning(
            "⚠  LIVE mode is active — real orders will be placed on CoinDCX. "
            "Ensure you have completed ≥ 14 days of simulation validation first."
        )
        if not config.COINDCX_API_KEY or not config.COINDCX_SECRET:
            logger.critical("COINDCX_API_KEY / COINDCX_SECRET not set.  Aborting.")
            sys.exit(1)

    init_db()

    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
    except RuntimeError as exc:
        logger.critical("Start-up failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unhandled fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
