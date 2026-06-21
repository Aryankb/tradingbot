"""CoinDCX private trading API client.

All order-placement calls are gated by config.SIMULATION_MODE.
In simulation mode every function logs the intended action and returns a
mock response without touching the exchange.
"""

import aiohttp
import hmac
import hashlib
import json
import time
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_ORDER_URL = "https://api.coindcx.com/exchange/v1/orders/create"
_CANCEL_URL = "https://api.coindcx.com/exchange/v1/orders/cancel"
_STATUS_URL = "https://api.coindcx.com/exchange/v1/orders/status"


def _sign_body(body: dict) -> str:
    """Return HMAC-SHA256 signature of the JSON-serialised body."""
    json_body = json.dumps(body, separators=(",", ":"))
    return hmac.new(
        config.COINDCX_SECRET.encode("utf-8"),
        json_body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _auth_headers(body: dict) -> dict:
    return {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": config.COINDCX_API_KEY,
        "X-AUTH-SIGNATURE": _sign_body(body),
    }


async def place_market_order(
    symbol: str,
    side: str,          # "buy" or "sell"
    quantity: float,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Place a market order on CoinDCX.

    In SIMULATION_MODE, logs the intent and returns a synthetic response.

    Args:
        symbol:   Trading pair in bot notation, e.g. "BTC/USDT".
        side:     "buy" or "sell".
        quantity: Asset quantity (not notional).
        session:  Shared aiohttp session.

    Returns:
        Exchange response dict (or synthetic dict in simulation).
    """
    market = config.COINDCX_MARKETS.get(symbol)
    if not market:
        raise ValueError(f"Symbol '{symbol}' not in COINDCX_MARKETS config.")

    if config.SIMULATION_MODE:
        logger.info("[SIM] place_market_order %s %s qty=%.6f", side.upper(), symbol, quantity)
        return {"id": f"sim_{int(time.time() * 1000)}", "status": "filled", "side": side}

    timestamp = int(time.time() * 1000)
    body = {
        "side": side,
        "order_type": "market_order",
        "market": market,
        "total_quantity": round(quantity, 6),
        "timestamp": timestamp,
    }
    headers = _auth_headers(body)

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with session.post(
            _ORDER_URL,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
            logger.info("Order placed: %s", result)
            return result
    except aiohttp.ClientError as exc:
        logger.error("place_market_order failed [%s %s]: %s", side, symbol, exc)
        raise
    finally:
        if owns_session:
            await session.close()


async def cancel_order(
    order_id: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Cancel an open limit order by its exchange ID."""
    if config.SIMULATION_MODE:
        logger.info("[SIM] cancel_order %s", order_id)
        return {"id": order_id, "status": "cancelled"}

    timestamp = int(time.time() * 1000)
    body = {"id": order_id, "timestamp": timestamp}
    headers = _auth_headers(body)

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with session.delete(
            _CANCEL_URL,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as exc:
        logger.error("cancel_order failed [%s]: %s", order_id, exc)
        raise
    finally:
        if owns_session:
            await session.close()


async def get_order_status(
    order_id: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Fetch the current status of an order."""
    if config.SIMULATION_MODE:
        return {"id": order_id, "status": "filled"}

    timestamp = int(time.time() * 1000)
    body = {"id": order_id, "timestamp": timestamp}
    headers = _auth_headers(body)

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with session.get(
            _STATUS_URL,
            params={"id": order_id},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as exc:
        logger.error("get_order_status failed [%s]: %s", order_id, exc)
        raise
    finally:
        if owns_session:
            await session.close()
