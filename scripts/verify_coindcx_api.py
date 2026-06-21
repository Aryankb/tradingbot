"""Verify live data feed endpoints used by the bot.

Candles  — Binance public klines (no auth required)
Ticker   — CoinDCX public ticker (no auth required)

Usage:
    python scripts/verify_coindcx_api.py
"""

import asyncio
import json
import sys
import aiohttp

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
TICKER_URL         = "https://api.coindcx.com/exchange/ticker"

BINANCE_SYMBOLS = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT"}

EXPECTED_CANDLE_COLS  = {"timestamp", "open", "high", "low", "close", "volume"}
EXPECTED_TICKER_FIELDS = {"market", "last_price"}


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


async def check_binance_candles(session: aiohttp.ClientSession) -> bool:
    _section("Binance candle feed (live data source)")
    ok = True

    for sym, binance_sym in BINANCE_SYMBOLS.items():
        params = {"symbol": binance_sym, "interval": "5m", "limit": 2}
        try:
            async with session.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                r.raise_for_status()
                data: list = await r.json()
        except Exception as exc:
            print(f"  [FAIL] {sym}: {exc}")
            ok = False
            continue

        if not data:
            print(f"  [FAIL] {sym}: empty response")
            ok = False
            continue

        # Binance returns array-of-arrays: [open_time, open, high, low, close, volume, ...]
        row = data[0]
        print(f"\n  {sym}  (binance symbol={binance_sym})")
        print(f"    Raw row (first 6 cols) : {row[:6]}")
        print(f"    [0] open_time_ms       : {row[0]}")
        print(f"    [1] open               : {row[1]}")
        print(f"    [2] high               : {row[2]}")
        print(f"    [3] low                : {row[3]}")
        print(f"    [4] close              : {row[4]}")
        print(f"    [5] volume             : {row[5]}")
        if len(row) >= 6 and all(row[i] for i in range(1, 6)):
            print(f"    [OK]   All OHLCV values present")
        else:
            print(f"    [FAIL] Unexpected row structure")
            ok = False

    return ok


async def check_coindcx_ticker(session: aiohttp.ClientSession) -> bool:
    _section("CoinDCX ticker (entry price + USD/INR rate)")
    try:
        async with session.get(
            TICKER_URL, timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            r.raise_for_status()
            data: list = await r.json()
    except Exception as exc:
        print(f"  [FAIL] Could not fetch ticker: {exc}")
        return False

    if not data:
        print("  [FAIL] Empty ticker response")
        return False

    sample = data[0]
    print(f"\n  First entry: {json.dumps(sample, indent=4)}")

    ok = True
    missing = EXPECTED_TICKER_FIELDS - set(sample.keys())
    if missing:
        print(f"  [FAIL] Missing fields: {missing}")
        ok = False
    else:
        print(f"  [OK]   Ticker entries have 'market' and 'last_price'")

    markets = {e.get("market", ""): e.get("last_price") for e in data}

    for expected_mkt in ("BTCUSDT", "ETHUSDT"):
        if expected_mkt in markets:
            print(f"  [OK]   Market '{expected_mkt}' found  (last_price={markets[expected_mkt]})")
        else:
            found = [m for m in markets if m.upper() == expected_mkt]
            if found:
                print(f"  [WARN] Market found as '{found[0]}' — update COINDCX_MARKETS in config.py")
            else:
                print(f"  [FAIL] Market '{expected_mkt}' NOT found in ticker")
                ok = False

    if "USDTINR" in markets:
        print(f"  [OK]   USDT/INR rate: {markets['USDTINR']}")
    else:
        candidates = [m for m in markets if "USDT" in m.upper() and "INR" in m.upper()]
        if candidates:
            print(f"  [WARN] USDT/INR market found as {candidates} — update get_usd_inr_rate()")
        else:
            print(f"  [FAIL] No USDT/INR market found in ticker")
        ok = False

    return ok


async def main() -> None:
    print("Trading bot — live feed verification")
    print("Candles: Binance  |  Ticker/orders: CoinDCX\n")

    async with aiohttp.ClientSession() as session:
        candle_ok = await check_binance_candles(session)
        ticker_ok = await check_coindcx_ticker(session)

    _section("Summary")
    if candle_ok and ticker_ok:
        print("  All checks passed — bot is ready to run.")
    else:
        if not candle_ok:
            print("  [FAIL] Binance candle feed has issues.")
        if not ticker_ok:
            print("  [FAIL] CoinDCX ticker has issues.")

    sys.exit(0 if (candle_ok and ticker_ok) else 1)


if __name__ == "__main__":
    asyncio.run(main())
