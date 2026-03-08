"""
broker.py — Load picks.json and rebalance via Alpaca.

Reads picks.json from the working directory, merges all configs by averaging
weights, and submits orders to Alpaca.

Environment variables required:
  ALPACA_KEY     — Alpaca API key ID
  ALPACA_SECRET  — Alpaca API secret key
  ALPACA_PAPER   — set to "false" to trade live (defaults to paper)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from typing import Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

log = logging.getLogger("broker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PICKS_BUCKET = os.environ.get("PICKS_BUCKET")
PICKS_KEY    = "picks/latest.json"
PICKS_PATH   = "picks.json"          # local fallback
MAX_GROSS    = 1.0
PER_SYMBOL_CAP = 0.10
PAPER        = os.environ.get("ALPACA_PAPER", "true").lower() != "false"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    target_weight: float
    price_band_bps: int = 50


# ---------------------------------------------------------------------------
# Alpaca adapter
# ---------------------------------------------------------------------------

class AlpacaAdapter:
    def __init__(self):
        key    = os.environ["ALPACA_KEY"]
        secret = os.environ["ALPACA_SECRET"]
        self.trading = TradingClient(key, secret, paper=PAPER)
        self.data    = StockHistoricalDataClient(key, secret)

    def account(self):
        return self.trading.get_account()

    def positions(self) -> Dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}

    def last_price(self, symbol: str) -> Optional[float]:
        resp  = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        trade = resp[symbol] if isinstance(resp, dict) else resp
        return float(trade.price) if trade and trade.price else None

    def submit_market(self, symbol: str, qty: int, side: OrderSide, tif: TimeInForce):
        if qty <= 0:
            return
        return self.trading.submit_order(
            MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=tif)
        )

    def submit_limit(self, symbol: str, qty: int, side: OrderSide, tif: TimeInForce, limit_price: float):
        if qty <= 0:
            return
        return self.trading.submit_order(
            LimitOrderRequest(symbol=symbol, qty=qty, side=side,
                              time_in_force=tif, limit_price=round(limit_price, 2))
        )


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------

def rebalance(signals: Dict[str, Signal]) -> None:
    """Rebalance the Alpaca account to match target weights.

    Sells are submitted as market orders first so proceeds fund buys.
    Buys are submitted as limit orders with a small price band.
    """
    brok   = AlpacaAdapter()
    acct   = brok.account()
    equity = float(acct.equity)
    log.info("Account equity: $%.2f", equity)

    # Scale down if total gross exceeds cap
    tot = sum(abs(s.target_weight) for s in signals.values())
    if tot > MAX_GROSS + 1e-9:
        scale = MAX_GROSS / tot
        for s in signals.values():
            s.target_weight *= scale
        log.warning("Scaled weights by %.4f to respect max gross %.2f", scale, MAX_GROSS)

    # Per-symbol cap
    for s in signals.values():
        s.target_weight = min(PER_SYMBOL_CAP, s.target_weight)

    pos = brok.positions()

    sells, buys = [], []

    # Liquidate any position not present in this week's signals
    for sym, qty in pos.items():
        if sym not in signals and qty > 0:
            log.info("%-6s  not in picks — liquidating %.0f shares.", sym, qty)
            sells.append((sym, int(qty), OrderSide.SELL))

    for sym, sig in signals.items():
        px = brok.last_price(sym)
        if not px:
            log.warning("No price for %s — skipping.", sym)
            continue

        tgt_shares = int(math.floor(sig.target_weight * equity / px))
        cur_shares = int(pos.get(sym, 0))
        delta      = tgt_shares - cur_shares

        if delta == 0:
            log.info("%-6s  no change (%.0f shares)", sym, cur_shares)
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty  = abs(delta)

        if side is OrderSide.SELL:
            sells.append((sym, qty, side))
        else:
            limit = px * (1 + sig.price_band_bps / 1e4)
            buys.append((sym, qty, side, limit))

    for sym, qty, side in sells:
        try:
            brok.submit_market(sym, qty, side, TimeInForce.DAY)
            log.info("MKT  SELL %s x%d", sym, qty)
            time.sleep(0.2)
        except Exception:
            log.exception("Sell failed: %s x%d", sym, qty)

    for sym, qty, side, limit in buys:
        try:
            brok.submit_limit(sym, qty, side, TimeInForce.DAY, limit)
            log.info("LMT  BUY  %s x%d @ %.2f", sym, qty, limit)
            time.sleep(0.2)
        except Exception:
            log.exception("Buy failed: %s x%d", sym, qty)


# ---------------------------------------------------------------------------
# picks.json loader — merges all configs by averaging weights
# ---------------------------------------------------------------------------

def load_merged_weights() -> Dict[str, float]:
    if PICKS_BUCKET:
        import boto3
        body = boto3.client("s3").get_object(Bucket=PICKS_BUCKET, Key=PICKS_KEY)["Body"].read()
        data = json.loads(body)
        log.info("Loaded picks from s3://%s/%s", PICKS_BUCKET, PICKS_KEY)
    else:
        with open(PICKS_PATH) as fh:
            data = json.load(fh)
        log.info("Loaded picks from %s", PICKS_PATH)

    log.info("picks.json generated at %s", data.get("generated_at", "unknown"))

    totals: Dict[str, float] = {}
    n_configs = len(data["results"])
    for entry in data["results"]:
        for p in entry["picks"]:
            totals[p["ticker"]] = totals.get(p["ticker"], 0.0) + p["weight"]

    avg = {t: w / n_configs for t, w in totals.items()}
    total = sum(avg.values())
    if total > 0:
        avg = {t: w / total for t, w in avg.items()}

    log.info("Merged %d configs → %d unique tickers.", n_configs, len(avg))
    return avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    weights = load_merged_weights()
    signals = {sym: Signal(symbol=sym, target_weight=w) for sym, w in weights.items()}
    rebalance(signals)
