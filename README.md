# auto_stock_trader

A weekly automated stock-picking and portfolio rebalancing system using KMeans clustering on technical indicators, with trade execution via the Alpaca broker API.

## How it works

1. **`core.py`** — runs every Friday after market close. Downloads S&P 500 OHLCV data, computes technical indicators (RSI, Bollinger Z, ATR%, MACD histogram, Garman-Klass volatility), clusters stocks using KMeans, selects the best cluster per strategy config, and optimizes portfolio weights. Writes `picks.json`.

2. **`broker.py`** — runs every Monday before market open. Reads `picks.json`, merges all strategy configs by averaging their weights, and rebalances the Alpaca account: liquidates stale positions, sells overweight holdings (market orders), and buys underweight holdings (limit orders with a 50bps band).

3. **`configs.json`** — defines named strategy configurations. Each config specifies clustering parameters and which indicator to rank clusters by.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Alpaca credentials:

```bash
cp .env.example .env
```

## Running locally

```bash
# Generate picks (Friday)
python core.py

# Execute trades (Monday)
python broker.py
```

## Configuration

Edit `configs.json` to define your strategies. Each config object supports:

| Field | Description |
|---|---|
| `name` | Unique strategy label |
| `lookback_days` | Days of OHLCV history to download |
| `seed` | Random seed for KMeans reproducibility |
| `k` | Number of clusters |
| `min_cluster_size` | Minimum stocks a cluster must have to be eligible |
| `best_by` | Indicator to rank clusters by (`rsi`, `atr_pct`, `macd_hist`, etc.) |
| `ascending` | `true` = pick lowest value cluster, `false` = pick highest |
| `agg` | Aggregation for cluster ranking: `mean` or `median` |
| `n_picks` | Number of stocks to select from the winning cluster |
| `scaler` | Feature scaler: `robust`, `standard`, or `minmax` |

## Environment variables

| Variable | Description |
|---|---|
| `ALPACA_KEY` | Alpaca API key ID |
| `ALPACA_SECRET` | Alpaca API secret key |
| `ALPACA_PAPER` | Set to `false` for live trading (default: `true`) |
| `PICKS_BUCKET` | S3 bucket name for picks handoff (optional, for cloud deployment) |

## Deployment (AWS ECS Fargate)

For automated scheduling, deploy both scripts as separate ECS Fargate tasks triggered by EventBridge Scheduler cron rules. Use an S3 bucket (set `PICKS_BUCKET`) to pass `picks.json` between the two tasks.

- `core` task: `python -m core` — scheduled Friday ~4:30 PM ET
- `broker` task: `python -m broker` — scheduled Monday ~9:20 AM ET
