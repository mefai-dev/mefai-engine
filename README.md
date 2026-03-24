# MEFAI Engine

Institutional grade AI trading engine for crypto perpetual futures. Built from scratch with a multi model ensemble architecture that combines gradient boosting and deep learning and reinforcement learning and NLP sentiment analysis into a unified decision making pipeline.

MEFAI Engine is not a simple trading bot. It is a full stack autonomous trading system with a multi agent orchestration layer where specialized agents (analyst and risk manager and sentinel and executor) collaborate to make and execute trading decisions.

## What Makes This Different

**Multi Model Ensemble with Regime Awareness** The meta learner dynamically adjusts model trust weights based on the current market regime. A transformer model might be heavily trusted during trending markets but downweighted in ranging conditions. XGBoost handles feature rich classification while the RL agent optimizes position sizing through simulated experience.

**Signal Evolution Tracking** Every trading signal is tracked over time. When new data arrives the system evaluates whether the original thesis has strengthened or weakened or been falsified. This prevents holding positions where the underlying logic no longer holds.

**Circuit Breaker Architecture** Three levels of protection: per exchange circuit breakers for API failures and a trading circuit breaker for consecutive losses and a sentinel agent that monitors for flash crashes and volume anomalies and funding rate extremes.

**55+ Feature Pipeline** Vectorized numpy based indicator computation covering trend (SMA EMA HMA ADX Aroon) and momentum (RSI MACD Stochastic CCI Williams%R ROC MFI) and volatility (ATR Bollinger Keltner historical vol) and volume (OBV VWAP CVD Chaikin Force Index) and microstructure (order book imbalance spread weighted mid price) and on chain (funding rate OI liquidation intensity) and regime detection (HMM inspired multi factor classification).

## Architecture

```
                        MEFAI Engine Architecture

    ┌─────────────────────────────────────────────────────────┐
    │                     DATA LAYER                          │
    │                                                         │
    │  Exchange WS ──→ Collector ──→ TimescaleDB              │
    │  News RSS    ──→ Aggregator ──→ Redis Cache             │
    │  TV Webhook  ──→ Receiver   ──→ Signal Queue            │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │                  FEATURE PIPELINE                        │
    │                                                         │
    │  Registry ──→ DAG Resolver ──→ Vectorized Compute       │
    │  55+ indicators across 7 categories                     │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │                    MODEL LAYER                           │
    │                                                         │
    │  XGBoost/LightGBM     ──→ Direction prediction          │
    │  Temporal Transformer  ──→ Multi-horizon forecast        │
    │  PPO Agent             ──→ Position sizing               │
    │  FinBERT Sentiment     ──→ News sentiment score          │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │               STRATEGY + META-LEARNER                    │
    │                                                         │
    │  Regime Detection ──→ Model Weight Adjustment            │
    │  Signal Generation ──→ Confidence Scoring                │
    │  Rate Limiting ──→ Max 4 signals/hour                    │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │              MULTI-AGENT ORCHESTRATOR                    │
    │                                                         │
    │  Analyst Agent    ──→ Proposes trades with confluence    │
    │  Risk Agent       ──→ Approves/reduces/vetoes            │
    │  Sentinel Agent   ──→ Monitors anomalies + can halt     │
    │  Signal Tracker   ──→ Tracks thesis evolution            │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │                 EXECUTION + RISK                         │
    │                                                         │
    │  Risk Manager ──→ Position limits + drawdown + daily     │
    │  Circuit Breaker ──→ Auto halt on loss streak            │
    │  Executor ──→ Market / TWAP / smart routing              │
    │  Reconciler ──→ Exchange position verification           │
    └─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/mefai-io/mefai-engine.git
cd mefai-engine

# Install
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env with your exchange API keys

# Start paper trading with API
mefai-engine paper --port 8080

# Or start with Docker
docker compose up -d
```

The API is available at `http://localhost:8080/docs` with full interactive documentation.

## API Reference

31 endpoints covering every aspect of the trading lifecycle.

### System
```
GET  /api/v1/health              Engine health check
GET  /api/v1/status              Full system status
GET  /api/v1/config              Current configuration
```

### Trading
```
GET  /api/v1/positions           All open positions
GET  /api/v1/balance             Account balance
GET  /api/v1/ticker/{symbol}     Real-time price
GET  /api/v1/orderbook/{symbol}  Order book depth
GET  /api/v1/funding/{symbol}    Funding rate + annualized
GET  /api/v1/signals             Tracked signals + evolution
POST /api/v1/orders              Place manual order
DELETE /api/v1/orders/{id}       Cancel order
POST /api/v1/positions/close     Close position at market
```

### Market Data
```
GET  /api/v1/candles/{symbol}    Historical OHLCV
GET  /api/v1/features/{symbol}   Computed indicators (55+)
GET  /api/v1/news                Aggregated crypto news
GET  /api/v1/sentiment/{symbol}  FinBERT sentiment score
```

### Models
```
GET  /api/v1/models              All model statuses
GET  /api/v1/models/{id}         Model detail + feature importance
POST /api/v1/models/predict      Run single prediction
POST /api/v1/models/train        Trigger model training
POST /api/v1/models/save         Save models to disk
POST /api/v1/models/load         Load models from disk
```

### Backtest
```
POST /api/v1/backtest/run        Run vectorized backtest
```

Returns Sharpe ratio and Sortino ratio and Calmar ratio and max drawdown and win rate and profit factor and total trades and expectancy and recovery factor with full equity curve.

### Monitoring
```
GET  /api/v1/metrics             PnL and performance metrics
GET  /api/v1/report              Daily performance report
GET  /api/v1/risk                Risk limits + circuit breaker status
POST /api/v1/report/send-telegram  Send report via Telegram
POST /api/v1/risk/circuit-breaker/reset  Manual circuit breaker reset
```

### Webhooks
```
POST /api/v1/webhook/tradingview   TradingView alert receiver
POST /api/v1/webhook/custom        Generic signal webhook
```

### WebSocket
```
WS   /ws/live    Real-time stream: ticker + positions + signals + PnL
```

## ML Models

### XGBoost / LightGBM Direction Classifier
3 class classification (LONG / SHORT / FLAT) with automatic class balancing and walk forward validation to prevent lookahead bias. Outputs direction probability and predicted magnitude. Feature importance tracking for explainability.

### Temporal Fusion Transformer
Multi head self attention with positional encoding and gated residual connections. Trained on sequences of 128 candles with all computed features. Multi horizon output with confidence estimation. Cosine annealing learning rate schedule with early stopping.

### PPO Position Sizing Agent
Custom Gymnasium trading environment simulating perpetual futures with realistic fees and funding rates and slippage. The PPO agent learns to optimize position size through simulated experience. Reward function combines risk adjusted returns with drawdown penalties.

### FinBERT Sentiment Analyzer
Pre trained financial BERT model scoring crypto news from negative 1.0 (bearish) to positive 1.0 (bullish). Batch processing with caching. Aggregates multiple news sources (CryptoPanic API and RSS feeds from CoinTelegraph and CoinDesk and Decrypt) into a single actionable sentiment signal.

## Feature Categories

| Category | Count | Examples |
|----------|-------|---------|
| Trend | 12 | SMA EMA HMA ADX Aroon |
| Momentum | 10 | RSI MACD Stochastic CCI Williams%R ROC MFI |
| Volatility | 8 | ATR Bollinger Keltner Historical Vol |
| Volume | 8 | OBV VWAP CVD Chaikin Money Flow Force Index |
| Microstructure | 6 | Book Imbalance Spread Weighted Mid Price |
| On Chain | 5 | Funding Rate OI Change Long/Short Ratio Liquidation |
| Regime | 4 | Trend Strength Volatility Regime Mean Reversion Score |
| Sentiment | 4 | News Score Sentiment Momentum Price Divergence |

## Built-in Strategies

### Momentum Strategy
Trend following with EMA crossover confirmation. Requires EMA 10 above/below EMA 50 with RSI in non extreme range and positive/negative MACD histogram and ADX above 20 for trending market confirmation. Confluence scoring from multiple indicator agreement.

### Mean Reversion Strategy
Bollinger Band extreme entries with RSI confirmation. Enters when price touches lower/upper band with RSI below 25 or above 75 in ranging markets (ADX below 25). Considers funding rate for squeeze potential. Smaller position sizes due to higher risk profile.

## Risk Management

The risk manager is the central authority. Every signal must pass through risk evaluation before execution.

**Position Limits** Maximum single position size as percentage of equity (default 10%). Maximum total exposure across all positions (default 30%).

**Drawdown Protection** Maximum daily loss limit (default 3% of equity). Maximum drawdown from equity peak triggers circuit breaker (default 10%).

**Circuit Breaker** Three states: CLOSED (normal trading) and OPEN (all trading halted) and HALF_OPEN (one test trade allowed). Triggers on consecutive losing trades (default 5) or max drawdown breach. Automatic cooldown with configurable duration.

**Sentinel Monitoring** Real time anomaly detection for flash crashes (5%+ price moves) and volume spikes (5x average) and extreme funding rates and wide spreads. Multiple simultaneous anomalies trigger automatic trading halt.

## Configuration

Single YAML file with environment variable overrides for secrets.

```yaml
engine:
  mode: paper
  symbols: ["BTCUSDT"]

exchanges:
  binance:
    enabled: true
    testnet: true

risk:
  max_position_pct: 10.0
  max_drawdown_pct: 10.0
  max_daily_loss_pct: 3.0

strategy:
  meta_learner:
    min_confidence: 0.65
    regime_filter: true
```

Full annotated configuration in `configs/default.yaml`.

## Project Structure

```
mefai-engine/
├── src/mefai_engine/
│   ├── exchange/          Exchange abstraction (Binance + WebSocket)
│   ├── data/              Collection + TimescaleDB + Redis + News + Webhook
│   ├── features/          55+ indicators with DAG based pipeline
│   ├── models/
│   │   ├── classical/     XGBoost / LightGBM
│   │   ├── deep/          Temporal Transformer
│   │   ├── rl/            PPO position sizing (Gymnasium env)
│   │   └── nlp/           FinBERT sentiment
│   ├── strategy/          Meta-learner + built-in strategies
│   ├── risk/              Risk manager + circuit breaker + PnL tracker
│   ├── execution/         Order executor (Market + TWAP)
│   ├── agents/            Multi-agent orchestrator + signal tracker
│   ├── backtest/          Vectorized backtesting engine
│   ├── monitoring/        Telegram alerts + report generator
│   ├── api/               FastAPI REST + WebSocket (31 endpoints)
│   ├── cli.py             CLI: run / paper / api / backtest / train
│   └── config.py          YAML + env var configuration
├── configs/               Default + backtest + paper configs
├── docker-compose.yml     Engine + TimescaleDB + Redis
├── Dockerfile             Multi-stage production build
└── pyproject.toml         PEP 621 with optional ML dependencies
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Database | TimescaleDB (PostgreSQL with hypertables) |
| Cache | Redis with hiredis |
| ML Classical | XGBoost + LightGBM + scikit-learn |
| Deep Learning | PyTorch |
| RL | Stable-Baselines3 + Gymnasium |
| NLP | HuggingFace Transformers (FinBERT) |
| Indicators | NumPy (vectorized) |
| DataFrames | Polars |
| Config | Pydantic Settings + YAML |
| CLI | Typer + Rich |
| Logging | structlog (JSON) |
| Exchange | aiohttp + websockets (native async) |

## Requirements

Python 3.11+
PostgreSQL 16+ with TimescaleDB extension (optional)
Redis 7+ (optional)
Exchange API keys (Binance Futures recommended)

Install core only (no ML):
```bash
pip install -e .
```

Install with all ML models:
```bash
pip install -e ".[all]"
```

## Docker Deployment

```bash
docker compose up -d
```

This starts the engine with TimescaleDB and Redis. Configure via `.env` file.

## License

Apache 2.0
