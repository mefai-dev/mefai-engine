# Changelog

All notable changes to MEFAI Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-29

### Added
- Walk-forward optimization engine with rolling train/validate/test windows
- Kelly criterion position sizing with fractional Kelly and confidence scaling
- VaR and CVaR risk calculation with Monte Carlo simulation
- Correlation risk manager with DCC-GARCH dynamic correlation tracking
- Liquidity risk assessment using order book depth analysis
- HMM-based market regime detection (trending / ranging / high vol / low vol)
- Microstructure features: book imbalance and cumulative volume delta and spread
- On-chain features: funding rate normalization and open interest change
- Sentiment features: news sentiment score and social volume and fear/greed index
- Feature drift detection with PSI and KS tests for model monitoring
- Hyperparameter optimization module using Optuna with walk-forward validation
- Multi-tenant system with API key management and plan-based access control
- Billing integration with Stripe for subscription management
- Audit logging system with file rotation and memory buffer
- 54 registered technical features across trend and momentum and volatility and volume categories
- Vectorized backtest engine with fee and slippage and funding rate simulation
- Meta-learner ensemble that combines gradient boost and transformer and RL and sentiment models
- CI/CD pipeline with GitHub Actions for linting and testing and type checking
- Comprehensive unit test suite with 20+ tests
- Example scripts for backtesting and custom strategies and risk management
- Apache 2.0 license
- Security policy and contributing guidelines

### Changed
- Moved stripe and prometheus-client and scipy and optuna to optional commercial dependency group
- Reorganized optional dependencies into ml and commercial and dev groups

## [0.1.1] - 2026-02-15

### Fixed
- Hardened API key generation with cryptographically secure random bytes
- Added rate limiting to all public API endpoints
- Removed hardcoded credentials from configuration defaults
- Fixed SQL injection vulnerability in tenant lookup queries
- Added input validation to all exchange API proxy endpoints
- Enforced HTTPS for all external API calls
- Added CORS configuration with explicit allowed origins
- Fixed WebSocket authentication bypass in monitoring endpoint

### Security
- Addressed findings from external penetration test
- All secrets now loaded exclusively from environment variables
- Added automated secret scanning to CI pipeline

## [0.1.0] - 2026-01-20

### Added
- Initial project structure with modular architecture
- Core type definitions: Candle and Signal and Prediction and RiskDecision and BacktestResult
- Exchange abstraction layer supporting Binance and OKX and Bybit
- Data collection pipeline with WebSocket and REST fallback
- Technical analysis feature pipeline with 35 indicators
- Gradient boosting model integration (XGBoost / LightGBM)
- Temporal Fusion Transformer for sequence prediction
- PPO reinforcement learning agent using Stable Baselines3
- FinBERT sentiment analysis from news feeds
- Risk manager with position limits and daily loss and drawdown checks
- Circuit breaker for automated trading halt
- PnL tracker with equity curve and drawdown monitoring
- TWAP and VWAP and Iceberg execution algorithms
- FastAPI REST API with OpenAPI documentation
- WebSocket feed for real-time data streaming
- TimescaleDB integration for time series storage
- Redis integration for caching and pub/sub
- Telegram notification bot
- Prometheus metrics exporter
- YAML configuration with environment variable overrides
- CLI interface via Typer
- Docker and Kubernetes deployment manifests
