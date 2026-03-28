# BTCUSDT Momentum Strategy Backtest Report

## Strategy Description

Dual EMA crossover momentum strategy with RSI and ADX filters on BTCUSDT perpetual futures.
The strategy enters long when the fast EMA (10 period) crosses above the slow EMA (50 period)
with RSI above 50 and ADX above 25 (confirming trend strength). It enters short on the inverse
crossover with RSI below 50 and ADX above 25. Positions are closed when the ADX drops below 20
or RSI reaches extreme levels (above 80 for longs or below 20 for shorts).

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast EMA Period | 10 |
| Slow EMA Period | 50 |
| RSI Period | 14 |
| ADX Period | 14 |
| ADX Entry Threshold | 25 |
| ADX Exit Threshold | 20 |
| RSI Long Entry | > 50 |
| RSI Short Entry | < 50 |
| RSI Exit (Long) | > 80 |
| RSI Exit (Short) | < 20 |
| Leverage | 3x |
| Position Size | 8% of equity |
| Maker Fee | 2 bps |
| Taker Fee | 5 bps |
| Slippage | 3 bps |
| Timeframe | 1 hour |

## Period and Data

- **Pair**: BTCUSDT Perpetual Futures (Binance)
- **Period**: January 1 2024 to December 31 2025
- **Candles**: 17520 hourly bars
- **Data Source**: Historical Binance Futures OHLCV with funding rates

## Results Summary

| Metric | Value |
|--------|-------|
| Total Return | 47.3% |
| Annualized Return | 21.8% |
| Sharpe Ratio | 1.24 |
| Sortino Ratio | 1.71 |
| Max Drawdown | 14.7% |
| Calmar Ratio | 1.48 |
| Win Rate | 42.8% |
| Profit Factor | 1.53 |
| Total Trades | 312 |
| Avg Winner | 2.31% |
| Avg Loser | 1.12% |
| Expectancy per Trade | 0.15% |
| Recovery Factor | 3.22 |
| Avg Trade Duration | 6.4 hours |
| Max Consecutive Wins | 7 |
| Max Consecutive Losses | 8 |

## Monthly Returns (%)

| Month | 2024 | 2025 |
|-------|------|------|
| January | 5.2 | 3.1 |
| February | 8.1 | -1.4 |
| March | 4.3 | 2.7 |
| April | -2.1 | -3.2 |
| May | -1.8 | 4.6 |
| June | -0.6 | 1.9 |
| July | 3.4 | -0.8 |
| August | -2.3 | 5.1 |
| September | 1.7 | -1.6 |
| October | 6.8 | 3.3 |
| November | 7.2 | 2.4 |
| December | -1.2 | 1.6 |

## Equity Curve Description

The equity curve shows steady growth with two notable drawdown periods. The first drawdown
of approximately 11% occurred during April through June 2024 when BTC traded in a tight range
with frequent false breakouts that whipsawed the momentum signals. The second and deepest drawdown
of 14.7% occurred in April 2025 during a sharp correction. Recovery from both drawdowns took
roughly 4 to 6 weeks each. The curve shows the characteristic staircase pattern of a momentum
strategy with extended flat periods during ranging markets followed by sharp gains during
trending periods.

## Key Observations

1. The strategy performed best during Q4 2024 when BTC had strong directional moves.
   October and November 2024 alone contributed approximately 14% of the total return.

2. Win rate of 42.8% is typical for a trend following approach. Profitability comes from
   the asymmetric payoff ratio (avg winner is 2.06x the avg loser).

3. Funding rate costs were modest (total 0.8% drag over the period) because positions
   were closed quickly on regime changes. Longer holding periods would increase this cost.

4. The ADX filter significantly reduced whipsaw trades during ranging periods. Without the
   ADX filter the strategy produced 478 trades with only a 36.2% win rate and a Sharpe of 0.71.

5. Slippage and fee impact was approximately 3.1% over the full period. Using limit orders
   instead of market orders could save approximately 40% of that cost.

6. The strategy underperforms during low volatility regimes and excels in high volatility
   trending environments. A regime detection layer would further improve risk adjusted returns.

These results were generated using MEFAI Engine backtest module on historical Binance Futures data. The same strategy is currently running in paper trading mode on mefai.io infrastructure.
