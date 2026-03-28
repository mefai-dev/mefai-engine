# BTCUSDT Mean Reversion Strategy Backtest Report

## Strategy Description

Bollinger Band mean reversion strategy with RSI confirmation on BTCUSDT perpetual futures.
The strategy enters long when price touches or drops below the lower Bollinger Band (2 standard
deviations) with RSI below 30 (oversold). It enters short when price touches or exceeds the
upper Bollinger Band with RSI above 70 (overbought). Positions are closed when price reverts
to the 20 period SMA (middle band) or when a time based stop of 12 hours is reached. A
volatility filter using ATR prevents entries during extreme volatility spikes.

## Parameters

| Parameter | Value |
|-----------|-------|
| Bollinger Period | 20 |
| Bollinger Std Dev | 2.0 |
| RSI Period | 14 |
| RSI Oversold | 30 |
| RSI Overbought | 70 |
| ATR Period | 14 |
| ATR Max Multiplier | 3.0 |
| Leverage | 2x |
| Position Size | 6% of equity |
| Maker Fee | 2 bps |
| Taker Fee | 5 bps |
| Slippage | 3 bps |
| Time Stop | 12 hours |
| Timeframe | 15 minutes |

## Period and Data

- **Pair**: BTCUSDT Perpetual Futures (Binance)
- **Period**: January 1 2024 to December 31 2025
- **Candles**: 70080 bars at 15 minute resolution
- **Data Source**: Historical Binance Futures OHLCV with funding rates

## Results Summary

| Metric | Value |
|--------|-------|
| Total Return | 31.6% |
| Annualized Return | 14.9% |
| Sharpe Ratio | 1.08 |
| Sortino Ratio | 1.42 |
| Max Drawdown | 11.3% |
| Calmar Ratio | 1.32 |
| Win Rate | 58.4% |
| Profit Factor | 1.31 |
| Total Trades | 487 |
| Avg Winner | 0.87% |
| Avg Loser | 0.91% |
| Expectancy per Trade | 0.065% |
| Recovery Factor | 2.80 |
| Avg Trade Duration | 3.2 hours |
| Max Consecutive Wins | 11 |
| Max Consecutive Losses | 6 |

## Monthly Returns (%)

| Month | 2024 | 2025 |
|-------|------|------|
| January | 1.8 | 2.1 |
| February | -0.4 | 1.5 |
| March | 2.3 | 1.9 |
| April | 3.1 | 3.4 |
| May | 2.4 | -0.7 |
| June | 1.6 | 2.2 |
| July | -1.2 | 1.8 |
| August | 0.9 | -2.3 |
| September | 2.0 | 1.4 |
| October | -0.3 | 0.8 |
| November | -1.7 | 1.1 |
| December | 2.1 | 1.3 |

## Equity Curve Description

The equity curve is notably smoother than a typical momentum strategy due to the higher win rate
and shorter holding periods. Drawdowns are more frequent but shallower. The maximum drawdown of
11.3% occurred in August 2025 during a period of unusually high volatility where price repeatedly
broke through Bollinger Bands without reverting. The ATR filter prevented some of these bad entries
but the extreme moves still generated losses on positions already open. Outside of these volatile
episodes the curve shows consistent incremental gains with a low correlation to the underlying
BTC direction.

## Key Observations

1. The mean reversion strategy shows negative correlation with the momentum strategy
   (approximately negative 0.35 on monthly returns). Combining both strategies in a portfolio
   would significantly reduce overall drawdown.

2. Win rate of 58.4% is strong for a mean reversion approach. The strategy profits from the
   high frequency of successful reversions rather than from outsized winners.

3. The strategy performed poorly during strong trending months (November 2024 and August 2025)
   as expected. Mean reversion strategies inherently suffer during extended directional moves.

4. The time stop (12 hours) was critical for limiting losses. Without the time stop the max
   drawdown increased to 19.8% with only marginal improvement in total return.

5. Funding rate costs were minimal (total 0.3% drag) because of the short holding periods
   averaging 3.2 hours.

6. The ATR volatility filter rejected approximately 23% of potential entries and improved the
   Sharpe ratio from 0.82 to 1.08. This filter is the single most impactful parameter.

7. Execution on the 15 minute timeframe requires fast and reliable connectivity. Latency above
   200ms would degrade performance meaningfully due to the tight entry and exit windows.

These results were generated using MEFAI Engine backtest module on historical Binance Futures data. The same strategy is currently running in paper trading mode on mefai.io infrastructure.
