# Intraday EMA and RSI Algorithmic Trading Strategy

## Project Overview
This repository contains a high-precision backtesting engine for an Intraday Algorithmic Trading Strategy focused on NSE (National Stock Exchange) equities. The system mimics a real-world execution environment to validate a trend-following logic based on Exponential Moving Averages (EMA) and the Relative Strength Index (RSI).

The core objective of this project is to implement a strict, verifiable trading strategy that eliminates common backtesting pitfalls such as look-ahead bias and unrealistic fill assumptions.

## Strategy Logic

The strategy operates on a 10-minute timeframe for signal generation, filtered by a 1-hour major trend, with execution handled on a 1-minute granular level.

### 1. Stock Selection
*   **Universe**: Dynamically selects the Top 10 stocks by turnover (Value Traded) daily.
*   **Window**: Selection is performed using data strictly between 09:15 AM and 09:25 AM.

### 2. Indicators and Setup
*   **Fast EMA**: 3-period EMA (calculated on 10-minute Close).
*   **Slow EMA**: 10-period EMA (calculated on 10-minute Close).
*   **Trend Filter**: 50-period EMA (calculated on 1-hour Close).
*   **Momentum**: 14-period RSI (calculated using Wilder's Smoothing method).

### 3. Entry Conditions
*   **Long Signal**:
    1.  EMA(3) crosses above EMA(10).
    2.  RSI(14) is above 60.
    3.  10-minute Close is above the 1-hour EMA(50).
    *   **Execution**: Place a Buy Stop order at the **High** of the signal candle.

*   **Short Signal**:
    1.  EMA(3) crosses below EMA(10).
    2.  RSI(14) is below 30.
    3.  10-minute Close is below the 1-hour EMA(50).
    *   **Execution**: Place a Sell Stop order at the **Lowest Low** of the previous 5 minutes (excluding the signal candle).

### 4. Risk Management
*   **Position Sizing**: Allocates 0.5% of the *current* total capital per trade risk.
    *   *Formula*: `Quantity = (Total Capital * 0.005) / (Entry Price * 0.005)`
*   **Stop Loss**: Fixed at 0.5% from Entry Price.
*   **Take Profit**: Fixed at 2.0% from Entry Price.
*   **Trailing Stop**:
    *   **Activation**: When floating profit exceeds 0.5%.
    *   **Trailing Step**: Trails the market price by 0.75%.

## Technical Implementation

The system is engineered to ensure mathematical correctness and historical accuracy.

### Precision Mechanisms
1.  **RSI Calculation**: Implements Wilder's Smoothing (alpha = 1/14), ensuring consistency with standard trading platforms, differing from simple arithmetic mean calculations.
2.  **Look-Ahead Bias Prevention**:
    *   **Signal Validation**: Trade signals are quantified only after the close of the 10-minute bar.
    *   **Trend Context**: The EMA-50 trend filter utilizes 1-hour candles. To prevent forward-looking bias, the strategy waits for the completion of the first full 1-hour candle before validating any trend, ensuring the "Trend Filter" is based on confirmed historical data.
3.  **Realistic Execution Engine**:
    *   **Granularity**: While signals are generated on 10-minute bars, price matching occurs on 1-minute bars to simulate realistic intraday volatility.
    *   **Short Window Logic**: The "Low of last 5 minutes" calculation strictly excludes the signal generation minute `t` and looks at `[t-5, t-1]`.
    *   **Gap Logic**: Explicitly handles gap openings. If a next-bar Open jumps past a trigger price, the system executes at the Open price rather than the theoretical limit price.

## Installation and Usage

### Prerequisites
*   Python 3.8 or higher
*   Pandas library
*   NumPy library

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/ekagra0012/Intraday_Trading_Algo.git
    cd Intraday_Trading_Algo
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy
    ```
3.  Verify that the `data/` directory contains standard NSE data CSV files (e.g., `dataNSE_20250801.csv`).

### Execution
Run the primary backtesting script:
```bash
python3 intraday_strategy_backtest.py
```

### Output Artifacts
The script produces two primary outputs:

1.  **Terminal Summary**:
    *   Displays daily processing progress.
    *   Provides final metrics: Total Return, Final Capital, Win Rate, Max Drawdown, and Sharpe Ratio.

2.  **Trade Log (trade_log.csv)**:
    A granular CSV file recording every executed trade. Columns include:
    *   `EntryTime` / `ExitTime`: Exact minute of fill.
    *   `EntryPrice` / `ExitPrice`: The actual filled price (accounting for gaps).
    *   `ExitType`: The logic that closed the trade (`Target`, `StopLoss`, `TrailingSL`, or `EOD_SquareOff`).
    *   `PnL` / `Return`: Net profit/loss for that specific position.

## Disclaimer
This project is intended for educational and research purposes. Algorithmic trading involves significant risk. The backtest results provided by this system reflect historical performance on specific datasets and do not guarantee future results.
