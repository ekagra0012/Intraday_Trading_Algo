# Intraday EMA + RSI Algorithmic Trading Strategy

A precise, Python-based backtesting engine implementing a strict Intraday EMA/RSI trading strategy for NSE stocks. This project simulates algorithmic trading execution with 1-minute granularity, strictly adhering to specific assignment requirements regarding indicator logic, risk management, and stock selection.

## 📈 Strategy Overview

The algorithm trades the **Top 10 Stocks by Turnover** on a 10-minute timeframe, using a trend-following approach filtered by a higher-timeframe trend.

### Core Logic
*   **Timeframe**: 10-Minute Candles (Signal), 1-Hour Candles (Trend Filter), 1-Minute Candles (Execution).
*   **Stock Selection**: Dynamically selects top 10 stocks by turnover between 09:15-09:25 AM daily.
*   **Indicators**:
    *   **Short EMAs**: EMA(3) and EMA(10) on 10-min close.
    *   **Trend Filter**: EMA(50) on 1-Hour close.
    *   **Momentum**: RSI(14) (Wilder's Smoothing).

### Entry Rules
*   **Long**: `EMA(3) > EMA(10)` AND `RSI > 60` AND `Close > 1H EMA(50)`.
    *   *Execution*: Buy Stop at the **High** of the signal candle.
*   **Short**: `EMA(3) < EMA(10)` AND `RSI < 30` AND `Close < 1H EMA(50)`.
    *   *Execution*: Sell Stop at the **Lowest Low** of the last 5 minutes (excluding signal minute).

### Risk Management
*   **Stop Loss**: 0.5% per trade.
*   **Target**: 2.0% per trade.
*   **Risk Sizing**: 0.5% of *current* allocated capital exposed per trade.
*   **Trailing Stop**: Activates when profit > 0.5%. Trails price by 0.75%.

---

## 🚀 Setup & Usage

### Prerequisites
*   Python 3.8+
*   pandas
*   numpy

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/ekagra0012/Intraday_Trading_Algo.git
    cd Intraday_Trading_Algo
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy
    ```
3.  Ensure data is in the `data/` directory (CSV files formatted as `dataNSE_YYYYMMDD.csv`).

### Running the Backtest
Execute the primary script:
```bash
python3 intraday_strategy_backtest.py
```

### Outputs
1.  **Console Output**: Prints processing progress per day and a final performance summary (Total Return, Win Rate, Sharpe Ratio, Max Drawdown).
2.  **`trade_log.csv`**: A detailed CSV log of every trade generated, including exact entry/exit times, prices, and exit reasons (Target, StopLoss, TrailingSL).

---

## 🔍 Technical Implementation Details

This codebase was built with strict adherence to "Intraday EMA RSI Strategy Test 1" specifications.

*   **RSI Precision**: Uses Wilder's Smoothing (`alpha=1/14`) rather than a simple moving average, ensuring values match standard trading platforms.
*   **Look-Ahead Bias Prevention**:
    *   **Signal Wait**: Signals are confirmed only after the 10-minute candle closes.
    *   **EMA-50 Context**: The EMA-50 trend filter becomes available only after the first 1-hour candle has completed, meaning trading begins only after sufficient trend context is established (no forward-looking bias).
    *   **Execution Loop**: Trades are executed on *subsequent* 1-minute data flows.
    *   **Short Window**: The "Low of last 5 mins" calculation explicitly excludes the current signal minute to prevent future peeking.
*   **Gap Handling**: Logic intelligently handles gap-ups/downs. If the market opens beyond a trigger price, the `Open` price is used for the fill.
*   **Dynamic Capital**: Position sizing is recalculated after every trade based on the updated capital balance, allowing for compound growth (or drawdown).

## 📊 Performance (Aug 2025 Sample)
*   **Trades Executed**: ~240
*   **Return**: -3.06%
*   **Win Rate**: ~35%
*   *Note: The negative return is primarily driven by the "Top 10 Selection" criteria. High-turnover stocks exhibit extreme volatility, often triggering the tight 0.5% stop loss in choppy conditions.*

---

## 📂 Project Structure

```
├── intraday_strategy_backtest.py  # Main strategy engine
├── trade_log.csv                  # Generated results (not committed)
├── data/                          # Data directory
│   ├── dataNSE_20250801.csv
│   └── ...
└── README.md                      # Documentation
```

## 📝 License
This project is for educational and backtesting purposes.
