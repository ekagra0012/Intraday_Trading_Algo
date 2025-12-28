import pandas as pd
import numpy as np
import glob
from datetime import timedelta

# ===============================
# CONFIG
# ===============================
BASE_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.005          # 0.5%
STOP_LOSS_PCT = 0.005           # 0.5%
TARGET_PCT = 0.02               # 2%
TRAIL_TRIGGER = 0.005           # +0.5%
TRAIL_STEP = 0.0075             # 0.75%

DATA_FILES = glob.glob("data/dataNSE_*.csv")  # all uploaded files


# ===============================
# HELPER FUNCTIONS
# ===============================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity/roll_max - 1
    return dd.min()

def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)


# ===============================
# LOAD & COMBINE DATA
# ===============================
df_list = []
for f in DATA_FILES:
    d = pd.read_csv(f)
    d["time"] = pd.to_datetime(d["time"])
    df_list.append(d)

data = pd.concat(df_list).sort_values(["ticker", "time"])
data = data.rename(columns={
    "ticker":"symbol"
}).reset_index(drop=True)

print(f"Loaded {len(data)} rows across {len(DATA_FILES)} days")


# ===============================
# BACKTEST
# ===============================
trade_log = []

for day, day_df in data.groupby(data["time"].dt.date):

    print(f"\nProcessing day: {day}")

    # ---- 9:25 STOCK SELECTION (TOP 10 TURNOVER) ----
    first_10 = day_df[
        (day_df["time"].dt.time >= pd.to_datetime("09:15").time()) &
        (day_df["time"].dt.time <= pd.to_datetime("09:25").time())
    ]

    turnover = (
        first_10.assign(turnover=lambda x: x["close"] * x["volume"])
        .groupby("symbol")["turnover"].sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    day_df = day_df[day_df["symbol"].isin(turnover)]

    # ---- PER STOCK BACKTEST ----
    for symbol, g in day_df.groupby("symbol"):

        g = g.set_index("time")
        m1 = g.copy()

        # 10-minute candles
        m10 = m1.resample("10min").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()

        # 1-hour candles
        h1 = m1.resample("1h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()
        h1["ema50"] = ema(h1["close"], 50)

        # indicators on 10-min
        m10["ema3"] = ema(m10["close"], 3)
        m10["ema10"] = ema(m10["close"], 10)
        m10["rsi14"] = rsi(m10["close"], 14)

        # align EMA50
        # align EMA50
        m10 = m10.merge(h1["ema50"], left_index=True, right_index=True, how="left").ffill()

        position = None
        capital = BASE_CAPITAL

        for t, row in m10.iterrows():
            price = row["close"]

            # =========================
            # ENTRY LOGIC
            # =========================
            if position is None:

                # ---- LONG ENTRY ----
                if row["ema3"] > row["ema10"] and row["rsi14"] > 60 and price > row["ema50"]:
                    entry_price = row["high"]

                    risk_per_share = entry_price * STOP_LOSS_PCT
                    capital_risk_amt = capital * RISK_PER_TRADE
                    qty = max(1, int(capital_risk_amt / risk_per_share))

                    sl = entry_price * (1 - STOP_LOSS_PCT)
                    target = entry_price * (1 + TARGET_PCT)

                    position = "LONG"
                    entry_time = t
                    trail_active = False
                    highest_price_since_entry = entry_price

                    continue

                # ---- SHORT ENTRY ----
                if row["ema3"] < row["ema10"] and row["rsi14"] < 30 and price < row["ema50"]:
                    # Strict Requirement: "Low of last 5 one-minute candles" (completed before signal)
                    # Use candles from t-5min to t-1min (inclusive)
                    last5_low = m1.loc[t - timedelta(minutes=5): t - timedelta(minutes=1)]["low"].min()
                    entry_price = last5_low

                    risk_per_share = entry_price * STOP_LOSS_PCT
                    capital_risk_amt = capital * RISK_PER_TRADE
                    qty = max(1, int(capital_risk_amt / risk_per_share))

                    sl = entry_price * (1 + STOP_LOSS_PCT)
                    target = entry_price * (1 - TARGET_PCT)

                    position = "SHORT"
                    entry_time = t
                    trail_active = False
                    lowest_price_since_entry = entry_price

                    continue

            # =========================
            # EXIT LOGIC
            # =========================
            if position is not None:
                
                # Check metrics for trailing
                # Ideally we check on 1-min data within the 10-min bar for strict accuracy, 
                # but for now we apply logic on the 10-min 'price' (close) to keep consistent with loop.
                # If 'price' is close, we might miss intraday spikes. 
                # To be safer/more strict, let's check 'high'/'low' of current bar?
                # The loop iterates 10-min bars. 'row' has 'high', 'low', 'close'.
                
                current_high = row["high"]
                current_low = row["low"]
                
                # Update max/min price since entry for trailing calculation
                if position == "LONG":
                     highest_price_since_entry = max(highest_price_since_entry, current_high)
                else:
                     lowest_price_since_entry = min(lowest_price_since_entry, current_low)

                # TRAILING ACTIVATION & UPDATE
                # Rule: Start continuous trailing by 0.75% after trade is in profit of 0.5%
                
                if position == "LONG":
                    # Check activation
                    if not trail_active:
                        if highest_price_since_entry >= entry_price * (1 + TRAIL_TRIGGER):
                            trail_active = True
                    
                    if trail_active:
                        # Trail 0.75% from highest price reached
                        proposed_sl = highest_price_since_entry * (1 - TRAIL_STEP)
                        sl = max(sl, proposed_sl)

                elif position == "SHORT":
                     # Check activation
                    if not trail_active:
                        if lowest_price_since_entry <= entry_price * (1 - TRAIL_TRIGGER):
                            trail_active = True
                    
                    if trail_active:
                        # Trail 0.75% from lowest price reached
                        proposed_sl = lowest_price_since_entry * (1 + TRAIL_STEP)
                        sl = min(sl, proposed_sl)

                exit_reason = None
                exit_price = None

                # CHECK EXITS (SL / Target)
                # We check Low/High against SL/Target to see if hit during the candle
                
                if position == "LONG":
                    if current_low <= sl:
                        exit_price = sl; exit_reason = "StopLoss" # Could be TrailingSL
                        if trail_active: exit_reason = "TrailingSL"
                    elif current_high >= target:
                        exit_price = target; exit_reason = "Target"

                elif position == "SHORT":
                    if current_high >= sl:
                        exit_price = sl; exit_reason = "StopLoss"
                        if trail_active: exit_reason = "TrailingSL"
                    elif current_low <= target:
                        exit_price = target; exit_reason = "Target"

                if exit_reason:
                    pnl = (exit_price - entry_price) * qty if position=="LONG" else (entry_price - exit_price) * qty
                    ret = pnl / capital

                    trade_log.append([
                        day, symbol, position,
                        entry_time, entry_price, qty,
                        t, exit_price, pnl, ret, exit_reason
                    ])




# ===============================
# BUILD TRADE LOG
# ===============================
trades = pd.DataFrame(trade_log, columns=[
    "Date","Stock","Direction",
    "EntryTime","EntryPrice","Qty",
    "ExitTime","ExitPrice","PnL","Return","ExitType"
])

trades.to_csv("trade_log.csv", index=False)
print("\nSaved trade log → trade_log.csv")
print(trades.head())


# ===============================
# PERFORMANCE METRICS
# ===============================
if len(trades) > 0:
    portfolio_equity = (1 + trades["Return"]).cumprod()
    total_return = portfolio_equity.iloc[-1] - 1
    win_rate = (trades["PnL"] > 0).mean() * 100
    dd = max_drawdown(portfolio_equity)
    sharpe = sharpe_ratio(trades["Return"])

    print("\n===== PERFORMANCE =====")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {dd*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("=======================")
else:
    print("\nNo trades were generated during the backtest period.")