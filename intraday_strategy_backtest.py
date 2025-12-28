import pandas as pd
import numpy as np
import glob
from datetime import timedelta, time

# ===============================
# CONFIG
# ===============================
BASE_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.005          # 0.5%
STOP_LOSS_PCT = 0.005           # 0.5%
TARGET_PCT = 0.02               # 2%
TRAIL_TRIGGER = 0.005           # +0.5% (Activation)
TRAIL_STEP = 0.0075             # 0.75% (Trailing Distance)

DATA_FILES = glob.glob("data/dataNSE_*.csv")

# ===============================
# INDICATORS
# ===============================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(series, period=14):
    """
    RSI using Wilder's Smoothing (Exponential Moving Average).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's Smoothing: alpha = 1/n
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ===============================
# METRICS
# ===============================
def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity/roll_max - 1
    return dd.min()

def sharpe_ratio(returns):
    if returns.std() == 0: return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

# ===============================
# ENGINE
# ===============================
def process_day(day_df, all_trades_list):
    """
    Runs the backtest for a single day dataframe and APPENDS trades to all_trades_list.
    """
    day_df = day_df.sort_values("time").copy()
    day_df = day_df.set_index("time")
    
    # 1. STOCK SELECTION (09:15 - 09:25)
    sel_start = day_df.index[0].replace(hour=9, minute=15, second=0)
    selection_window = day_df.between_time("09:15", "09:25")
    
    turnover_calc = (
        selection_window.assign(turnover=lambda x: x["close"] * x["volume"])
        .groupby("symbol")["turnover"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    top_10 = turnover_calc.index.tolist()
    day_df = day_df[day_df["symbol"].isin(top_10)]
    
    for symbol, m1 in day_df.groupby("symbol"):
        # Resample 10min (Signals)
        m10 = m1.resample("10min", origin="start", closed="right", label="right").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        # Resample 1H (EMA 50 Filter)
        h1 = m1.resample("1h", origin="start", closed="right", label="right").agg({
            "close": "last"
        }).dropna()
        h1["ema50"] = ema(h1["close"], 50)
        
        m10["ema3"] = ema(m10["close"], 3)
        m10["ema10"] = ema(m10["close"], 10)
        m10["rsi14"] = rsi_wilder(m10["close"], 14)
        
        # Merge H1 EMA50 backward (so 10:25 sees 10:15 H1 close, effectively known data? 
        # Actually 10:15 close is known at 10:15. So 10:20, 10:30 candles can use it.
        # merge_asof backward matches <= timestamp.
        # 10:25 matches 10:15. Correct.)
        m10 = pd.merge_asof(m10, h1["ema50"], left_index=True, right_index=True, direction="backward")
        
        valid_idx = m10.index[m10["ema10"].notna() & m10["rsi14"].notna()]
        
        active_trade = None
        current_time_cursor = m1.index[0]
        
        for t_idx, row in m10.loc[valid_idx].iterrows():
            signal_time = t_idx
            if signal_time < current_time_cursor:
                continue 
            
            if active_trade is not None:
                continue # One trade at a time per stock
            
            # --- CONDITIONS ---
            # If EMA50 is NaN (early in day), condition fails.
            raw_ema50 = row["ema50"]
            if pd.isna(raw_ema50):
                continue
            
            long_signal = (row["ema3"] > row["ema10"]) and (row["rsi14"] > 60) and (row["close"] > raw_ema50)
            short_signal = (row["ema3"] < row["ema10"]) and (row["rsi14"] < 30) and (row["close"] < raw_ema50)
            
            if not (long_signal or short_signal):
                continue
                
            # --- SETUP ---
            direction = "LONG" if long_signal else "SHORT"
            trigger_price = 0.0
            
            if direction == "LONG":
                trigger_price = row["high"]
            else:
                subset_m1 = m1.loc[signal_time - timedelta(minutes=4) : signal_time]
                if subset_m1.empty: continue
                trigger_price = subset_m1["low"].min()
            
            # --- CHECK FILL (Next 10 mins) ---
            # Limit fill check window to reasonable time (next signal candle duration)
            next_data = m1.loc[signal_time + timedelta(minutes=1) : signal_time + timedelta(minutes=10)]
            
            entry_filled = False
            fill_info = {}
            
            for min_t, p_row in next_data.iterrows():
                if direction == "LONG":
                    if p_row["high"] >= trigger_price:
                        entry_filled = True
                        fill_info["time"] = min_t
                        fill_info["price"] = max(p_row["open"], trigger_price)
                        break
                else:
                    if p_row["low"] <= trigger_price:
                        entry_filled = True
                        fill_info["time"] = min_t
                        fill_info["price"] = min(p_row["open"], trigger_price)
                        break
            
            if entry_filled:
                ep = fill_info["price"]
                risk_amt = BASE_CAPITAL * RISK_PER_TRADE # 5000
                dist = ep * STOP_LOSS_PCT
                qty = max(1, int(risk_amt / dist))
                
                # Setup Trade
                sl = ep * (1 - STOP_LOSS_PCT) if direction=="LONG" else ep * (1 + STOP_LOSS_PCT)
                tgt = ep * (1 + TARGET_PCT) if direction=="LONG" else ep * (1 - TARGET_PCT)
                
                active_trade = {
                    "symbol": symbol, "direction": direction,
                    "entry_time": fill_info["time"], "entry_price": ep, "qty": qty,
                    "sl": sl, "target": tgt,
                    "trail_active": False, "extreme_price": ep
                }
                
                # Monitor until Close
                # From fill_time...
                trade_data = m1.loc[fill_info["time"]:] 
                # Note: This includes the fill candle. But we assume fill happens 
                # linearly before exit? To be safe, verify exit on SUBSEQUENT candles?
                # Or check High/Low carefully.
                # If we entered at Open of fill candle, we can stop out in same candle. 
                # Let's check loop.
                
                for t_trade, row_trade in trade_data.iterrows():
                    # Skip the exact minute of entry if we want to avoid "instant out"
                    # unless we are sure logic holds.
                    # Simplified: Check SL/Target on current bars.
                    
                    c_high = row_trade["high"]
                    c_low = row_trade["low"]
                    
                    # TRAILING
                    if direction == "LONG":
                        active_trade["extreme_price"] = max(active_trade["extreme_price"], c_high)
                        if not active_trade["trail_active"]:
                            if active_trade["extreme_price"] >= active_trade["entry_price"] * (1 + TRAIL_TRIGGER):
                                active_trade["trail_active"] = True
                        if active_trade["trail_active"]:
                            new_sl = active_trade["extreme_price"] * (1 - TRAIL_STEP)
                            active_trade["sl"] = max(active_trade["sl"], new_sl)
                    else:
                        active_trade["extreme_price"] = min(active_trade["extreme_price"], c_low)
                        if not active_trade["trail_active"]:
                            if active_trade["extreme_price"] <= active_trade["entry_price"] * (1 - TRAIL_TRIGGER):
                                active_trade["trail_active"] = True
                        if active_trade["trail_active"]:
                            new_sl = active_trade["extreme_price"] * (1 + TRAIL_STEP)
                            active_trade["sl"] = min(active_trade["sl"], new_sl)
                    
                    # EXIT CHECK
                    exit_reason = None; exit_px = 0.0
                    
                    if direction == "LONG":
                        if c_low <= active_trade["sl"]:
                            exit_px = active_trade["sl"]
                            exit_reason = "StopLoss" if not active_trade["trail_active"] else "TrailingSL"
                        elif c_high >= active_trade["target"]:
                            exit_px = active_trade["target"]
                            exit_reason = "Target"
                    else:
                        if c_high >= active_trade["sl"]:
                            exit_px = active_trade["sl"]
                            exit_reason = "StopLoss" if not active_trade["trail_active"] else "TrailingSL"
                        elif c_low <= active_trade["target"]:
                            exit_px = active_trade["target"]
                            exit_reason = "Target"
                    
                    if exit_reason:
                         pnl = (exit_px - active_trade["entry_price"]) * qty if direction=="LONG" else (active_trade["entry_price"] - exit_px) * qty
                         all_trades_list.append({
                             "Date": t_trade.date(), "Stock": symbol, "Direction": direction,
                             "EntryTime": active_trade["entry_time"], "EntryPrice": active_trade["entry_price"],
                             "Qty": qty, "ExitTime": t_trade, "ExitPrice": exit_px,
                             "PnL": pnl, "Return": pnl/BASE_CAPITAL, "ExitType": exit_reason
                         })
                         current_time_cursor = t_trade
                         active_trade = None
                         break
                
                # EOD Square Off
                if active_trade:
                    last_row = trade_data.iloc[-1]
                    exit_px = last_row["close"]
                    pnl = (exit_px - active_trade["entry_price"]) * qty if direction=="LONG" else (active_trade["entry_price"] - exit_px) * qty
                    all_trades_list.append({
                         "Date": last_row.name.date(), "Stock": symbol, "Direction": direction,
                         "EntryTime": active_trade["entry_time"], "EntryPrice": active_trade["entry_price"],
                         "Qty": qty, "ExitTime": last_row.name, "ExitPrice": exit_px,
                         "PnL": pnl, "Return": pnl/BASE_CAPITAL, "ExitType": "EOD_SquareOff"
                    })
                    current_time_cursor = last_row.name
                    active_trade = None

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    raw_dfs = []
    print("Loading data...")
    for f in DATA_FILES:
        try:
            d = pd.read_csv(f)
            d["time"] = pd.to_datetime(d["time"])
            raw_dfs.append(d)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    if not raw_dfs:
        print("No data loaded.")
        exit()

    full_data = pd.concat(raw_dfs).sort_values(["time", "ticker"])
    full_data = full_data.rename(columns={"ticker": "symbol"})

    all_trades = []
    
    print("Starting backtest loop...")
    for day, group in full_data.groupby(full_data["time"].dt.date):
        print(f"Processing {day}...")
        try:
            process_day(group, all_trades)
        except Exception as e:
            print(f"Error on {day}: {e}")

    # RESULTS
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.sort_values("EntryTime")
        cols = ["Date","Stock","Direction","EntryTime","EntryPrice","Qty","ExitTime","ExitPrice","PnL","Return","ExitType"]
        trades_df = trades_df[cols]
        
        trades_df.to_csv("trade_log.csv", index=False)
        print("\nSaved trade_log.csv")
        
        # Metrics
        portfolio_equity = (1 + trades_df["Return"]).cumprod()
        total_return = portfolio_equity.iloc[-1] - 1
        win_rate = (trades_df["PnL"] > 0).mean() * 100
        dd = max_drawdown(portfolio_equity)
        sharpe = sharpe_ratio(trades_df["Return"])
        
        print("\n===== STRICT PERFORMANCE =====")
        print(f"Trades: {len(trades_df)}")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Max Drawdown: {dd*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print("==============================")
        
    else:
        print("\nNo trades generated.")