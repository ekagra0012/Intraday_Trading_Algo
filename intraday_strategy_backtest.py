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
def process_day(day_df, all_trades_list, current_capital):
    """
    Runs the backtest for a single day dataframe and APPENDS trades to all_trades_list.
    Returns the updated current_capital.
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
    
    day_trades = [] # Temporary list for this day
    
    for symbol, m1 in day_df.groupby("symbol"):
        # Resample 10min (Signals)
        m10 = m1.resample("10min", origin="start", closed="right", label="right").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        h1 = m1.resample("1h", origin="start", closed="right", label="right").agg({
            "close": "last"
        }).dropna()
        h1["ema50"] = ema(h1["close"], 50)
        
        m10["ema3"] = ema(m10["close"], 3)
        m10["ema10"] = ema(m10["close"], 10)
        m10["rsi14"] = rsi_wilder(m10["close"], 14)
        
        m10 = pd.merge_asof(m10, h1["ema50"], left_index=True, right_index=True, direction="backward")
        
        valid_idx = m10.index[m10["ema10"].notna() & m10["rsi14"].notna()]
        
        active_trade = None
        current_time_cursor = m1.index[0]
        
        for t_idx, row in m10.loc[valid_idx].iterrows():
            signal_time = t_idx
            if signal_time < current_time_cursor:
                continue 
            
            if active_trade is not None:
                continue 
            
            raw_ema50 = row["ema50"]
            if pd.isna(raw_ema50):
                continue
            
            long_signal = (row["ema3"] > row["ema10"]) and (row["rsi14"] > 60) and (row["close"] > raw_ema50)
            short_signal = (row["ema3"] < row["ema10"]) and (row["rsi14"] < 30) and (row["close"] < raw_ema50)
            
            if not (long_signal or short_signal):
                continue
                
            direction = "LONG" if long_signal else "SHORT"
            trigger_price = 0.0
            
            if direction == "LONG":
                trigger_price = row["high"]
            else:
                # STRICT Short Window: t-5 to t-1
                # [t-5, t-1] is 5 candles. Exclude signal minute t.
                subset_m1 = m1.loc[signal_time - timedelta(minutes=5) : signal_time - timedelta(minutes=1)] 
                if subset_m1.empty: continue
                trigger_price = subset_m1["low"].min()
            
            # --- CHECK FILL (Next 10 mins) ---
            next_data = m1.loc[signal_time + timedelta(minutes=1) : signal_time + timedelta(minutes=10)]
            
            entry_filled = False
            fill_info = {}
            
            for min_t, p_row in next_data.iterrows():
                if direction == "LONG":
                    if p_row["high"] >= trigger_price:
                        entry_filled = True
                        fill_info["time"] = min_t
                        # Explicit Gap Logic: If Open > Trigger, Fill at Open. Else Trigger.
                        fill_info["price"] = p_row["open"] if p_row["open"] > trigger_price else trigger_price
                        break
                else:
                    if p_row["low"] <= trigger_price:
                        entry_filled = True
                        fill_info["time"] = min_t
                        # Explicit Gap Logic: If Open < Trigger, Fill at Open. Else Trigger.
                        fill_info["price"] = p_row["open"] if p_row["open"] < trigger_price else trigger_price
                        break
            
            if entry_filled:
                ep = fill_info["price"]
                sl = ep * (1 - STOP_LOSS_PCT) if direction=="LONG" else ep * (1 + STOP_LOSS_PCT)
                tgt = ep * (1 + TARGET_PCT) if direction=="LONG" else ep * (1 - TARGET_PCT)
                
                active_trade = {
                    "symbol": symbol, "direction": direction,
                    "entry_time": fill_info["time"], "entry_price": ep, 
                    "sl": sl, "target": tgt,
                    "trail_active": False, "extreme_price": ep
                }
                
                trade_data = m1.loc[fill_info["time"]:] 
                
                exit_found = False
                
                for t_trade, row_trade in trade_data.iterrows():
                    c_high = row_trade["high"]
                    c_low = row_trade["low"]
                    
                    # 1. CHECK EXIT FIRST (using existing SL)
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
                         pnl_per_share = (exit_px - active_trade["entry_price"]) if direction=="LONG" else (active_trade["entry_price"] - exit_px)
                         day_trades.append({
                             "Date": t_trade.date(), "Stock": symbol, "Direction": direction,
                             "EntryTime": active_trade["entry_time"], "EntryPrice": active_trade["entry_price"],
                             "ExitTime": t_trade, "ExitPrice": exit_px,
                             "PnL_Per_Share": pnl_per_share, "ExitType": exit_reason
                         })
                         current_time_cursor = t_trade
                         active_trade = None
                         exit_found = True
                         break

                    # 2. UPDATE TRAILING (After verifying we survived this bar)
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
                
                # EOD Square Off 
                if active_trade and not exit_found:
                    last_row = trade_data.iloc[-1]
                    exit_px = last_row["close"]
                    pnl_per_share = (exit_px - active_trade["entry_price"]) if direction=="LONG" else (active_trade["entry_price"] - exit_px)
                    day_trades.append({
                         "Date": last_row.name.date(), "Stock": symbol, "Direction": direction,
                         "EntryTime": active_trade["entry_time"], "EntryPrice": active_trade["entry_price"],
                         "ExitTime": last_row.name, "ExitPrice": exit_px,
                         "PnL_Per_Share": pnl_per_share, "ExitType": "EOD_SquareOff"
                    })
                    current_time_cursor = last_row.name
                    active_trade = None

    # POST-PROCESSING: Sort by Entry Time and Apply Dynamic Capital
    day_trades.sort(key=lambda x: x["EntryTime"])
    
    for tr in day_trades:
        # Dynamic Sizing: 0.5% of CURRENT capital
        risk_amt = current_capital * RISK_PER_TRADE
        dist = tr["EntryPrice"] * STOP_LOSS_PCT
        qty = max(1, int(risk_amt / dist))
        
        tr["Qty"] = qty
        tr["PnL"] = tr["PnL_Per_Share"] * qty
        tr["Return"] = tr["PnL"] / current_capital
        
        # Update Capital
        current_capital += tr["PnL"]
        
        all_trades_list.append(tr)
    
    return current_capital

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
    current_capital = BASE_CAPITAL
    
    print("Starting backtest loop...")
    for day, group in full_data.groupby(full_data["time"].dt.date):
        print(f"Processing {day} | Start Capital: {current_capital:.2f}")
        try:
            current_capital = process_day(group, all_trades, current_capital)
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
        final_capital = current_capital
        total_return_pct = (final_capital - BASE_CAPITAL) / BASE_CAPITAL
        win_rate = (trades_df["PnL"] > 0).mean() * 100
        trades_df["Equity"] = BASE_CAPITAL + trades_df["PnL"].cumsum()
        dd = max_drawdown(trades_df["Equity"])
        sharpe = sharpe_ratio(trades_df["Return"]) 
        
        print("\n===== PRECISION PERFORMANCE =====")
        print(f"Trades: {len(trades_df)}")
        print(f"Final Capital: {final_capital:.2f}")
        print(f"Total Return: {total_return_pct*100:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Max Drawdown: {dd*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print("=================================")
    else:
        print("\nNo trades generated.")