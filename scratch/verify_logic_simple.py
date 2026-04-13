import yfinance as yf
import pandas as pd
from datetime import timedelta

def get_price_history_mock(symbol, target_date):
    end_dt = pd.to_datetime(target_date)
    start_dt = end_dt - timedelta(days=5)
    hist = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"), 
                       end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"), 
                       progress=False, auto_adjust=True)
    if isinstance(hist, pd.DataFrame) and isinstance(hist.columns, pd.MultiIndex):
        hist = hist.droplevel(-1, axis=1)
    return hist

def test_logic(target_date):
    print(f"\n--- Testing As Of: {target_date} ---")
    hist = get_price_history_mock("AAPL", target_date)
    if not hist.empty:
        close = hist["Close"]
        price = float(close.iloc[-1])
        date = close.index[-1]
        print(f"Nearest Trading Date: {date.date()}")
        print(f"Adjusted Close: {price:.2f}")
    else:
        print("No data found.")

test_logic("2024-01-01") # Expected Dec 29, 2023 ~191.91
test_logic("2024-01-02") # Expected Jan 2, 2024 ~183.73
