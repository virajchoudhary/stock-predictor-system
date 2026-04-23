import yfinance as yf
import pandas as pd
from datetime import timedelta

symbol = "AAPL"
target_date = "2024-01-01"
end_dt = pd.to_datetime(target_date)
start_dt = end_dt - timedelta(days=10)

print(f"Fetching {symbol} from {start_dt.date()} to {(end_dt + timedelta(days=1)).date()}")
hist = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"), end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
print(hist.tail())

if not hist.empty:
    print(f"\nLast available price: {hist.iloc[-1]['Close']}")
    print(f"Last available date: {hist.index[-1]}")
