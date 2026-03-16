import quant_reporter as qr
import os
from datetime import datetime, timedelta

try:
    print("Starting debug script...")
    
    # Portfolio inputs
    tickers = ["AAPL", "MSFT", "SPY", "QQQ"]
    benchmark = "SPY"
    weight = 1.0 / len(tickers)
    portfolio_dict = {t: weight for t in tickers}
    
    # Dates
    end_date = datetime.now() - timedelta(days=90)
    start_date = end_date - timedelta(days=365*2)
    
    report_file = os.path.abspath("debug_report.html")
    print(f"Target file: {report_file}")
    
    qr.create_combined_report(
        portfolio_dict=portfolio_dict,
        benchmark_ticker=benchmark,
        train_start=start_date.strftime('%Y-%m-%d'),
        train_end=end_date.strftime('%Y-%m-%d'),
        filename=report_file
    )
    
    if os.path.exists(report_file):
        print("SUCCESS: File created.")
    else:
        print("FAILURE: File not found.")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
