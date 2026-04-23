import sys
import os
import pandas as pd
from datetime import datetime

# Setup Django environment
sys.path.append("/Users/dhruvaambhaikar/Downloads/stock-predictor-system-integration-final/backend")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

from api.ai_services import get_ticker_info

def test_price(symbol, target_date):
    print(f"\nTesting {symbol} as of {target_date}...")
    info = get_ticker_info(symbol, target_date=target_date)
    print(f"Name: {info.get('name')}")
    print(f"Current Price (Adjusted): {info.get('currentPrice')}")
    print(f"Market Cap: {info.get('marketCap')}")

# AAPL split was in 2020, so 2024 prices are just dividend adjusted.
test_price("AAPL", "2024-01-01") # Expected 191.91
test_price("AAPL", "2024-01-02") # Expected 183.73
