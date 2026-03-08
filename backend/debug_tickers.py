import yfinance as yf

tickers = ["^NSEI", "RELIANCE.NS", "TCS.NS", "SPY"]

print("Checking Ticker Options Availability...")
for t in tickers:
    try:
        tick = yf.Ticker(t)
        opts = tick.options
        if opts:
            print(f"[SUCCESS] {t}: {len(opts)} expiries found. Next: {opts[0]}")
        else:
             print(f"[FAILURE] {t}: No expirations found.")
    except Exception as e:
        print(f"[ERROR] {t}: {e}")
