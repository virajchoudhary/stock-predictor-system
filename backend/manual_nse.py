import requests
import pandas as pd
import json

# Try manual fetch with headers if library fails
url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

print("Attempting manual fetch...")
try:
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers) # Visit homepage for cookies
    r = s.get(url, headers=headers)
    
    if r.status_code == 200:
        data = r.json()
        print("Success!")
        print(f"Top Level Keys: {data.keys()}")
        records = data.get('records', {})
        print(f"Records Keys: {records.keys()}")
        print(f"Expiry Dates: {records.get('expiryDates', [])[:3]}")
    else:
        print(f"Failed with Status: {r.status_code}")

except Exception as e:
    print(f"Manual Fetch Error: {e}")
