from nsepython import nse_optionchain_scrapper
import json

try:
    print("Fetching NIFTY Option Chain via nsepython...")
    oi_data = nse_optionchain_scrapper('NIFTY')
    
    # Just print a snippet to understand structure
    print("Keys:", oi_data.keys())
    if 'records' in oi_data:
        print("Expiry Dates:", oi_data['records']['expiryDates'][:3])
        print("Sample Data:", oi_data['records']['data'][0])
    
    if 'filtered' in oi_data:
         print("Filtered Sample (CE):", oi_data['filtered']['data'][0]['CE'])

except Exception as e:
    print(f"Error: {e}")
