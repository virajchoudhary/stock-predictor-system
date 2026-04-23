import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:8000/api/optimize/', data=json.dumps({'tickers': ['AAPL', 'MSFT'], 'risk_tolerance': 0.5}).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
    print("Success")
except urllib.error.HTTPError as e:
    print(e.read().decode('utf-8'))
