import urllib.request, urllib.error, json
req = urllib.request.Request('http://127.0.0.1:8000/api/optimize/', data=json.dumps({'tickers': ['AAPL', 'MSFT'], 'risk_tolerance': 0.5}).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    with open('out_opt.txt', 'wb') as f:
        f.write(e.read())
