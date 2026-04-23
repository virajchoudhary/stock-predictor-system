import urllib.request, urllib.error
req=urllib.request.Request('http://127.0.0.1:8000/api/backtest-predict/AAPL/?past_date=2024-01-01', headers={'Accept':'application/json'})
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    with open('out_bt.txt', 'wb') as f:
        f.write(e.read())
