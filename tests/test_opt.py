import urllib.request, json, re

req = urllib.request.Request('http://127.0.0.1:8000/api/optimize/', data=json.dumps({'tickers': ['AAPL', 'MSFT'], 'risk_tolerance': 0.5}).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    urllib.request.urlopen(req)
    print("Success")
except urllib.error.HTTPError as e:
    html = e.read().decode('utf-8')
    val = re.search(r'Exception Value:\n\s*</dt>\n\s*<dd>(.*?)</dd>', html, re.DOTALL)
    loc = re.search(r'Exception Location:.*?<span class="fname">(.*?)</span>,\s*line\s*(\d+)', html)
    print("Error:", val.group(1).strip() if val else "unknown val")
    print("Location:", loc.group(1) if loc else "unknown loc", "line", loc.group(2) if loc else "unknown loc")
