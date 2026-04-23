import json

data = open('out.txt', 'rb').read()
try:
    print("JSON:", json.loads(data))
except Exception:
    html = data.decode('utf-8')
    s = html.find('Exception Value:')
    e = html.find('Exception Location:')
    print(html[s:e].strip())
