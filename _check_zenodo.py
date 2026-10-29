import urllib.request, json
req = urllib.request.Request('https://zenodo.org/api/records/16565801', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)
for f in data.get('files', []):
    size_mb = f['size'] / 1024 / 1024
    print(f"{f['key']}  {size_mb:.2f} MB  {f['links']['self']}")
