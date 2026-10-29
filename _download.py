import urllib.request
import os
import time

url = "https://zenodo.org/records/16565801/files/KuaiLive.zip"
out = r"e:\kuailive_project\KuaiLive直播推荐实验\data\raw\KuaiLive_real.zip"

req = urllib.request.Request(url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': '*/*',
})

print(f"Downloading {url} ...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        print(f"Server reports size: {total/1024/1024:.2f} MB" if total else "Server did not report Content-Length")
        downloaded = 0
        chunk_size = 1024 * 256
        with open(out, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024*1024*5) < chunk_size:
                    print(f"  {downloaded/1024/1024:.1f} MB ...")
        print(f"Done. Downloaded {downloaded/1024/1024:.2f} MB in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"ERROR: {e}")
    if os.path.exists(out):
        sz = os.path.getsize(out)
        print(f"Partial file size: {sz/1024/1024:.2f} MB")
