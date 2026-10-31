"""Read members of a remote zip via HTTP range requests (no full download)."""
import io, urllib.request, zipfile
UA = {"User-Agent": "Mozilla/5.0 (bento3-lora)"}
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA))
        self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        h = dict(UA, Range=f"bytes={self.pos}-{end}")
        for attempt in range(5):
            try:
                b = urllib.request.urlopen(urllib.request.Request(self.url, headers=h), timeout=120).read(); break
            except Exception:
                if attempt == 4: raise
        self.pos += len(b); return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
def open_zip(url): return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=8 << 20))
def extract(url, member, dst, chunk=64 << 20):
    z = open_zip(url)
    with z.open(member) as f, open(dst + ".part", "wb") as o:
        while True:
            b = f.read(chunk)
            if not b: break
            o.write(b)
    import os; os.replace(dst + ".part", dst)
if __name__ == "__main__":
    import sys
    for i in open_zip(sys.argv[1]).infolist(): print(f"{i.file_size/1e6:10.1f} MB  {i.compress_size/1e6:10.1f} MB  {i.filename}")
