"""Fetch Overture rows inside a bbox using HTTP range requests (row-group pruning on bbox stats).
usage: python3 ovfetch.py <theme=…/type=…> <col1,col2> <out.json> <lon_min,lat_min,lon_max,lat_max>"""
import io, re, sys, json, requests, concurrent.futures as cf
import pyarrow.parquet as pq, pyarrow as pa

BASE = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"
REL = "release/2026-09-23.1"
XMIN = YMIN = XMAX = YMAX = None  # set by set_bbox() or the command line
S = requests.Session()

def set_bbox(xmin, ymin, xmax, ymax):
    global XMIN, YMIN, XMAX, YMAX
    XMIN, YMIN, XMAX, YMAX = xmin, ymin, xmax, ymax

class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.cache = url, size, 0, {}
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        end = min(self.pos + n, self.size) - 1
        if end < self.pos: return b""
        for _ in range(4):
            try:
                r = S.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120); r.raise_for_status(); break
            except Exception as e: err = e
        else: raise err
        self.pos = end + 1
        return r.content
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def list_files(t):
    x = S.get(f"{BASE}/?list-type=2&prefix={REL}/{t}/").text
    return [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x)]

def bbox_cols(md):
    names = {}
    rg = md.row_group(0)
    for i in range(rg.num_columns):
        p = rg.column(i).path_in_schema
        if p in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax"): names[p] = i
    return names

def scan(key, size, columns):
    f = pq.ParquetFile(RangeFile(f"{BASE}/{key}", size))
    md = f.metadata; bc = bbox_cols(md); hits = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        st = {k: rg.column(i).statistics for k, i in bc.items()}
        if st["bbox.xmin"].min > XMAX or st["bbox.xmax"].max < XMIN or st["bbox.ymin"].min > YMAX or st["bbox.ymax"].max < YMIN: continue
        hits.append(g)
    out = []
    for g in hits:
        tb = f.read_row_group(g, columns=columns + ["bbox"])
        for row in tb.to_pylist():
            b = row["bbox"]
            if b["xmin"] > XMAX or b["xmax"] < XMIN or b["ymin"] > YMAX or b["ymax"] < YMIN: continue
            row.pop("bbox"); out.append(row)
    return key, hits, out

if __name__ == "__main__":
    t, cols, outp = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    set_bbox(*map(float, sys.argv[4].split(",")))
    files = list_files(t)
    print(t, len(files), "files", file=sys.stderr)
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for key, hits, out in ex.map(lambda kv: scan(kv[0], kv[1], cols), files):
            if hits: print(key.split("/")[-1][:12], "groups", hits, "rows", len(out), file=sys.stderr)
            rows += out
    def enc(o):
        if isinstance(o, bytes): return o.hex()
        return str(o)
    json.dump(rows, open(outp, "w"), default=enc)
    print("total", len(rows), file=sys.stderr)
