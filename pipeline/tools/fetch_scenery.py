"""Download what build_world.py needs for one circuit: outline, terrain tiles and Overture map layers.

usage: python3 fetch_scenery.py <f1-circuits id, e.g. sg-2008> <cache_dir>
Writes <cache_dir>/track.geojson, dem.pkl (Terrarium z15 heights) and ov/{land_use,land,water,seg,bld}.json.
"""
import io, json, math, os, pickle, subprocess, sys
import requests
from PIL import Image

CID, CACHE = sys.argv[1], sys.argv[2]
os.makedirs(f"{CACHE}/ov", exist_ok=True)
allc = requests.get("https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson", timeout=120).json()
feat = next(f for f in allc["features"] if f["properties"]["id"] == CID)
json.dump({"type": "FeatureCollection", "features": [feat]}, open(f"{CACHE}/track.geojson", "w"))
co = feat["geometry"]["coordinates"]
lon = [c[0] for c in co]; lat = [c[1] for c in co]
lat0 = sum(lat) / len(lat)
PAD = 700  # m, a little more than build_world's 650 m
dlat = PAD / 110540; dlon = PAD / (111320 * math.cos(math.radians(lat0)))
bb = (min(lon) - dlon, min(lat) - dlat, max(lon) + dlon, max(lat) + dlat)
print("bbox", bb, flush=True)

# terrain: Terrarium tiles (height = R*256 + G + B/256 - 32768 m) at zoom 15
z = 15; n = 2 ** z
def tile(lo, la): return (lo + 180) / 360 * n, (1 - math.log(math.tan(math.radians(la)) + 1 / math.cos(math.radians(la))) / math.pi) / 2 * n
x0, y1 = tile(bb[0], bb[1]); x1, y0 = tile(bb[2], bb[3])
tx0, ty0, tx1, ty1 = int(x0) - 1, int(y0) - 1, int(x1) + 1, int(y1) + 1
W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
hs = [0.0] * (W * H)
for ty in range(ty0, ty1 + 1):
    for tx in range(tx0, tx1 + 1):
        im = Image.open(io.BytesIO(requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png", timeout=60).content)).convert("RGB")
        px = im.load()
        for j in range(256):
            row = (ty - ty0) * 256 + j
            for i in range(256):
                r, g, b = px[i, j]
                hs[row * W + (tx - tx0) * 256 + i] = r * 256 + g + b / 256 - 32768
pickle.dump((tx0, ty0, W, H, hs), open(f"{CACHE}/dem.pkl", "wb"))
print("terrain tiles", (tx1 - tx0 + 1) * (ty1 - ty0 + 1), flush=True)

here = os.path.dirname(os.path.abspath(__file__))
B = ",".join(map(str, bb))
for name, t, cols in [("land_use", "theme=base/type=land_use", "class,subtype,geometry"),
                      ("land", "theme=base/type=land", "class,subtype,geometry"),
                      ("water", "theme=base/type=water", "class,subtype,geometry"),
                      ("seg", "theme=transportation/type=segment", "class,subtype,geometry"),
                      ("bld", "theme=buildings/type=building", "class,height,num_floors,geometry")]:
    if os.path.exists(f"{CACHE}/ov/{name}.json"): continue
    subprocess.run([sys.executable, "-P", f"{here}/ovfetch.py", t, cols, f"{CACHE}/ov/{name}.json", B], check=True)
print("done")
