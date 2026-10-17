"""Build a circuit's 3D world (terrain, land cover, roads, water, trees, buildings) in local metres.

usage: python3 build_world.py <cache_dir from fetch_scenery.py> <out world.json>
"""
import json, math, pickle, base64, random
import numpy as np
import shapely
from shapely import wkb
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.strtree import STRtree

import sys
D = sys.argv[1].rstrip("/") + "/"; OUTW = sys.argv[2]
coords = json.load(open(D + "track.geojson"))["features"][0]["geometry"]["coordinates"][:-1]
lon0 = sum(c[0] for c in coords) / len(coords)
lat0 = sum(c[1] for c in coords) / len(coords)
mLon = 111320 * math.cos(math.radians(lat0)); mLat = 110540
def proj(lon, lat): return ((lon - lon0) * mLon, -(lat - lat0) * mLat)  # x east, z south
def projg(g): return shapely.transform(g, lambda a: np.column_stack([(a[:, 0] - lon0) * mLon, -(a[:, 1] - lat0) * mLat]))

track = LineString([proj(*c) for c in coords] + [proj(*coords[0])])
# better: the line the cars actually drive (from the position feed, already placed on the map by fit_feed.py)
import os
if os.path.exists(D + "track_local.json"):
    _tl = json.load(open(D + "track_local.json")); track = LineString(_tl + [_tl[0]])
    print("track from the position feed", len(_tl), "points")
tx0, tz0, tx1, tz1 = track.bounds
PAD = 650
X0, Z0, X1, Z1 = tx0 - PAD, tz0 - PAD, tx1 + PAD, tz1 + PAD
region = box(X0, Z0, X1, Z1)
print("region m", round(X1 - X0), round(Z1 - Z0), "track len", round(track.length))

# ---------- terrain ----------
tx, ty, W, H, hs = pickle.load(open(D + "dem.pkl", "rb"))
dem = np.array(hs, dtype=np.float64).reshape(H, W)
z = 15; n = 2 ** z
def lonlat_to_px(lon, lat):
    x = (lon + 180) / 360 * n * 256 - tx * 256
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n * 256 - ty * 256
    return x, y
def blur(a, sigma_px):
    r = int(3 * sigma_px); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2); k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, a)
# the tiles are a surface model in cities (rooftops, bridges, spikes): take the 20th percentile of 8x8-pixel blocks
# (~40 m), then a grey opening (min then max over ~5 blocks) that keeps the ground and drops anything narrower,
# then back to full resolution
from numpy.lib.stride_tricks import sliding_window_view as _sw
def _filt(a, r, f):
    p = np.pad(a, r, mode="edge")
    p = f(_sw(p, 2 * r + 1, axis=1), axis=-1)
    return f(_sw(p, 2 * r + 1, axis=0), axis=-1)
_raw = dem.copy()
_B = 8
_g = np.percentile(dem.reshape(H // _B, _B, W // _B, _B).transpose(0, 2, 1, 3).reshape(H // _B, W // _B, -1), 20, axis=-1)
_g = _filt(_filt(_g, 2, np.min), 2, np.max)
_yi = (np.arange(H) + 0.5) / _B - 0.5; _xi = (np.arange(W) + 0.5) / _B - 0.5
_y0 = np.clip(np.floor(_yi).astype(int), 0, H // _B - 2); _x0 = np.clip(np.floor(_xi).astype(int), 0, W // _B - 2)
_fy = np.clip(_yi - _y0, 0, 1)[:, None]; _fx = np.clip(_xi - _x0, 0, 1)[None]
dem = (_g[_y0][:, _x0] * (1 - _fx) + _g[_y0][:, _x0 + 1] * _fx) * (1 - _fy) + (_g[_y0 + 1][:, _x0] * (1 - _fx) + _g[_y0 + 1][:, _x0 + 1] * _fx) * _fy
# ground from the road network: streets are at ground level even downtown. Sample the raw tiles along every road,
# keep a low percentile per block, spread it smoothly; where roads are dense it replaces the opening above
try:
    from shapely import wkb as _wkb
    _px = []
    for r in json.load(open(D + "ov/seg.json")):
        if r.get("subtype") != "road": continue
        g = _wkb.loads(bytes.fromhex(r["geometry"]))
        for l in ([g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))):
            c = np.array(l.coords)
            for k in range(len(c) - 1):
                m_ = max(1, int(np.hypot(*((c[k + 1] - c[k]) * [mLon, mLat])) / 8))
                for f in np.arange(m_) / m_:
                    lo_, la_ = c[k] + (c[k + 1] - c[k]) * f
                    _px.append(lonlat_to_px(lo_, la_))
    _px = np.array(_px); _ix = _px[:, 0].astype(int); _iy = _px[:, 1].astype(int)
    _ok = (_ix >= 0) & (_ix < W) & (_iy >= 0) & (_iy < H); _ix, _iy = _ix[_ok], _iy[_ok]
    _v = _raw[_iy, _ix]; _cell = (_iy // _B) * (W // _B) + _ix // _B
    _G = np.full((H // _B) * (W // _B), np.nan)
    _o = np.argsort(_cell); _cs, _vs = _cell[_o], _v[_o]
    _st = np.flatnonzero(np.r_[True, _cs[1:] != _cs[:-1]]); _en = np.r_[_st[1:], len(_cs)]
    for a_, b_ in zip(_st, _en): _G[_cs[a_]] = np.percentile(_vs[a_:b_], 25)
    _G = _G.reshape(H // _B, W // _B); _m = ~np.isnan(_G); _G[~_m] = 0
    _num = blur(_G * _m, 3.0); _den = blur(_m.astype(float), 3.0)
    _road = _num / np.maximum(_den, 1e-6); _w = np.clip(_den / 0.3, 0, 1)
    _g2 = _road * _w + _g * (1 - _w)
    dem = (_g2[_y0][:, _x0] * (1 - _fx) + _g2[_y0][:, _x0 + 1] * _fx) * (1 - _fy) + (_g2[_y0 + 1][:, _x0] * (1 - _fx) + _g2[_y0 + 1][:, _x0 + 1] * _fx) * _fy
    print("road ground samples", len(_v))
except FileNotFoundError:
    pass
dem_s = blur(dem, 6.0)  # ~ 20 m
def dem_at(x, zz):
    lon = x / mLon + lon0; lat = -zz / mLat + lat0
    px, py = lonlat_to_px(lon, lat)
    i0, j0 = int(px), int(py); fx, fy = px - i0, py - j0
    a = dem_s[j0, i0] * (1 - fx) + dem_s[j0, i0 + 1] * fx
    b = dem_s[j0 + 1, i0] * (1 - fx) + dem_s[j0 + 1, i0 + 1] * fx
    return a * (1 - fy) + b * fy

# dense track heights, smoothed along the lap (window ~240 m)
L = track.length; step = 4.0; m = int(L / step)
dense = [track.interpolate(i * step) for i in range(m)]
th = np.array([dem_at(p.x, p.y) for p in dense])
k = np.ones(61) / 61
th = np.convolve(np.concatenate([th[-30:], th, th[:30]]), k, "valid")
print("track heights (every 400 m)", np.round(th[::100], 1).tolist())
dpts = np.array([[p.x, p.y] for p in dense])

G = 8.0
nx = int((X1 - X0) / G) + 1; nz = int((Z1 - Z0) / G) + 1
gx = X0 + np.arange(nx) * G; gz = Z0 + np.arange(nz) * G
terr = np.array([[dem_at(x, zz) for x in gx] for zz in gz])
# flatten a corridor around the track to the track height (slightly below)
GX, GZ = np.meshgrid(gx, gz)
flat = np.stack([GX.ravel(), GZ.ravel()], 1)
best_d = np.full(len(flat), 1e9); best_h = np.zeros(len(flat))
for i in range(0, m, 1):
    d = np.hypot(flat[:, 0] - dpts[i, 0], flat[:, 1] - dpts[i, 1])
    sel = d < best_d; best_d[sel] = d[sel]; best_h[sel] = th[i]
best_d = best_d.reshape(terr.shape); best_h = best_h.reshape(terr.shape)
w = np.clip((best_d - 26) / 60, 0, 1); w = w * w * (3 - 2 * w)
terr = (best_h - 0.25) * (1 - w) + terr * w
base = float(np.floor(terr.min()))
terr_dm = np.round((terr - base) * 10).astype("<u2")
print("terrain grid", nx, nz, "base", base, "range m", round(float(terr.max() - terr.min()), 1), "track h range", round(float(th.max() - th.min()), 1))

# ---------- vectors ----------
def load(name): return json.load(open(D + f"ov/{name}.json"))
def rnd(g, q=0.5): return [[round(x / q) * q for x in pt] for pt in g]
def polys(g):
    if g.is_empty: return []
    if g.geom_type == "Polygon": return [g]
    if g.geom_type in ("MultiPolygon", "GeometryCollection"): return [p for gg in g.geoms for p in polys(gg)]
    return []
def lines(g):
    if g.is_empty: return []
    if g.geom_type == "LineString": return [g]
    if g.geom_type in ("MultiLineString", "GeometryCollection"): return [l for gg in g.geoms for l in lines(gg)]
    return []
def poly_out(p, tol=0.8):
    p = p.simplify(tol)
    if p.is_empty or p.area < 4: return None
    return [rnd(list(p.exterior.coords)[:-1])] + [rnd(list(r.coords)[:-1]) for r in p.interiors if Polygon(r).area > 20]

areas = []  # [kind, rings]
LU = {"residential": "urban", "industrial": "urban", "retail": "urban", "commercial": "urban", "works": "urban", "brownfield": "urban",
      "construction": "urban", "school": "urban", "kindergarten": "urban", "religious": "urban", "railway": "urban", "military": "urban",
      "grass": "lawn", "pitch": "lawn", "playground": "lawn", "garden": "lawn", "dog_park": "lawn", "golf_course": "rough",
      "fairway": "fairway", "green": "fairway", "driving_range": "fairway", "bunker": "sand",
      "farmland": "farm", "meadow": "meadow", "allotments": "farm", "farmyard": "farm", "vineyard": "farm", "greenfield": "meadow",
      "cemetery": "urban", "pedestrian": "paved", "plaza": "paved"}
ORDER = ["farm", "meadow", "urban", "paved", "rough", "lawn", "fairway", "forest", "scrub", "sand", "water"]
for r in load("land_use"):
    kind = LU.get(r["class"])
    if not kind: continue
    for p in polys(projg(wkb.loads(bytes.fromhex(r["geometry"]))).intersection(region)):
        o = poly_out(p)
        if o: areas.append([kind, o])
forest_polys = []
for r in load("land"):
    kind = {"forest": "forest", "wood": "forest", "scrub": "scrub", "sand": "sand", "grass": "lawn"}.get(r["class"])
    if not kind: continue
    for p in polys(projg(wkb.loads(bytes.fromhex(r["geometry"]))).intersection(region)):
        if kind == "forest": forest_polys.append(p)
        o = poly_out(p)
        if o: areas.append([kind, o])
water_lines = []
for r in load("water"):
    g = projg(wkb.loads(bytes.fromhex(r["geometry"]))).intersection(region)
    for p in polys(g):
        o = poly_out(p, 0.5)
        if o: areas.append(["water", o])
    for l in lines(g):
        water_lines.append([{"river": 10, "canal": 5, "stream": 3}.get(r["subtype"], 2), rnd(list(l.simplify(1).coords))])
areas.sort(key=lambda a: ORDER.index(a[0]))

RW = {"motorway": 14, "secondary": 9, "tertiary": 7, "residential": 6, "unclassified": 5, "living_street": 5, "service": 4,
      "unknown": 4, "pedestrian": 4, "track": 3, "path": 2, "footway": 2, "cycleway": 2, "bridleway": 2, "steps": 2}
roads = []
for r in load("seg"):
    if r["subtype"] == "rail":
        wdt, kind = 4, "rail"
    else:
        wdt = RW.get(r["class"]);
        if not wdt: continue
        kind = "path" if r["class"] in ("path", "footway", "cycleway", "bridleway", "steps", "track") else "road"
    for l in lines(projg(wkb.loads(bytes.fromhex(r["geometry"]))).intersection(region)):
        roads.append([kind, wdt, rnd(list(l.simplify(0.8).coords))])

# ---------- buildings ----------
HB = {"house": 7, "detached": 7, "residential": 12, "apartments": 16, "terrace": 9, "garage": 3, "garages": 3, "shed": 3,
      "roof": 5, "grandstand": 11, "industrial": 9, "warehouse": 9, "commercial": 10, "retail": 6, "office": 14, "church": 16,
      "school": 10, "farm": 7, "farm_auxiliary": 5, "service": 4, "shelter": 3, "public": 10}
blds = []; bgeoms = []
for r in load("bld"):
    g = projg(wkb.loads(bytes.fromhex(r["geometry"])))
    if not g.intersects(region): continue
    h = r.get("height"); f = r.get("num_floors")
    try: h = float(h) if h not in (None, "None") else None
    except: h = None
    try: f = float(f) if f not in (None, "None") else None
    except: f = None
    hh = h or (f * 3.1 if f else HB.get(r["class"] or "", 8))
    for p in polys(g.intersection(region)):
        o = poly_out(p, 0.4)
        if not o: continue
        bgeoms.append(p)
        blds.append([round(hh, 1), 1 if r["class"] == "grandstand" else 0, o[0]])
print("buildings", len(blds), "grandstands", sum(b[1] for b in blds))

# ---------- trees ----------
random.seed(11)
corridor = track.buffer(22)
btree = STRtree(bgeoms)
road_buf = shapely.union_all([LineString(r[2]).buffer(r[1] / 2 + 1.5) for r in roads if len(r[2]) > 1 and r[0] == "road"])
blocked = shapely.union_all([corridor, road_buf] + bgeoms)
shapely.prepare(blocked)
trees = []
for r in load("land"):
    g = projg(wkb.loads(bytes.fromhex(r["geometry"])))
    if r["class"] == "tree" and g.geom_type == "Point" and region.contains(g):
        trees.append((g.x, g.y))
    if r["class"] == "tree_row":
        for l in lines(g.intersection(region)):
            for i in range(int(l.length / 8) + 1): p = l.interpolate(i * 8); trees.append((p.x, p.y))
SP = 11.0  # one tree per ~120 m^2 in mapped forests
for fp in forest_polys:
    shapely.prepare(fp)
    a, b, c, d = fp.bounds
    xs = np.arange(a, c, SP); zs = np.arange(b, d, SP)
    for x in xs:
        for zz in zs:
            px, pz = x + random.uniform(-4.5, 4.5), zz + random.uniform(-4.5, 4.5)
            if fp.contains(Point(px, pz)): trees.append((px, pz))
if trees:  # (none mapped at all around some circuits, e.g. Jeddah)
    keep = ~shapely.contains_xy(blocked, np.array(trees)[:, 0], np.array(trees)[:, 1])
    trees = [t for t, kk in zip(trees, keep) if kk]
print("trees", len(trees))
tree_flat = []
for x, zz in trees: tree_flat += [round(x, 1), round(zz, 1), random.randint(0, 99)]

out = {
    "origin": [lon0, lat0], "region": [X0, Z0, X1, Z1],
    "track": [[round(x, 2), round(zz, 2)] for x, zz in list(track.coords)[:-1]],
    "terrain": {"nx": nx, "nz": nz, "step": G, "base": base, "dm": base64.b64encode(terr_dm.tobytes()).decode()},
    "areas": areas, "water": water_lines, "roads": roads, "buildings": blds, "trees": tree_flat,
    "sources": "Tracciato: bacinger/f1-circuits (MIT). Edifici, uso del suolo, alberi, acque, strade: Overture Maps 2026-09-23 (dati OpenStreetMap, ODbL). Quote: Terrarium elevation tiles su AWS Open Data.",
}
s = json.dumps(out, separators=(",", ":"))
open(OUTW, "w").write(s)
print("json MB", round(len(s) / 1e6, 2), "areas", len(areas), "roads", len(roads), "water lines", len(water_lines))
