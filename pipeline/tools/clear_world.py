"""Clear buildings and trees off the track and the pit lane in a world.json.

Map footprints sometimes cover the circuit (bridges, roofs, gantries, misplaced or stale outlines, a slightly
offset fit), so cars drive through houses. Every building that reaches into the track corridor is trimmed back to
the corridor edge, or dropped when little of it is left.

usage: python3 clear_world.py <world.json> <replay.json> [more replay.json sharing this world]
"""
import json, sys
import shapely
from shapely.geometry import LineString, Polygon

TRACK_R = 13.0  # half width 6 m + kerb 1.2 m + a little run-off
PIT_R = 9.0
MIN_AREA = 12.0

wpath, rpaths = sys.argv[1], sys.argv[2:]
W = json.load(open(wpath))
parts = []
for rp in rpaths:
    R = json.load(open(rp))
    line = R.get("racingLine") or []
    if len(line) > 2: parts.append(LineString(line + [line[0]]).buffer(TRACK_R))
    pit = R.get("pitLane") or []
    if len(pit) > 1: parts.append(LineString(pit).buffer(PIT_R))
if not parts and len(W.get("track", [])) > 2:
    parts.append(LineString(W["track"] + [W["track"][0]]).buffer(TRACK_R))
corr = shapely.union_all(parts)
shapely.prepare(corr)

def rnd(c): return [[round(x, 1), round(z, 1)] for x, z in c]

out, dropped, trimmed = [], 0, 0
for b in W["buildings"]:
    h, stand, ring = b[0], b[1], b[2]
    p = Polygon(ring)
    if not p.is_valid: p = p.buffer(0)
    if p.is_empty or not corr.intersects(p): out.append(b); continue
    rest = p.difference(corr)
    pieces = [g for g in getattr(rest, "geoms", [rest]) if g.geom_type == "Polygon" and g.area >= MIN_AREA]
    # a building mostly inside the corridor is a mapping error: drop it whole
    if not pieces or sum(g.area for g in pieces) < 0.35 * p.area: dropped += 1; continue
    trimmed += 1
    for g in pieces:
        g = g.simplify(0.3)
        if g.geom_type == "Polygon" and g.area >= MIN_AREA:
            out.append([h, stand, rnd(list(g.exterior.coords)[:-1])])
W["buildings"] = out
# trees too: the scenery builder only clears them along its own track line, not the pit lane
tr = W.get("trees") or []
if tr:
    xs, zs = tr[0::3], tr[1::3]
    inside = shapely.contains_xy(corr, xs, zs)
    W["trees"] = [v for i, hit in enumerate(inside) if not hit for v in tr[3 * i:3 * i + 3]]
    print("trees removed", int(inside.sum()))
json.dump(W, open(wpath, "w"), separators=(",", ":"))
print(wpath, "dropped", dropped, "trimmed", trimmed, "kept", len(out))
