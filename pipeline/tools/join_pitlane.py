"""Make a replay's pit lane continuous and join its ends to the track.

The lane is rebuilt from where cars drive off the racing line, in 5 m bins that need a few samples each. Where the
samples thin out the lane has holes (drawn as straight cuts across the grass) and its ends stop 10-40 m to the side
of the track (most qualifying feeds), so cars cross the grass to reach it. Holes are filled by following the track
with the lateral offset blended between the two sides of the hole; each loose end gets a smooth ramp back to the
track centre line, before the entry and after the exit. Idempotent.

usage: python3 join_pitlane.py <replay.json> [more replay.json]
"""
import json, sys, math
from shapely.geometry import LineString, Point

STEP = 5.0
LOOSE = 3.0  # an end more than this off the centre line gets a ramp

def frame(C, s):
    L = C.length; s %= L
    p = C.interpolate(s); a = C.interpolate((s - 2) % L); b = C.interpolate((s + 2) % L)
    tx, tz = b.x - a.x, b.y - a.y; n = math.hypot(tx, tz) or 1
    return p.x, p.y, -tz / n, tx / n  # point, left normal

def station_lat(C, p):
    s = C.project(Point(p)); x, z, nx, nz = frame(C, s)
    return s, (p[0] - x) * nx + (p[1] - z) * nz

def fill_holes(C, P):
    L = C.length; out = [P[0]]
    for a, b in zip(P, P[1:]):
        if math.dist(a, b) > 2.5 * STEP:
            sa, la = station_lat(C, a); sb, lb = station_lat(C, b)
            ds = (sb - sa + L / 2) % L - L / 2  # short way round, in driving order
            n = int(abs(ds) / STEP)
            for i in range(1, n):
                u = i / n; x, z, nx, nz = frame(C, sa + ds * u); l = la + (lb - la) * u
                out.append([round(x + nx * l, 2), round(z + nz * l, 2)])
        out.append(b)
    return out

def ramp(C, end, forward):
    """points from the centre line to `end`: before it along the track (entry) or after it (exit)"""
    s = C.project(Point(end)); x, z, nx, nz = frame(C, s)
    lat = (end[0] - x) * nx + (end[1] - z) * nz
    if abs(lat) <= LOOSE: return []
    D = min(max(8 * abs(lat), 60.0), 160.0); n = int(D / STEP)
    pts = []
    for i in range(n):  # u = 0 on the track .. just short of 1 at the lane end
        u = i / n; w = u * u * (3 - 2 * u)
        si = s + D * (1 - u) if forward else s - D * (1 - u)
        x, z, nx, nz = frame(C, si)
        pts.append([round(x + nx * lat * w, 2), round(z + nz * lat * w, 2)])
    return pts[::-1] if forward else pts

for path in sys.argv[1:]:
    R = json.load(open(path)); P = R.get("pitLane") or []; line = R.get("racingLine") or []
    if len(P) < 2 or len(line) < 3: print(path, "no lane"); continue
    C = LineString(line + [line[0]])
    n0 = len(P); P = fill_holes(C, P); filled = len(P) - n0
    head, tail = ramp(C, P[0], False), ramp(C, P[-1], True)
    if head or tail or filled:
        R["pitLane"] = head + P + tail
        json.dump(R, open(path, "w"), separators=(",", ":"))
    print(path, "holes filled", filled, "entry ramp", len(head), "exit ramp", len(tail))
