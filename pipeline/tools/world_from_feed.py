"""Track and minimal world straight from the OpenF1 location feed, for any circuit (no map data needed).

usage: python3 -I world_from_feed.py <openf1_dir> <world.json>
Writes <world.json> (track centre guess, flat terrain, no scenery) and <openf1_dir>/xf.json (OpenF1 -> local metres:
x east-ish, z = -y, origin at the track centroid). build_replay.py then refines the line from every car.
"""
import json, sys, base64
from datetime import datetime
import numpy as np

OF, OUTW = sys.argv[1:3]
J = lambda n: json.load(open(f"{OF}/{n}.json"))
ts = lambda s: datetime.fromisoformat(s).timestamp()
S, R = 0.1, np.array([[1.0, 0.0], [0.0, -1.0]])   # OpenF1 x/y are decimetres; flip y so the map is not mirrored

laps = [l for l in J("laps") if l.get("lap_duration") and l.get("date_start") and not l.get("is_pit_out_lap")]
pit_laps = {(p["driver_number"], p["lap_number"]) for p in J("pit")}
by = {}
for l in laps:
    if (l["driver_number"], l["lap_number"]) in pit_laps or l["lap_number"] < 2: continue
    by.setdefault(l["driver_number"], []).append(l)
M = 1500  # points per lap
acc = []
for n, ls in sorted(by.items(), key=lambda kv: -len(kv[1])):
    med = np.median([l["lap_duration"] for l in ls])
    good = [l for l in ls if abs(l["lap_duration"] - med) < 0.02 * med]
    rows = [r for r in J(f"loc_{n}") if r["x"] or r["y"]]
    t = np.array([ts(r["date"]) for r in rows]); xy = np.array([[r["x"], r["y"]] for r in rows], float)
    o = np.argsort(t); t, xy = t[o], xy[o]
    for l in good[:6]:
        a = ts(l["date_start"]); b = a + l["lap_duration"]
        k = (t >= a - 1) & (t <= b + 1)
        if k.sum() < 50: continue
        tt, p = t[k], xy[k]
        # Catmull-Rom in time between samples (much closer to the real path than straight chords)
        tg = np.arange(a, b, 0.05)
        i = np.clip(np.searchsorted(tt, tg) - 1, 1, len(tt) - 3)
        u = ((tg - tt[i]) / (tt[i + 1] - tt[i]))[:, None]
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        q = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        if abs(np.linalg.norm(q[-1] - q[0])) > 0.02 * s[-1]: continue   # lap does not close: skip
        g = np.linspace(0, s[-1], M, endpoint=False)
        acc.append(np.column_stack([np.interp(g, s, q[:, 0]), np.interp(g, s, q[:, 1])]))
    if len(acc) >= 24: break
print("laps used", len(acc))
# align every lap to the first by circular shift (they start at slightly different points), then average
ref = acc[0]; al = [ref]
for q in acc[1:]:
    d = [np.mean(np.linalg.norm(np.roll(q, -k, 0) - ref, axis=1)) for k in range(-40, 41)]
    al.append(np.roll(q, -(int(np.argmin(d)) - 40), 0))
line = np.median(np.array(al), axis=0)
# light periodic smoothing (Fourier low-pass)
F = np.fft.rfft(line, axis=0); F[300:] = 0; line = np.fft.irfft(F, n=M, axis=0)
loc = (S * (R @ line.T)).T
c = loc.mean(0); loc -= c
json.dump({"s": S, "R": R.tolist(), "t": (-c).tolist()}, open(f"{OF}/xf.json", "w"))
L = np.sum(np.linalg.norm(np.diff(np.vstack([loc, loc[:1]]), axis=0), axis=1))
print("track length m", round(L))
PAD = 400
x0, z0 = loc.min(0) - PAD; x1, z1 = loc.max(0) + PAD
G = 8.0; nx = int((x1 - x0) / G) + 1; nz = int((z1 - z0) / G) + 1
out = {
    "origin": None, "region": [float(x0), float(z0), float(x1), float(z1)],
    "track": [[round(float(x), 2), round(float(z), 2)] for x, z in loc[::3]],
    "terrain": {"nx": nx, "nz": nz, "step": G, "base": 0, "dm": base64.b64encode(np.full(nx * nz, 10, "<u2").tobytes()).decode()},
    "areas": [], "water": [], "roads": [], "buildings": [], "trees": [],
    "sources": "Tracciato ricavato dalle posizioni OpenF1; nessun dato cartografico per lo sfondo.",
}
json.dump(out, open(OUTW, "w"), separators=(",", ":"))
