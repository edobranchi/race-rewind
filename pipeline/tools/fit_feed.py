"""Fit a similarity transform OpenF1 (x,y) -> local metres (x east, z south) by ICP against the mapped centreline.
usage: python3 fit_feed.py <openf1_dir> <world.json>   (writes <openf1_dir>/xf.json)"""
import json, math, sys
import numpy as np

OF = sys.argv[1]; DATA = sys.argv[2]
D = json.load(open(DATA))
tr = np.array(D["track"], float)
# dense closed centreline every 1 m
seg = np.vstack([tr, tr[:1]])
dense = []
for a, b in zip(seg[:-1], seg[1:]):
    n = max(1, int(np.hypot(*(b - a))))
    for t in np.arange(n) / n: dense.append(a + (b - a) * t)
dense = np.array(dense)

def nearest(p):
    out = np.empty(len(p), int)
    for i in range(0, len(p), 2000):
        d = ((p[i:i + 2000, None, :] - dense[None]) ** 2).sum(-1)
        out[i:i + 2000] = d.argmin(1)
    return out

from datetime import datetime, timedelta
ses = json.load(open(f"{OF}/sessions.json"))[0]
a = datetime.fromisoformat(ses["date_start"]); b = datetime.fromisoformat(ses["date_end"])
mid = a + (b - a) / 2
w0, w1 = (mid - timedelta(minutes=10)).isoformat()[:16], (mid + timedelta(minutes=10)).isoformat()[:16]
pts = []; used = 0
for d in json.load(open(f"{OF}/drivers.json")):
    p_ = [(r["x"], r["y"]) for r in json.load(open(f"{OF}/loc_{d['driver_number']}.json"))
          if (r["x"] or r["y"])]
    # only cars lapping in that window (not parked in the garage or retired)
    # moving samples only: a car parked in the garage after retiring sits far off the track
    q_ = np.array(p_, float).reshape(-1, 2)
    q_ = q_[1:][np.hypot(*np.diff(q_, axis=0).T) > 20] if len(q_) > 1 else q_[:0]
    if len(q_) < 200 or q_.std(0).max() < 1000: continue
    pts += [tuple(v) for v in q_]; used += 1
    if used == 6: break
# the whole session (a red flag or safety car can empty any fixed window), thinned to ~1800 samples
allp = np.array(pts, float); thin = max(1, len(allp) // 1800)
src = allp[::thin]
dense_full = dense; dense = dense[::3]
print("samples", len(src))

def umeyama(a, b, refl, fix_s=None):
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    d = np.eye(2)
    if (np.linalg.det(U @ Vt) < 0) != refl: d[1, 1] = -1
    R = U @ d @ Vt
    s = fix_s if fix_s else np.trace(np.diag(S) @ d) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma

best = None
c0 = src.mean(0); t0 = dense.mean(0)
for refl in (False, True):
    for ang in range(0, 360, 10):
        th = math.radians(ang); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        if refl: R = R @ np.diag([1, -1])
        s = 0.1; t = t0 - s * R @ c0
        # rigid at the feed's own scale (decimetres) first, so the fit can't shrink into one part of the map
        for it in range(25):
            q = (s * (R @ src.T)).T + t
            nn = dense[nearest(q)]
            s, R, t = umeyama(src, nn, refl, 0.1 if it < 18 else None)
        q = (s * (R @ src.T)).T + t
        err = np.sqrt(((q - dense[nearest(q)]) ** 2).sum(1))
        # OpenF1 x/y are decimetres: a fit far from scale 0.1 has latched onto the wrong part of the map
        if abs(s - 0.1) > 0.008: continue
        score = np.median(err) + 0.25 * np.percentile(err, 95)
        if best is None or score < best[0]: best = (score, s, R, t, refl, ang, np.percentile(err, 95))
s, R, t = best[1], best[2], best[3]
dense = dense_full; src = allp[::max(1, len(allp) // 9000)]
for it in range(10):
    q = (s * (R @ src.T)).T + t; s, R, t = umeyama(src, dense[nearest(q)], best[4])
q = (s * (R @ src.T)).T + t; err = np.sqrt(((q - dense[nearest(q)]) ** 2).sum(1))
best = (np.median(err), s, R, t, best[4], best[5], np.percentile(err, 95))
if abs(s - 0.1) > 0.008: best = (99.0,) + best[1:]  # refinement drifted off scale: report as a failed fit
print("median err m", best[0], "p95", best[6], "scale", best[1], "refl", best[4], "init", best[5])
json.dump({"s": best[1], "R": best[2].tolist(), "t": best[3].tolist()}, open(f"{OF}/xf.json", "w"))
