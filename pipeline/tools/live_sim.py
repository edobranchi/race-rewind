"""Stream a finished OpenF1 race through the live engine in time order, as if it were live, and compare with the
offline replay. usage: python3 -I live_sim.py <race_dir> <track.npz> <offline_out_dir> <delay_s> [latency_s]"""
import json, sys, base64, gzip, time
from datetime import datetime
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from live_engine import LiveEngine

OF, TRK, OFF, DELAY = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
LAT = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0   # how late each sample reaches us
J = lambda n: json.load(open(f"{OF}/{n}.json"))
ts = lambda s: datetime.fromisoformat(s).timestamp()
SES = J("sessions")[0]; T0, T1 = ts(SES["date_start"]), ts(SES["date_end"])
nums = sorted(d["driver_number"] for d in J("drivers"))
ev = []
for n in nums:
    for r in J(f"loc_{n}"):
        if r["session_key"] == SES["session_key"]: ev.append((ts(r["date"]), 0, n, r["x"], r["y"], 0))
    for r in J(f"car_{n}"):
        if r["session_key"] == SES["session_key"]: ev.append((ts(r["date"]), 1, n, r["speed"], r["throttle"], r["brake"]))
for r in J("race_control"): ev.append((ts(r["date"]), 2, 0, r["message"], 0, 0))
ev.sort(key=lambda e: e[0])
print("events", len(ev))
E = LiveEngine(TRK, nums, delay=DELAY)
RP = json.load(open(f"{OFF}/replay.json")); HZ = RP["hz"]; NF = RP["frames"]
q = np.frombuffer(gzip.decompress(base64.b64decode(open(f"{OFF}/replay-pos.txt").read())), "<i2").astype(np.int32)
OFFXY = np.cumsum(q.reshape(NF, -1), 0).reshape(NF, len(nums), 2) / 10.0
T_off = np.load(f"{OFF}/track.npz"); shift = T_off["xf_t"] - E.TT   # both frames share scale/rotation
LIVE = np.full((NF, len(nums), 2), np.nan)
k = 0; t_cpu = time.time(); steps = 0
for now in np.arange(T0 + 5, T1, 1.0):           # the engine runs once a second
    while k < len(ev) and ev[k][0] + LAT <= now:
        tt, kind, n, a, b, c = ev[k]; k += 1
        if kind == 0: E.add_location(n, tt, a, b)
        elif kind == 1: E.add_car_data(n, tt, a, b, c)
        else: E.add_race_control(tt, a)
    tg, out = E.step(now); steps += 1
    fr = np.round((tg - T0) * HZ).astype(int)
    for n, xy in out.items():
        m = (fr >= 0) & (fr < NF); LIVE[fr[m], nums.index(n)] = xy[m] + shift
cpu = (time.time() - t_cpu) / steps
# compare where the offline replay has the car moving on track
v = np.linalg.norm(np.gradient(OFFXY, axis=0), axis=2) * HZ
ok = (v > 15) & ~np.isnan(LIVE[..., 0])
err = np.linalg.norm(LIVE - OFFXY, axis=2)[ok]
jump = np.linalg.norm(np.diff(LIVE, axis=0), axis=2) * HZ; vv = v[1:]
mj = ok[1:] & ~np.isnan(jump)
acc = np.abs(np.diff(np.linalg.norm(np.diff(LIVE, axis=0), axis=2), axis=0)) * HZ * HZ
print(f"delay {DELAY}s latency {LAT}s: distance from offline replay median {np.median(err):.2f} m, p95 {np.percentile(err, 95):.2f} m, p99 {np.percentile(err, 99):.2f} m")
print(f"  speed-change p99 between frames {np.nanpercentile(acc[ok[2:]], 99):.1f} m/s2, cpu per 1 s step {cpu * 1000:.0f} ms for {len(nums)} cars")
np.save(f"{OFF}/live_{DELAY:g}.npy", LIVE.astype(np.float32))
