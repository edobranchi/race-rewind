"""Build replay.json + replay-pos.txt for the Monza 3D replay from OpenF1 data.

usage: python3 -I build_replay.py <openf1_dir> <monza-data.json> <out_dir>
openf1_dir holds drivers/laps/position/race_control/pit .json and loc_<n>.json (OpenF1 /v1/<endpoint>?session_key=11361),
plus xf.json from fit.py (similarity transform OpenF1 x,y -> local metres).
"""
import json, sys, math
from datetime import datetime
import numpy as np

OF, DATA, OUT = sys.argv[1:4]
HZ = 4
ts = lambda s: datetime.fromisoformat(s).timestamp()
import gzip as _gz, os as _os
# per-car files are gzipped after a build to save disk; both forms load
J = lambda n: json.load(_gz.open(f"{OF}/{n}.json.gz", "rt") if _os.path.exists(f"{OF}/{n}.json.gz") else open(f"{OF}/{n}.json"))
# session window and labels come from OpenF1 sessions/meetings (fetch_openf1.py saves them), nothing per-track here
SES = J("sessions")[0]; MEET = (J("meetings") or [{}])[0]
SESSION = SES["session_key"]
IS_RACE = SES.get("session_type") == "Race"
IS_QUALI = SES.get("session_type") == "Qualifying"
T0 = ts(SES["date_start"]); T1 = ts(SES["date_end"])
# a red flag can push the finish past the scheduled end: run to the chequered flag + 4 min
_chq = [ts(r["date"]) for r in J("race_control") if r["message"].startswith("CHEQUERED")]
if _chq: T1 = max(T1, max(_chq) + 240)

xf = J("xf"); S, R, TT = xf["s"], np.array(xf["R"]), np.array(xf["t"])
def to_local(xy): return (S * (R @ xy.T)).T + TT

# ---------- mapped centreline, dense every 1 m ----------
tr = np.array(json.load(open(DATA))["track"], float)
seg = np.vstack([tr, tr[:1]]); dense = []
for a, b in zip(seg[:-1], seg[1:]):
    n = max(1, int(round(np.hypot(*(b - a)))))
    for t in np.arange(n) / n: dense.append(a + (b - a) * t)
C = np.array(dense); NC = len(C); LEN = NC  # ~1 m spacing
TAN = np.roll(C, -3, 0) - np.roll(C, 3, 0); TAN /= np.linalg.norm(TAN, axis=1)[:, None]
NRM = np.column_stack([-TAN[:, 1], TAN[:, 0]])

def nearest_idx(p):
    out = np.empty(len(p), int)
    for i in range(0, len(p), 1500):
        out[i:i + 1500] = ((p[i:i + 1500, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    return out

def tel_for(n):
    rows = [r for r in J(f"car_{n}") if r["session_key"] == SESSION]
    t = np.array([ts(r["date"]) for r in rows]); o = np.argsort(t)
    arr = np.array([[r["speed"] or 0, r["throttle"] or 0, r["brake"] or 0, r["n_gear"] or 0, r.get("rpm") or 0] for r in rows], float)[o]
    t = t[o]
    # 104 in throttle/brake marks a frozen packet (speed then stays stuck too): drop those
    ok = (arr[:, 1] <= 100) & (arr[:, 2] <= 100)
    return t[ok], arr[ok]

# ---------- drivers ----------
drivers = sorted(J("drivers"), key=lambda d: d["driver_number"])
all_nums = [d["driver_number"] for d in drivers]
# entrants with no location data at all (did not start) keep an empty grid slot but get no car
# (also cars whose telemetry never shows them driving: the feed sends frozen packets for cars left in the garage)
nodata = [n for n in all_nums if not any(r["session_key"] == SESSION and (r["x"] or r["y"]) for r in J(f"loc_{n}"))
          or (tel_for(n)[1][:, 0] > 60).sum() < 100]
drivers = [d for d in drivers if d["driver_number"] not in nodata]
nums = [d["driver_number"] for d in drivers]
if nodata: print("no location data", nodata)
# colours: team colour, nudged apart where two teams look alike; second car of a team gets a white stripe
PAL = {"McLaren": "#f47600", "Red Bull Racing": "#2b45b8", "Audi": "#8c1d3c", "Alpine": "#ff6fb5", "Cadillac": "#55606b",
       "Mercedes": "#00d7b6", "Aston Martin": "#1f8a5f", "Ferrari": "#e8101f", "Racing Bulls": "#9db6ff", "Williams": "#14a4e6",
       "Haas F1 Team": "#e6e8ea"}
seen = {}
cars = []
for d in drivers:
    team = d["team_name"]; k = seen.get(team, 0); seen[team] = k + 1
    cars.append({"color": PAL.get(team, "#" + (d.get("team_colour") or "888888")), "stripe": k, "code": d.get("name_acronym") or ""})

# ---------- positions resampled on a common 4 Hz clock ----------
# feed timestamps jitter by ~0.1 s (apparent speed swings 170-590 km/h between samples), so a plain interpolation
# makes cars surge and brake; a local quadratic fit over ~2 s of samples keeps the path and evens out the speed
def smooth_resample(t, xy, tg, sigma=0.8, K=28):
    c = np.searchsorted(t, tg)
    idx = np.clip(c[:, None] + np.arange(-K // 2, K // 2)[None], 0, len(t) - 1)
    u = t[idx] - tg[:, None]
    w = np.exp(-0.5 * (u / sigma) ** 2) + 1e-9
    A = np.stack([np.ones_like(u), u, u * u], -1)
    M = np.einsum("nk,nki,nkj->nij", w, A, A) + np.eye(3) * 1e-6
    out = np.empty((len(tg), 2))
    for d in range(2):
        b = np.einsum("nk,nki,nk->ni", w, A, xy[idx, d])
        out[:, d] = np.linalg.solve(M, b[..., None])[:, 0, 0]
    # inside a gap in the samples the local fit has nothing to hold on to: interpolate linearly there
    far_ = np.abs(u).min(1) > 1.5
    if far_.any():
        for d in range(2): out[far_, d] = np.interp(tg[far_], t, xy[:, d])
    # outside the data range hold the first/last sample
    out[tg <= t[0]] = xy[0]; out[tg >= t[-1]] = xy[-1]
    return out

NF = int((T1 - T0) * HZ) + 1
tgrid = T0 + np.arange(NF) / HZ
XY = np.zeros((NF, len(nums), 2))
raw = {}; stalled = {}
for j, n in enumerate(nums):
    rows = [r for r in J(f"loc_{n}") if r["session_key"] == SESSION and (r["x"] or r["y"])]
    t = np.array([ts(r["date"]) for r in rows]); xy = to_local(np.array([[r["x"], r["y"]] for r in rows], float))
    o = np.argsort(t); t, xy = t[o], xy[o]
    keep = np.concatenate([[True], np.diff(t) > 0]); t, xy = t[keep], xy[keep]
    # the feed parks cars on a fixed placeholder point far off the circuit at times (e.g. on the grid): drop
    # anything over 100 m from the track (pit lane and garages are closer than that)
    C5 = C[::5]; far = np.zeros(len(xy), bool)
    for i0 in range(0, len(xy), 2000):
        far[i0:i0 + 2000] = (((xy[i0:i0 + 2000, None, :] - C5[None]) ** 2).sum(-1).min(1)) > 100 ** 2
    t, xy = t[~far], xy[~far]
    # the feed sometimes freezes a moving car for a second or so (same x/y repeated), then jumps ahead:
    # drop samples that barely move while the car is clearly moving over the surrounding ~4 s
    w = 8
    a_ = np.clip(np.arange(len(t)) - w, 0, len(t) - 1); b_ = np.clip(np.arange(len(t)) + w, 0, len(t) - 1)
    vwin = np.linalg.norm(xy[b_] - xy[a_], axis=1) / np.maximum(t[b_] - t[a_], 1e-3)
    step = np.concatenate([[np.inf], np.linalg.norm(np.diff(xy, axis=0), axis=1)])
    dts = np.concatenate([[1], np.diff(t)])
    stall = (vwin > 12) & (step < 0.3 * vwin * dts)
    stalled[n] = int(stall.sum())
    t, xy = t[~stall], xy[~stall]
    # where samples are missing for a while (placeholders dropped above: a car standing on the grid, in the pit
    # lane), hold the last real position instead of sliding slowly across the gap
    gap = np.where(np.diff(t) > 3)[0]
    if len(gap):
        t = np.insert(t, gap + 1, t[gap + 1] - 0.3); xy = np.insert(xy, gap + 1, xy[gap], axis=0)
    raw[n] = (t, xy)

# ---------- cars the position feed lost for long stretches: distance from speed telemetry, anchored at lap starts ----------
# (the feed can go missing for most of a race, e.g. Monaco 2026; car_data and lap times usually remain)
laps_by = {}
grid_rank = {r["driver_number"]: k for k, r in enumerate(sorted(J("starting_grid"), key=lambda r: r.get("position") or 99))}
for l in J("laps"):
    if l.get("date_start") and l["driver_number"] in nums: laps_by.setdefault(l["driver_number"], []).append((l["lap_number"], ts(l["date_start"])))
fin = []
for n in nums:
    tr_, xy_ = raw[n]
    for ln, tl in laps_by.get(n, []):
        if ln < 2 or not len(tr_): continue
        k = np.searchsorted(tr_, tl)
        if 0 < k < len(tr_) and tr_[k] - tr_[k - 1] < 0.6:
            p_ = xy_[k - 1] + (xy_[k] - xy_[k - 1]) * (tl - tr_[k - 1]) / (tr_[k] - tr_[k - 1])
            fin.append(int(nearest_idx(p_[None])[0]))
s_fin = None
if fin:
    ang = np.array(fin) * 2 * np.pi / NC
    s_fin = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * NC / (2 * np.pi)
filled = {}
for j, n in enumerate(nums):
    t, xy = raw[n]
    ls = sorted(laps_by.get(n, []))
    # (races only: in qualifying and practice a car missing for minutes is simply in the garage)
    if s_fin is None or len(ls) < 3 or not IS_RACE: continue
    tt_, tl_ = tel_for(n)
    if len(tt_) < 100: continue
    lt = np.array([x[1] for x in ls]); lnum = np.array([x[0] for x in ls])
    # samples every 0.25 s from the start to the last lap start + one lap, where the feed has nothing within 2 s;
    # lap 1 starts from the car's grid slot (8 m apart, behind the line), every later lap from the line
    slot = grid_rank.get(n, len(grid_rank))
    s_grid = s_fin - 4 - 8 * slot
    tq = np.arange(lt[0] if lnum[0] == 1 else lt[1], lt[-1] + np.median(np.diff(lt)), 0.25)
    if len(t):
        k = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
        near = np.minimum(np.abs(t[k] - tq), np.abs(t[k - 1] - tq)) < 2
    else:
        near = np.zeros(len(tq), bool)
    # only long holes (over 90 s: longer than a pit stop or a grid wait, which the gap handling above covers)
    miss = ~near; run_id = np.cumsum(np.concatenate([[1], np.diff(miss.astype(int)) != 0]))
    run_len = np.bincount(run_id)[run_id]
    miss &= run_len > 4 * 90
    if not miss.any(): continue
    dist = np.concatenate([[0], np.cumsum(np.diff(tt_) * tl_[1:, 0] / 3.6)])
    dq = np.interp(tq, tt_, dist)
    # lap k of the car spans [lt[i], lt[i+1]] and covers exactly one lap length: rescale the speed integral to that
    dl = np.interp(lt, tt_, dist)
    if lnum[0] == 1:  # anchors: lap 1 starts on the grid slot, lap m at the line + (m - 1) laps
        anc = np.where(lnum == 1, s_grid, s_fin + (lnum - 1) * LEN).astype(float)
    else:
        anc = s_fin + (lnum - lnum[1]).astype(float) * LEN
    seg = np.clip(np.searchsorted(lt, tq, side="right") - 1, 0, len(lt) - 1)
    inner = seg < len(lt) - 1; sg1 = np.minimum(seg + 1, len(lt) - 1)
    frac = np.where(inner, (dq - dl[seg]) / np.maximum(dl[sg1] - dl[seg], 1), (dq - dl[-1]) / max(float(np.median(np.diff(dl))), 1))
    frac = np.clip(frac, 0, 1)
    st = np.where(inner, anc[seg] + frac * (anc[sg1] - anc[seg]), anc[-1] + frac * LEN)
    pq = C[np.floor(st).astype(int) % NC] + NRM[np.floor(st).astype(int) % NC] * np.where(tq < lt[0] + 3, 2.5 * (1 if slot % 2 == 0 else -1), 0)[:, None]
    # before the start nothing comes from the feed either: hold the car on its grid slot
    if lnum[0] == 1 and (not len(t) or t[0] > lt[0] - 5):
        tpre = np.arange(T0, lt[0], 1.0)
        ig = int(np.floor(s_grid)) % NC
        tq = np.concatenate([tpre, tq]); pq = np.vstack([np.repeat((C[ig] + NRM[ig] * 2.5 * (1 if slot % 2 == 0 else -1))[None], len(tpre), 0), pq])
        miss = np.concatenate([np.ones(len(tpre), bool), miss])
    keep = miss
    t2 = np.concatenate([t, tq[keep]]); xy2 = np.vstack([xy.reshape(-1, 2), pq[keep]])
    o = np.argsort(t2); raw[n] = (t2[o], xy2[o]); filled[n] = int(keep.sum())
if filled: print("positions rebuilt from speed + lap times (samples)", filled)
for j, n in enumerate(nums):
    t, xy = raw[n]
    XY[:, j] = smooth_resample(t, xy, tgrid) if len(t) else 0
print("frames", NF, "cars", len(nums), "stalled samples dropped", sum(stalled.values()))

# ---------- pit intervals ----------
# OpenF1 pit.date is the pit lane exit in some races and the entry in others: per race, pick the reading in
# which cars are slower (pit lane speed limit) over the lane interval
pit_rows = [p for p in J("pit") if p.get("lane_duration") and p.get("date") and p["driver_number"] in nums]
_tel = {}
def _med_speed(n, a, b):
    if n not in _tel: _tel[n] = tel_for(n)
    tt_, tl_ = _tel[n]; m_ = (tt_ >= a) & (tt_ <= b)
    return float(np.median(tl_[m_, 0])) if m_.sum() > 5 else np.nan
v_exit = np.nanmedian([_med_speed(p["driver_number"], ts(p["date"]) - p["lane_duration"], ts(p["date"])) for p in pit_rows] or [np.nan])
v_entry = np.nanmedian([_med_speed(p["driver_number"], ts(p["date"]), ts(p["date"]) + p["lane_duration"]) for p in pit_rows] or [np.nan])
date_is_entry = bool(v_entry < v_exit)
print("pit date is", "entry" if date_is_entry else "exit", "median km/h exit-reading", round(v_exit, 1) if v_exit == v_exit else None, "entry-reading", round(v_entry, 1) if v_entry == v_entry else None)
pits = []
for p in pit_rows:
    t1 = ts(p["date"]) + (p["lane_duration"] if date_is_entry else 0); t0 = t1 - p["lane_duration"]
    pits.append([nums.index(p["driver_number"]), round(t0 - T0, 1), round(t1 - T0, 1)])

# ---------- green-flag windows ----------
rc = J("race_control")
ev = []
for r in rc:
    m, t = r["message"], ts(r["date"]) - T0
    if m.startswith("RACE START") or m.startswith("STANDING START") or m.startswith("ROLLING START") or m.startswith("VSC ENDING") or m.startswith("SAFETY CAR IN THIS LAP"):
        st = "green"
    elif m.startswith("SAFETY CAR DEPLOYED"): st = "sc"
    elif m.startswith("VSC DEPLOYED"): st = "vsc"
    elif m.startswith("RED FLAG"): st = "red"
    elif m == "SESSION STARTED" and ev and ev[-1][1] == "red": st = "restart"
    elif m == "SESSION STARTED" and IS_RACE and not any(e[1] == "green" for e in ev): st = "formation"
    elif m.startswith("CHEQUERED"): st = "chequered"
    else: continue
    if t < 0: continue
    ev.append([round(t, 1), st])
ev.sort(key=lambda e: e[0])
# repeated states (two standing starts): keep the last one
clean = []
for e in ev:
    if clean and clean[-1][1] == e[1]: clean[-1] = e; continue
    clean.append(e)
# many races carry no "RACE START" message: the formation lap start stands in, and the green is then moved
# to the moment the front of the field launches from standstill (telemetry), within 10 min of it
def launch_time(t_from, t_to):
    cand = []
    for n_ in nums:
        tt_, tel_ = tel_for(n_)
        m_ = (tt_ - T0 > t_from) & (tt_ - T0 < t_to)
        tq, vq = tt_[m_] - T0, tel_[m_, 0]
        # a real start, not a formation-lap departure: from standstill to over 120 km/h within 6 s
        for g_ in np.where((vq[:-1] < 5) & (vq[1:] >= 5))[0]:
            if np.any((tq > tq[g_]) & (tq < tq[g_] + 6) & (vq > 120)): cand.append(tq[g_]); break
    cand = np.sort(cand)
    # the front of the field reacts first; take the median of the first four cars to pull away
    return round(float(np.median(cand[:4])), 2) if len(cand) >= 4 else None
if any(e[1] == "formation" for e in clean):
    k_ = [e[1] for e in clean].index("formation")
    if any(e[1] == "green" for e in clean[:k_]): clean.pop(k_)
    else:
        nxt = clean[k_ + 1][0] if k_ + 1 < len(clean) else clean[k_][0] + 600
        lo_ = launch_time(clean[k_][0] - 30, min(clean[k_][0] + 600, nxt))
        if lo_ is not None: clean[k_] = [lo_, "green"]
        else:  # rolling start behind the safety car: green from the formation lap on, under SC
            clean[k_] = [clean[k_][0], "sc"]
            clean = [e for k, e in enumerate(clean) if k == 0 or clean[k - 1][1] != e[1]]
# a restart after a red flag with no green message after it (Interlagos 2023): the standing start found from
# telemetry, or a rolling restart at the restart message itself
for k_, (t_, st_) in enumerate(clean):
    if IS_RACE and st_ == "restart" and (k_ + 1 >= len(clean) or clean[k_ + 1][1] not in ("green", "sc", "vsc", "red")):
        lo_ = launch_time(t_ - 30, t_ + 900)
        clean.insert(k_ + 1, [lo_ if lo_ is not None else t_ + 1, "green"])
        break
flags = [[0, "pre"]] + clean
# no chequered flag message (Monza 2023): the flag falls when the leader completes the last lap
if IS_RACE and not any(st == "chequered" for _, st in flags):
    _le = [ts(l["date_start"]) + l["lap_duration"] - T0 for l in J("laps") if l.get("date_start") and l.get("lap_duration")]
    if _le: flags.append([round(max(_le), 1), "chequered"])
# qualifying: Q1/Q2/Q3 (sprint: SQ1-SQ3) from the session status messages, a break between them, red flags inside
PHASES = []
if IS_QUALI:
    flags = [[0, "pre"]]
    sq = "Sprint" in SES.get("session_name", "")
    for r in sorted(rc, key=lambda r: r["date"]):
        m, t = r["message"], round(ts(r["date"]) - T0, 1)
        ph = r.get("qualifying_phase")
        # some feeds leave the phase out of the first start: a start after a break (not a red flag) opens the next phase
        if m == "SESSION STARTED" and not ph and flags[-1][1] != "red": ph = len(PHASES) + 1
        if m == "SESSION STARTED":
            if ph and not any(p[2].endswith(str(ph)) for p in PHASES): PHASES.append([t, None, ("SQ" if sq else "Q") + str(ph)])
            st = "green"
        elif m == "SESSION RESUMED": st = "green"
        elif m.startswith("RED FLAG"): st = "red"
        elif m.startswith("CHEQUERED"):
            st = "break"
            if PHASES and PHASES[-1][1] is None: PHASES[-1][1] = t
        else: continue
        if t >= 0 and flags[-1][1] != st: flags.append([t, st])
    if flags[-1][1] == "break": flags[-1][1] = "chequered"
    for p in PHASES:
        if p[1] is None: p[1] = round(T1 - T0, 1)
    print("phases", PHASES)
print("flags", flags)

def flagAt(trel):
    st = "pre"
    for t, s_ in flags:
        if t <= trel: st = s_
    return st
def is_green(trel):
    st = "pre"
    for t, s in flags:
        if t <= trel: st = s
    return st == "green"
green_mask = np.array([is_green(t - T0) for t in tgrid])
# skip the first lap after each start and the laps around pit stops
for t, s in flags:
    if s == "green": green_mask[(tgrid - T0 >= t) & (tgrid - T0 < t + 100)] = False
if SES.get("session_type") != "Race":  # practice/qualifying (track model before a race): every moving car counts
    green_mask[:] = True

# ---------- station / lateral offset for racing samples ----------
def station_lateral(p, idx):
    d = p - C[idx]
    return idx + (d * TAN[idx]).sum(1), (d * NRM[idx]).sum(1)

sub = slice(None, None, 2)  # 2 Hz is plenty for statistics
st_all, lat_all, sp_all = [], [], []
for j, n in enumerate(nums):
    inpit = np.zeros(NF, bool)
    for k, a, b in pits:
        if k == j: inpit |= (tgrid - T0 >= a - 8) & (tgrid - T0 <= b + 8)
    m = green_mask & ~inpit
    # moving cars only (> 30 m/s), i.e. not retired or crawling
    v = np.linalg.norm(np.gradient(XY[:, j], axis=0), axis=1) * HZ
    m &= v > 30
    # line geometry from the raw samples (smoothing in x/z would round off the chicanes); speed from the smoothed track
    tr, xyr = raw[n]
    fr = np.clip(np.round((tr - T0) * HZ).astype(int), 0, NF - 1)
    mr = m[fr] & (tr >= T0) & (tr <= T1)
    p = xyr[mr][sub]
    if not len(p): continue
    idx = nearest_idx(p)
    s, l = station_lateral(p, idx)
    ok = np.abs(l) < 25
    st_all.append(s[ok] % LEN); lat_all.append(l[ok]); sp_all.append(v[fr[mr]][sub][ok] * 3.6)
st_all = np.concatenate(st_all); lat_all = np.concatenate(lat_all); sp_all = np.concatenate(sp_all)
print("racing samples", len(st_all))

BIN = 2.0
nb = int(LEN / BIN)
b = (st_all / BIN).astype(int) % nb
med = np.zeros(nb); lo = np.zeros(nb); hi = np.zeros(nb); cnt = np.zeros(nb)
for i in range(nb):
    v = lat_all[b == i]; cnt[i] = len(v)
    if len(v) >= 20: med[i], lo[i], hi[i] = np.percentile(v, [50, 2, 98])
    else: med[i] = lo[i] = hi[i] = np.nan
def fill_smooth(a, w):
    idx = np.arange(nb); good = ~np.isnan(a)
    a = np.interp(idx, idx[good], a[good], period=nb)
    k = np.ones(w) / w
    return np.convolve(np.concatenate([a[-w:], a, a[:w]]), k, "same")[w:-w]
spd_b = np.array([np.median(sp_all[b == i]) if (b == i).sum() >= 20 else np.nan for i in range(nb)])
spd_s = fill_smooth(spd_b, 5)
med_s = fill_smooth(med, 3); lo_s = fill_smooth(lo, 9); hi_s = fill_smooth(hi, 9)
print("bins", nb, "min count", int(cnt.min()), "median lateral |m|", round(float(np.median(np.abs(med_s))), 2),
      "car-to-car spread median", round(float(np.median(hi_s - lo_s)), 1), "mid offset p95", round(float(np.percentile(np.abs((hi_s + lo_s) / 2), 95)), 1))

def at_station(s_arr, lat_arr):
    i = np.floor(s_arr).astype(int) % NC; f = (s_arr - np.floor(s_arr))[:, None]
    base = C[i] * (1 - f) + C[(i + 1) % NC] * f
    nrm = NRM[i] * (1 - f) + NRM[(i + 1) % NC] * f
    return base + nrm * lat_arr[:, None]
s_bins = (np.arange(nb) + 0.5) * BIN
racing_line = at_station(s_bins, med_s)
# the feed puts every car on practically the same line (spread ~0.2 m), so this line is also the best track centre we have

# the race start = the first green of the race (even when the formation ran behind the safety car), plus restarts
_g = [t for t, st in flags if st == "green"]
# a "RACE START" that is really a formation behind the safety car, followed by the standing start: no launch from
# standstill after it, but one after the next green within 15 min -> that one is the start
if IS_RACE and len(_g) > 1 and _g[1] - _g[0] < 900 and launch_time(_g[0] - 30, _g[1] - 5) is None and launch_time(_g[1] - 30, _g[1] + 300) is not None:
    flags = [[t, "sc"] if (t == _g[0] and st == "green") else [t, st] for t, st in flags]
    flags = [f for k, f in enumerate(flags) if k == 0 or flags[k - 1][1] != f[1]]
    _g = _g[1:]
    print("formation behind the safety car; start at", _g[0])
STARTS = _g[:1] + [t for k, (t, st) in enumerate(flags) if st == "green" and flags[k - 1][1] == "restart" and (not _g or t != _g[0])]
if not IS_RACE: STARTS = []
print("starts", STARTS)

# ---------- starting grid: one box per entry, left empty for pit-lane starters and non-starters ----------
# order: OpenF1 starting_grid (fetched with the qualifying session_key) when present, else running order 2 s before the start
import os
sg = sorted(J("starting_grid"), key=lambda r: r["position"]) if os.path.exists(f"{OF}/starting_grid.json") else []
first_start = STARTS[0] if STARTS else 0.0
def pos_at(n, t_rel):
    tr_, xy_ = raw[n]; t_abs = T0 + t_rel
    k = np.searchsorted(tr_, t_abs)
    if k == 0 or k >= len(tr_): return None
    if tr_[k] - tr_[k - 1] > 10: return xy_[k - 1] if np.allclose(xy_[k], xy_[k - 1]) else None  # held through a feed gap
    return xy_[k - 1] + (xy_[k] - xy_[k - 1]) * (t_abs - tr_[k - 1]) / (tr_[k] - tr_[k - 1])
grid_nums = [r["driver_number"] for r in sg if r["driver_number"] in all_nums]
if not grid_nums:  # fallback: last classified position published before the start
    last = {}
    for r in sorted(J("position"), key=lambda r: r["date"]):
        if ts(r["date"]) - T0 <= first_start: last[r["driver_number"]] = r["position"]
    grid_nums = sorted((n for n in last if n in all_nums), key=last.get)
grid_nums += [n for n in all_nums if n not in grid_nums]
if not IS_RACE: grid_nums = []
grid = []
for n in grid_nums:
    if n in nodata: grid.append([-1, "dns"]); continue
    p = pos_at(n, first_start - 2)
    if p is None: state = "dns"
    else:
        d = np.min(np.linalg.norm(racing_line - p, axis=1))
        state = "pit" if d > 10 else ""
    grid.append([nums.index(n), state])
print("grid", [(drivers[j]["name_acronym"] if j >= 0 else "-", st) for j, st in grid])
# ---------- re-sample positions along that line ----------
# smoothing x/z directly cuts the chicanes; instead each sample becomes (distance along the line, sideways offset),
# both are smoothed in time, and the car is put back on the line. Far from the line (garages) the x/z smoothing stays.
seg2 = np.vstack([racing_line, racing_line[:1]]); dl = []; seg_start = []
for a, b2 in zip(seg2[:-1], seg2[1:]):
    n2 = max(1, int(round(np.hypot(*(b2 - a))))); seg_start.append(len(dl))
    for tt in np.arange(n2) / n2: dl.append(a + (b2 - a) * tt)
LC = np.array(dl); NL = len(LC)
LT = np.roll(LC, -2, 0) - np.roll(LC, 2, 0); LT /= np.linalg.norm(LT, axis=1)[:, None]
LN = np.column_stack([-LT[:, 1], LT[:, 0]])
RAD = 30
gx0, gz0 = LC.min(0) - RAD - 2
gw, gh = (np.ceil((LC.max(0) - LC.min(0)) + 2 * RAD + 4)).astype(int)
bd = np.full((gh, gw), np.inf); bi = np.zeros((gh, gw), int)
oz, ox = np.mgrid[-RAD:RAD + 1, -RAD:RAD + 1]
for i, pnt in enumerate(LC):
    cx, cz = int(pnt[0] - gx0), int(pnt[1] - gz0)
    X = cx + ox; Z = cz + oz
    d2 = (gx0 + X + 0.5 - pnt[0]) ** 2 + (gz0 + Z + 0.5 - pnt[1]) ** 2
    sl = (slice(cz - RAD, cz + RAD + 1), slice(cx - RAD, cx + RAD + 1))
    better = d2 < bd[sl]
    bd[sl][better] = d2[better]; bi[sl][better] = i
def on_line(p):
    ix = np.clip((p[:, 0] - gx0).astype(int), 0, gw - 1); iz = np.clip((p[:, 1] - gz0).astype(int), 0, gh - 1)
    i = bi[iz, ix]; ok = bd[iz, ix] < 28 ** 2
    d = p - LC[i]
    return i + (d * LT[i]).sum(1), (d * LN[i]).sum(1), ok
def line_point(s_, lat_):
    s_ = np.mod(s_, NL); i = np.floor(s_).astype(int) % NL; f = (s_ - np.floor(s_))[:, None]
    base = LC[i] * (1 - f) + LC[(i + 1) % NL] * f
    nrm = LN[i] * (1 - f) + LN[(i + 1) % NL] * f
    return base + nrm * lat_[:, None]

# ---------- computed racing line: minimum curvature inside the track ----------
# the feed has no per-car lateral position, so the line a car would take is computed: the smoothest path that stays
# within the asphalt (half width 6 m, minus a car half width and a margin) around the feed line
RL_STEP = 4
cs = LC[::RL_STEP]; ns_ = LN[::RL_STEP]; M_ = len(cs)
BOUND = 6.0 - 1.4
off = np.zeros(M_)
for it in range(60000):
    P_ = cs + ns_ * off[:, None]
    d2 = np.roll(P_, 1, 0) - 2 * P_ + np.roll(P_, -1, 0)
    g = np.roll(d2, 1, 0) - 2 * d2 + np.roll(d2, -1, 0)
    off = np.clip(off - 0.08 * (g * ns_).sum(1), -BOUND, BOUND)
rl_off = np.interp(np.arange(NL), np.arange(M_) * RL_STEP, off, period=NL)
P_ = cs + ns_ * off[:, None]
kappa = np.linalg.norm(np.roll(P_, 1, 0) - 2 * P_ + np.roll(P_, -1, 0), axis=1) / RL_STEP ** 2
print("racing line: offset range", round(off.min(), 2), round(off.max(), 2), "max curvature 1/m", round(float(kappa.max()), 4))
import os
os.makedirs(OUT, exist_ok=True)
# track model for the live engine (built before the race, e.g. from qualifying): line, normals, computed line offset
np.savez_compressed(f"{OUT}/track.npz", LC=LC, LT=LT, LN=LN, rl_off=rl_off, racing_line=racing_line, med_s=med_s, spd_s=spd_s, xf_t=TT, xf_s=S, xf_R=R)
if os.environ.get("TRACK_ONLY"): sys.exit(0)

# ---------- telemetry (car_data): speed, throttle, brake, gear ----------
TEL = np.zeros((NF, len(nums), 5), np.uint8)  # speed/1.5, throttle %, brake %, gear, rpm/100

for j, n in enumerate(nums):
    tr, xyr = raw[n]
    s_, lat_, ok = on_line(xyr)
    tt, tel = tel_for(n)
    v_grid = np.interp(tgrid, tt, tel[:, 0]) / 3.6
    k_ = np.clip(np.searchsorted(tt, tgrid), 0, len(tt) - 1)
    TEL[:, j, 0] = np.clip(np.round(v_grid * 3.6 / 1.5), 0, 255)
    TEL[:, j, 1] = np.clip(tel[k_, 1], 0, 100); TEL[:, j, 2] = np.clip(tel[k_, 2], 0, 100); TEL[:, j, 3] = np.clip(tel[k_, 3], 0, 8)
    TEL[:, j, 4] = np.clip(np.round(np.interp(tgrid, tt, tel[:, 4]) / 100), 0, 255)
    if ok.sum() < 10: continue
    to, so, lo_ = tr[ok], s_[ok], lat_[ok]
    prog = np.unwrap(so * 2 * np.pi / NL) * NL / (2 * np.pi)
    # distance along the lap: the car's own speed telemetry gives the shape (braking, traction), the position feed
    # anchors it; their difference is smoothed over ~5 s, so feed stalls and timestamp jitter don't show
    p_v = np.cumsum(v_grid) / HZ
    corr = prog - np.interp(to, tgrid, p_v)
    corr = np.array([np.median(corr[max(0, q - 8):q + 9]) for q in range(len(corr))])
    cg = smooth_resample(to, np.column_stack([corr, corr]), tgrid, sigma=2.5, K=72)[:, 0]
    pg = np.maximum.accumulate(p_v + cg)
    latg = smooth_resample(to, np.column_stack([lo_, lo_]), tgrid, sigma=0.8, K=28)[:, 0]
    # on the track the car follows the computed line; in the pit lane (well off the line) the feed's own offset rules
    w = np.clip(1 - (np.abs(latg) - 3) / 5, 0, 1)
    for t_start in STARTS:
        s0 = np.interp(t_start - 2 - T0, tgrid - T0, pg)
        win = (tgrid - T0 > t_start - 450) & (tgrid - T0 < t_start + 60)
        w[win] *= np.clip((pg[win] - s0 - 150) / 450, 0, 1) * ((pg[win] - s0) > 0) + ((pg[win] - s0) > 600)
        w[win] = np.clip(w[win], 0, 1)
    lat_final = latg + w * np.interp(np.mod(pg, NL), np.arange(NL), rl_off)
    k = np.clip(np.searchsorted(tr, tgrid), 1, len(tr) - 1)
    near_ok = ok[k] & ok[k - 1]
    kk = np.clip(np.searchsorted(to, tgrid), 1, len(to) - 1)
    gap = np.minimum(np.abs(to[kk] - tgrid), np.abs(to[kk - 1] - tgrid))
    use = near_ok & (gap < 3)
    XY[use, j] = line_point(pg[use], lat_final[use])
print("line-following resample done")

# ---------- pit lane: median path of cars inside their pit intervals ----------
pp = []
for k, a, bb in pits:
    pad = 15 if bb - a < 60 else 0  # normal stops: include the entry and exit roads where they leave the track
    # raw feed samples (the replayed track puts cars on the computed line, which is not what this needs)
    tr, xyr = raw[nums[k]]
    m = (tr - T0 >= a - pad) & (tr - T0 <= bb + pad)
    p = xyr[m]
    if len(p) < 3: continue
    # moving samples only (the red-flag queue sits still for half an hour)
    v = np.concatenate([[0], np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(tr[m]), 1e-3)])
    pp.append(p[(v > 1.5)])
    pp.append(p[::40])
PB = 5.0
pit_s, pit_l = [], []
if pp:  # (no pit stops at all in the data, e.g. some qualifying feeds: no lane from the samples)
    pp = np.concatenate(pp)
    idx = nearest_idx(pp); s, l = station_lateral(pp, idx)
    s = s % LEN
    half = LEN / 2
    sw = np.where(s > half, s - LEN, s)  # stations around the finish line become continuous
    lo_b, hi_b = math.floor(sw.min() / PB), math.ceil(sw.max() / PB)
else: lo_b = hi_b = 0
for i in range(lo_b, hi_b):
    v = l[(sw >= i * PB) & (sw < (i + 1) * PB)]
    if len(v) >= 6: pit_s.append((i + 0.5) * PB); pit_l.append(float(np.median(v)))
pit_s = np.array(pit_s); pit_l = np.array(pit_l)
# trim to where the lane leaves / rejoins the racing line
racing_lat = np.interp(pit_s % LEN, s_bins, med_s, period=LEN)
off = np.abs(pit_l - racing_lat)
print("pit offset profile", [(int(a), round(float(o), 1)) for a, o in zip(pit_s[::4], off[::4])])
inside = np.where(off > 1.0)[0]
synth_pit = len(inside) < 3
if not synth_pit:
    # the lane = the longest run of bins off the racing line (single noisy bins elsewhere ignored, 2-bin dips bridged)
    runs, a_ = [], inside[0]
    for x0, x1 in zip(inside[:-1], inside[1:]):
        if x1 - x0 > 3: runs.append((a_, x0)); a_ = x1
    runs.append((a_, inside[-1]))
    r0, r1 = max(runs, key=lambda r: pit_s[r[1]] - pit_s[r[0]])
    i0 = max(r0 - 1, 0); i1 = min(r1 + 1, len(pit_s) - 1)
    pit_s, pit_l = pit_s[i0:i1 + 1], pit_l[i0:i1 + 1]
else:
    # some feeds hide cars while they are in the pit lane (placeholder point): build the lane from where the cars
    # vanish and reappear, set off to the side with the pit buildings (or the left without map data)
    ents, exts = [], []
    for k, a, bb in pits:
        tr_, xy_ = raw[nums[k]]
        m = np.where((tr_ - T0 >= a - 20) & (tr_ - T0 <= bb + 20))[0]
        if len(m) < 3: continue
        g = np.diff(tr_[m]).argmax()
        if tr_[m[g + 1]] - tr_[m[g]] < 5: continue
        e_ = m[g]; x_ = m[min(g + 2, len(m) - 1)]
        pe = xy_[[e_, x_]]; st_, _ = station_lateral(pe, nearest_idx(pe))
        ents.append(st_[0] % LEN); exts.append(st_[1] % LEN)
    if len(ents) >= 2:
        sA = float(np.median(ents)); Lp = float(np.median([(x - sA) % LEN for x in exts]))
        if not 150 < Lp < 1500: Lp = 400.0
        side = 1.0
        try:
            bs = [np.array(b_[2], float).mean(0) for b_ in json.load(open(DATA)).get("buildings", [])]
            if bs:
                bs = np.array(bs); bi = nearest_idx(bs); bst, blat = station_lateral(bs, bi)
                sel = (((bst - sA) % LEN) < Lp) & (np.abs(blat) > 15) & (np.abs(blat) < 120)
                if sel.sum(): side = float(np.sign(np.sum(np.sign(blat[sel]))) or 1)
        except Exception as e_:
            print("pit side from buildings failed", e_)
        pit_s = sA + np.arange(0, Lp + 0.1, PB)
        ramp = np.clip(np.minimum(pit_s - sA, sA + Lp - pit_s) / 90, 0, 1); ramp = ramp * ramp * (3 - 2 * ramp)
        pit_l = np.interp(pit_s % LEN, s_bins, med_s, period=LEN) + side * 13 * ramp
        print("pit lane rebuilt from feed gaps: entry station", round(sA), "length", round(Lp), "side", side, "stops", len(ents))
    else:
        synth_pit = False
        pit_s = np.array([0.0, PB]); pit_l = np.zeros(2)
k = np.ones(9) / 9
pit_l = np.convolve(np.pad(pit_l, 4, mode="edge"), k, "valid")
pit_line = at_station(pit_s % LEN, pit_l)
if synth_pit:  # cars hidden by the feed in the lane: drive them along the rebuilt lane during their stop
    for k, a, bb in pits:
        fr = np.where((tgrid - T0 >= a) & (tgrid - T0 <= bb))[0]
        if len(fr): XY[fr, k] = at_station((pit_s[0] + (tgrid[fr] - T0 - a) / max(bb - a, 1) * (pit_s[-1] - pit_s[0])) % LEN,
                                           np.interp((tgrid[fr] - T0 - a) / max(bb - a, 1), np.linspace(0, 1, len(pit_l)), pit_l))
print("pit lane", len(pit_line), "points, from station", pit_s[0], "to", pit_s[-1], "lateral range", round(pit_l.min(), 1), round(pit_l.max(), 1))

# ---------- leader lap counter and running order ----------
laps = J("laps")
lap_ev = sorted((ts(l["date_start"]) - T0, l["lap_number"]) for l in laps if l.get("date_start"))
lapline, best = [], 0
for t, n in lap_ev:
    if n > best: best = n; lapline.append([round(t, 1), n])
total_laps = best
order = []
for p in J("position"):
    t = ts(p["date"]) - T0
    if p["driver_number"] not in nums: continue
    order.append([round(max(t, 0), 1), nums.index(p["driver_number"]), p["position"]])
order.sort()

# ---------- retirements: cars that stop moving for good before the flag ----------
chq = next((t for t, st in flags if st == "chequered"), T1 - T0)
# after a crash the feed keeps moving the car along the line for a while; the running order shows the truth:
# a car that drops 5+ places in 20 s and never finishes that lap stopped where the drop began
lap_done = {}
for l in laps:
    if l.get("date_start"): lap_done.setdefault(l["driver_number"], []).append((ts(l["date_start"]) - T0, l["lap_number"], l.get("lap_duration")))
for j, n in enumerate(nums if IS_RACE else []):
    evj = [(t, p) for t, k, p in order if k == j]
    for a in range(1, len(evj)):
        t_a, p_a = evj[a]; p_prev = evj[a - 1][1]
        later = [p for t, p in evj[a:] if t <= t_a + 20]
        if not later or max(later) - p_prev < 5: continue
        # unfinished lap at that moment?
        lp = [x for x in lap_done.get(n, []) if x[0] <= t_a]
        if not lp or lp[-1][2] is not None: continue
        if is_green(t_a) is False and flagAt(t_a) not in ("green", "sc"): continue
        tc = t_a - 2
        fc = int(tc * HZ)
        XY[fc:, j] = XY[fc, j]
        print("crash", n, "frozen at t", round(tc, 1))
        break
spd = np.linalg.norm(np.diff(XY, axis=0), axis=2) * HZ
for j, c in enumerate(cars):
    mv = np.where(spd[:, j] > 5)[0]
    last = mv[-1] / HZ if len(mv) else 0
    if last < chq - 120 and IS_RACE: c["out"] = round(last + 120, 1)
# qualifying: cars parked in the garage (still, off the racing line, for over 20 s) are hidden until they leave
if not IS_RACE:
    P1 = XY[::4]  # 1 Hz
    ix = np.clip((P1[..., 0] - gx0).astype(int), 0, gw - 1); iz = np.clip((P1[..., 1] - gz0).astype(int), 0, gh - 1)
    dline = np.sqrt(bd[iz, ix])  # distance to the line (inf beyond 30 m)
    mv1 = np.vstack([spd, spd[-1:]])[::4] > 1.5
    for j, c in enumerate(cars):
        park = (~mv1[:, j]) & (dline[:, j] > 12)
        hide, k = [], 0
        while k < len(park):
            if park[k]:
                e = k
                while e < len(park) and park[e]: e += 1
                if e - k >= 20 or k == 0: hide.append([float(k + (0 if k == 0 else 3)), float(e if e == len(park) else e - 1)])
                k = e
            else: k += 1
        if hide: c["hide"] = hide

# ---------- write ----------
q = np.round(XY * 10).astype(np.int32).reshape(NF, -1)  # decimetres, frame-major: [frame][car][x,z]
# replay-pos.txt: frame-to-frame differences as int16, gzip, base64 (the page undoes the three steps)
import gzip, base64
dl = np.diff(q, axis=0, prepend=0)
if np.abs(dl).max() >= 32768:
    bad = np.argwhere(np.abs(dl) >= 32768); print("bad jumps", bad[:5], XY.reshape(NF, -1)[bad[0][0] - 1:bad[0][0] + 2, bad[0][1]]); raise SystemExit(1)
open(f"{OUT}/replay-tel.txt", "wb").write(base64.b64encode(gzip.compress(TEL.tobytes(), 9)))
open(f"{OUT}/replay-pos.txt", "wb").write(base64.b64encode(gzip.compress(dl.astype("<i2").tobytes(), 9)))
# ---------- tyres: compound per stint; a new stint starts at the middle of the matching pit stop ----------
lap_t = {(l["driver_number"], l["lap_number"]): ts(l["date_start"]) - T0 for l in J("laps") if l.get("date_start")}
tyres = [[] for _ in nums]
for n in nums:
    j = nums.index(n); stops = sorted((a + b) / 2 for k, a, b in pits if k == j)
    for st in sorted((x for x in J("stints") if x["driver_number"] == n), key=lambda x: x["stint_number"]):
        k = st["stint_number"] - 1
        t = 0 if k == 0 else (stops[k - 1] if k - 1 < len(stops) else lap_t.get((n, st["lap_start"]), 0))
        tyres[j].append([round(t, 1), (st["compound"] or "?")[0], st["tyre_age_at_start"] or 0])
# race control messages, as shown on the broadcast (time, category, flag, text)
race_msgs = sorted([round(ts(r["date"]) - T0, 1), r.get("category") or "", r.get("flag") or "", r["message"]] for r in rc if ts(r["date"]) - T0 >= -1800)
# gaps: per car [t, gap to leader, interval to the car ahead] (numbers in s, or text such as "+1 LAP")
gaps = [[] for _ in nums]
if os.path.exists(f"{OF}/intervals.json"):
    for r in sorted(J("intervals"), key=lambda r: r["date"]):
        if r["driver_number"] not in nums: continue
        g_, i_ = r.get("gap_to_leader"), r.get("interval")
        gaps[nums.index(r["driver_number"])].append([round(ts(r["date"]) - T0, 1), round(g_, 3) if isinstance(g_, (int, float)) else g_, round(i_, 3) if isinstance(i_, (int, float)) else i_])
# fastest lap: each time someone beats the best lap so far [t lap completed, car, lap time s, lap number]
fastest, best_lt = [], 1e9
for l in sorted((l for l in laps if l.get("lap_duration") and l.get("date_start")), key=lambda l: ts(l["date_start"]) + l["lap_duration"]):
    if l["lap_duration"] < best_lt and l["driver_number"] in nums:
        best_lt = l["lap_duration"]
        fastest.append([round(ts(l["date_start"]) + l["lap_duration"] - T0, 1), nums.index(l["driver_number"]), round(l["lap_duration"], 3), l["lap_number"]])
# qualifying: valid lap times (lap deleted by the stewards removed), best per car in each phase, who went out where
qual = {}
if IS_QUALI:
    import re
    deleted = set()
    for r in rc:
        mm = re.match(r"CAR (\d+) .*?TIME (\d+):(\d+\.\d+) DELETED", r["message"])
        if mm: deleted.add((int(mm.group(1)), round(int(mm.group(2)) * 60 + float(mm.group(3)), 3)))
    lt = [[] for _ in nums]; qbest = [[] for _ in nums]; best_ph = {}
    phase_of = lambda t: next((k for k, p in enumerate(PHASES) if p[0] - 5 <= t <= p[1] + 200), -1)
    for l in sorted((l for l in laps if l.get("date_start") and l["driver_number"] in nums), key=lambda l: l["date_start"]):
        j = nums.index(l["driver_number"]); a = ts(l["date_start"]) - T0; d = l.get("lap_duration")
        dl_ = bool(d and (l["driver_number"], round(d, 3)) in deleted)
        lt[j].append([round(a, 2), round(d, 3) if d else None, 1 if dl_ else 0, 1 if l.get("is_pit_out_lap") else 0])
        if d and not dl_ and not l.get("is_pit_out_lap"):
            ph = phase_of(a + d)
            if ph >= 0 and d < best_ph.get((j, ph), 1e9):
                best_ph[(j, ph)] = d; qbest[j].append([round(a + d, 1), ph, round(d, 3)])
    # knocked out in phase k = last phase with a classified time in the session result (missing result: last phase with a lap)
    res = {r["driver_number"]: r for r in J("session_result")} if os.path.exists(f"{OF}/session_result.json") else {}
    qout = []
    # final position decides how far a car got (top 10 reach the last phase, the rest split evenly): a car that
    # reached Q3 but set no time there has no Q3 duration
    n_q2 = 10 + (len(res) - 10) // 2
    for j, n in enumerate(nums):
        dur = (res.get(n) or {}).get("duration"); pos = (res.get(n) or {}).get("position")
        pos = pos if isinstance(pos, int) else None
        if pos and len(res) > 10: k = 2 if pos <= 10 else 1 if pos <= n_q2 else 0
        elif isinstance(dur, list): k = max([i for i, v in enumerate(dur) if v] or [0])
        else: k = max([ph for (jj, ph) in best_ph if jj == j] or [0])
        qout.append(k)
    fastest = [f for f in fastest if (nums[f[1]], f[2]) not in deleted]
    qual = {"phases": PHASES, "lapTimes": lt, "qbest": qbest, "qout": qout}
# start lights: lights out = when the front row pulls away (speed telemetry), around each standing start
lights_out = [lo_ for lo_ in (launch_time(t_flag - 30, t_flag + 300) for t_flag in STARTS) if lo_ is not None]
print("lights out", lights_out, "flags", STARTS)
# known gaps in this session's data, shown next to it in the session menu
notes = []
_reb = sum(1 for v in filled.values() if v > 4 * 600)
if _reb: notes.append(f"Positions rebuilt from speed and lap times for {_reb} car{'s' if _reb > 1 else ''} (the position feed has gaps)")
if len(pit_line) < 20 and pits: notes.append("Pit lane approximate (pit data incomplete)")
if IS_RACE and lights_out and STARTS and lights_out[0] - STARTS[0] > 60: notes.append("Green flag shown before the lights went out")
if IS_RACE and not any(m[3].startswith("CHEQUERED") for m in race_msgs): notes.append("No chequered flag in the data: finish estimated from lap times")
if IS_QUALI and not any(qual.get("qbest", [])): notes.append("No lap times in the data")
if notes: print("notes", notes)
r2 = lambda a: [[round(float(x), 2), round(float(z), 2)] for x, z in a]
out = {
    "session": f"{MEET.get('meeting_name', SES.get('location', ''))} {SES['year']} · {SES['session_name']}", "circuit": SES.get("circuit_short_name", ""), "year": SES["year"],
    "date": SES["date_start"][:10], "hz": HZ, "frames": int(NF), "t0": SES["date_start"], "gmtOffset": SES.get("gmt_offset", "00:00:00"),
    "cars": cars, "laps": total_laps, "lapline": lapline, "order": order, "flags": flags, "pits": pits, "tyres": tyres, "grid": grid, "rc": race_msgs, "gaps": gaps, "fastest": fastest, "lightsOut": lights_out,
    "racingLine": r2(racing_line), "lineOffset": [round(float(v), 2) for v in np.interp(np.array(seg_start, float), np.arange(NL), rl_off)], "telScale": [1.5, 1, 1, 1, 100], "telN": 5, "lineSpeed": [int(round(float(v))) for v in spd_s], "pitLane": r2(pit_line),
    "sources": "Positions, laps, order, pits and flags: OpenF1 (api.openf1.org, CC BY-NC-SA 4.0), session " + str(SESSION) + ".",
    "notes": notes, "kind": "quali" if IS_QUALI else ("sprint" if SES.get("session_name") == "Sprint" else ("race" if IS_RACE else "practice")), **qual,
}
json.dump(out, open(f"{OUT}/replay.json", "w"), separators=(",", ":"))
print("frames x cars", q.shape, "laps", total_laps)
