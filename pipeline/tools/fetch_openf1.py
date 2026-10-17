"""Download everything the replay build needs for one OpenF1 session.

usage: python3 -I fetch_openf1.py <session_key> <out_dir>
Saves sessions, meetings, drivers, laps, position, race_control, pit, stints, starting_grid (from the meeting's
qualifying, or sprint qualifying for a sprint), and per driver loc_<n>.json (location) and car_<n>.json (car_data).
"""
import json, os, sys, time, urllib.request, urllib.error

API = "https://api.openf1.org/v1/"
SK, OUT = int(sys.argv[1]), sys.argv[2]
os.makedirs(OUT, exist_ok=True)

def get(path, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(API + path, timeout=300) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return []
            if k == tries - 1: raise
        except Exception:
            if k == tries - 1: raise
        time.sleep(3 * (k + 1))

def save(name, data):
    json.dump(data, open(f"{OUT}/{name}.json", "w"))
    print(name, len(data) if hasattr(data, "__len__") else "", flush=True)

ses = get(f"sessions?session_key={SK}"); save("sessions", ses)
s = ses[0]
save("meetings", get(f"meetings?meeting_key={s['meeting_key']}"))
for ep in ("drivers", "laps", "position", "race_control", "pit", "stints", "intervals", "session_result"):
    save(ep, get(f"{ep}?session_key={SK}"))
grid = []
# a sprint starts from the sprint qualifying (called "Sprint Shootout" in 2023), everything else from qualifying
want = ("Sprint Qualifying", "Sprint Shootout") if s["session_name"] == "Sprint" else ("Qualifying",)
for q in get(f"sessions?meeting_key={s['meeting_key']}"):
    if q["session_name"] in want and s["session_type"] == "Race":
        grid = get(f"starting_grid?session_key={q['session_key']}")
        if not isinstance(grid, list) or not grid:  # no published grid: the qualifying classification instead
            grid = [{"position": r["position"], "driver_number": r["driver_number"]} for r in get(f"session_result?session_key={q['session_key']}") or [] if r.get("position")]
if not isinstance(grid, list) or not grid:
    g = get(f"starting_grid?session_key={SK}"); grid = g if isinstance(g, list) else []
save("starting_grid", grid)
a, b = s["date_start"][:19], s["date_end"][:19]
# a red flag can run the race past its scheduled end: extend to the last race-control message (+5 min)
from datetime import datetime, timedelta
rc_ = json.load(open(f"{OUT}/race_control.json"))
if rc_:
    last = max(datetime.fromisoformat(r["date"]) for r in rc_) + timedelta(minutes=5)
    b = max(b, last.isoformat()[:19])
for d in json.load(open(f"{OUT}/drivers.json")):
    n = d["driver_number"]
    for ep, pre in (("location", "loc"), ("car_data", "car")):
        if os.path.exists(f"{OUT}/{pre}_{n}.json"): continue
        save(f"{pre}_{n}", get(f"{ep}?session_key={SK}&driver_number={n}&date>{a}&date<{b}"))
