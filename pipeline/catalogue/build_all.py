"""Build every finished weekend in catalogue/plan.json: the race (with its 3D world) first, then the other sessions
on that world. Resumable: anything with a replay.json already is skipped.

usage: python3 -I catalogue/build_all.py [years...]   (from the pipeline folder; output in ../web/races, scratch in work/)
"""
import json, os, subprocess, shutil, sys
from concurrent.futures import ThreadPoolExecutor
PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(PIPE)
WEB, WORK = "../web", "work"
plan = [m for m in json.load(open("catalogue/plan.json")) if m["ended"]]
years = [int(y) for y in sys.argv[1:]] or sorted({m["year"] for m in plan}, reverse=True)
plan = sorted((m for m in plan if m["year"] in years), key=lambda m: (years.index(m["year"]), m["date"]))
os.makedirs(WORK, exist_ok=True)
log = open(f"{WORK}/done.txt", "a", buffering=1)
def run(cmd, logf):
    with open(logf, "w") as fh: return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
def meeting(m):
    race = m["sessions"]["race"]; rwd = f"{WORK}/{race}"; world = f"{WEB}/races/{race}/world.json"
    if not os.path.exists(f"{WEB}/races/{race}/replay.json"):
        # map data is downloaded once per circuit and reused by later seasons
        cache = f"{WORK}/scenery/{m['cid']}"
        if os.path.exists(f"{cache}/dem.pkl") and not os.path.exists(f"{rwd}/scen/dem.pkl"):
            shutil.copytree(cache, f"{rwd}/scen", dirs_exist_ok=True)
        rc = run(["sh", "tools/add_race.sh", str(race), m["cid"], rwd, f"{WEB}/races/{race}"], f"{WORK}/{race}.log")
        log.write(f"{m['year']} {m['loc']} race {race} {rc}\n")
        if rc != 0: return
        if m["cid"] != "-" and not os.path.exists(f"{cache}/dem.pkl"):
            shutil.copytree(f"{rwd}/scen", cache, dirs_exist_ok=True, ignore=shutil.ignore_patterns("track_local.json"))
        subprocess.call(f"gzip -f -1 {rwd}/of/loc_*.json {rwd}/of/car_*.json", shell=True)
    for typ, key in m["sessions"].items():
        if typ == "race" or os.path.exists(f"{WEB}/races/{key}/replay.json"): continue
        rc = run(["sh", "tools/add_session.sh", str(key), rwd, world, f"{WORK}/{key}", f"{WEB}/races/{key}"], f"{WORK}/{key}.log")
        log.write(f"{m['year']} {m['loc']} {typ} {key} {rc}\n")
with ThreadPoolExecutor(3) as ex: list(ex.map(meeting, plan))
