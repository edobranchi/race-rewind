"""Write web/races/index.json (the page's session menu) from catalogue/plan.json and the sessions built so far.

usage: python3 -I catalogue/make_index.py [years...]
"""
import json, os, sys, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, "..", "..", "web")
# names as people say them, where OpenF1's short circuit name differs
NICE = {"Monte Carlo": "Monaco", "Catalunya": "Barcelona", "Spa-Francorchamps": "Spa", "Hungaroring": "Budapest", "Yas Marina Circuit": "Abu Dhabi",
        "Madring": "Madrid", "Kuala Lumpur": "Sepang", "Interlagos": "São Paulo", "Montreal": "Montréal", "Sakhir": "Bahrain"}
years = [int(y) for y in sys.argv[1:]]
out, latest = [], None
for m in json.load(open(f"{HERE}/plan.json")):
    if years and m["year"] not in years: continue
    world = f"races/{m['sessions']['race']}/world.json"
    if not os.path.exists(f"{WEB}/{world}"): continue
    d = datetime.date.fromisoformat(m["date"]); latest = m if m["sessions"]["race"] and os.path.exists(f"{WEB}/races/{m['sessions']['race']}/replay.json") else latest
    for typ, key in m["sessions"].items():
        if os.path.exists(f"{WEB}/races/{key}/replay.json"):
            out.append({"key": key, "year": m["year"], "meeting": m["meeting"], "track": NICE.get(m["loc"], m["loc"]), "type": typ,
                        "dateLabel": f"{d.day} {d.strftime('%b')}", "world": world})
            # known data problems (from build_replay), flagged in the menu
            n = json.load(open(f"{WEB}/races/{key}/replay.json")).get("notes")
            if n: out[-1]["notes"] = n
# the page opens the most recent race unless the link names a session
last = next((e for e in out if latest and e["key"] == latest["sessions"]["race"]), None)  # plan.json is in date order
if last: last["default"] = True
json.dump(out, open(f"{WEB}/races/index.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(len(out), "sessions in", len({e["meeting"] for e in out}), "weekends")
