"""List every weekend with a race in OpenF1 and its sessions (race, sprint, qualifying, sprint qualifying).

usage: python3 -I catalogue/plan.py [years...]   (default 2023 to this year)
Writes catalogue/plan.json; a weekend counts as "ended" once its race has finished.
"""
import json, os, sys, urllib.request, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.openf1.org/v1/"
# OpenF1 circuit_short_name -> bacinger/f1-circuits id (outline used to fetch the map data around the track)
CID = {"Sakhir": "bh-2002", "Jeddah": "sa-2021", "Melbourne": "au-1953", "Baku": "az-2016", "Miami": "us-2022", "Imola": "it-1953",
       "Monte Carlo": "mc-1929", "Catalunya": "es-1991", "Montreal": "ca-1978", "Spielberg": "at-1969", "Silverstone": "gb-1948",
       "Hungaroring": "hu-1986", "Spa-Francorchamps": "be-1925", "Zandvoort": "nl-1948", "Monza": "it-1922", "Singapore": "sg-2008",
       "Suzuka": "jp-1962", "Lusail": "qa-2004", "Austin": "us-2012", "Mexico City": "mx-1962", "Interlagos": "br-1940",
       "Las Vegas": "us-2023", "Yas Marina Circuit": "ae-2009", "Shanghai": "cn-2004", "Madring": "es-2026", "Kuala Lumpur": "my-1999"}
TYPES = {"Race": "race", "Sprint": "sprint", "Qualifying": "quali", "Sprint Qualifying": "sq", "Sprint Shootout": "sq"}
get = lambda p: json.load(urllib.request.urlopen(API + p, timeout=120))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
years = [int(y) for y in sys.argv[1:]] or list(range(2023, datetime.date.today().year + 1))
out = []
for y in years:
    M = {m["meeting_key"]: m for m in get(f"meetings?year={y}")}
    S = get(f"sessions?year={y}")
    for mk, m in M.items():
        ss = {TYPES[s["session_name"]]: s for s in S if s["meeting_key"] == mk and s["session_name"] in TYPES}
        if "race" not in ss or ss["race"].get("is_cancelled"): continue
        out.append({"year": y, "meeting": mk, "name": m["meeting_name"], "loc": m["circuit_short_name"], "cid": CID.get(m["circuit_short_name"], "-"),
                    "date": ss["race"]["date_start"][:10], "sessions": {k: s["session_key"] for k, s in ss.items() if not s.get("is_cancelled")},
                    "ended": ss["race"]["date_end"] < now})
out.sort(key=lambda r: r["date"])
json.dump(out, open(f"{HERE}/plan.json", "w"), indent=0)
print(len(out), "weekends,", sum(len(r["sessions"]) for r in out if r["ended"]), "sessions finished")
