"""Put the feed-derived track (world_from_feed.py) on the map frame using the ICP transform from fit_feed.py.
usage: python3 place_feed_track.py <feed_world.json> <xf_feed.json> <xf_map.json> <out track_local.json>"""
import json, sys
import numpy as np
W, XA, XB, OUT = sys.argv[1:5]
a, b = json.load(open(XA)), json.load(open(XB))
loc = np.array(json.load(open(W))["track"], float)
raw = (np.linalg.inv(np.array(a["R"])) @ (loc - np.array(a["t"])).T).T / a["s"]   # back to OpenF1 x/y
geo = (b["s"] * (np.array(b["R"]) @ raw.T)).T + np.array(b["t"])
json.dump([[round(float(x), 2), round(float(z), 2)] for x, z in geo], open(OUT, "w"))
print("points", len(geo))
