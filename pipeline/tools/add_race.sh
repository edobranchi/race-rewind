#!/bin/sh
# Build one race end to end: OpenF1 data, track from the feed, 3D world from map data, replay.
# usage: add_race.sh <session_key> <f1-circuits id or "-"> <work_dir> <out_race_dir>
# Steps already done in <work_dir> (OpenF1 download, map scenery) are reused, so a failed build can be re-run.
set -e
KEY=$1; CID=$2; WD=$3; OUT=$4
T=$(dirname "$0"); B=$(dirname "$T")
mkdir -p "$WD"
[ -f "$WD/of/loc_done" ] || { python3 -I "$T/fetch_openf1.py" "$KEY" "$WD/of" > "$WD/fetch.log" 2>&1; touch "$WD/of/loc_done"; }
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$WD/of/sessions.json'))[0].get('is_cancelled') else 1)"; then
  echo "cancelled $KEY"; exit 3
fi
python3 -I "$T/world_from_feed.py" "$WD/of" "$WD/feed_world.json"
cp "$WD/of/xf.json" "$WD/of/xf_feed.json"
WORLD="$WD/feed_world.json"
if [ "$CID" != "-" ]; then
  [ -f "$WD/scen/dem.pkl" ] || python3 -P "$T/fetch_scenery.py" "$CID" "$WD/scen" > "$WD/scen.log" 2>&1
  rm -f "$WD/scen/track_local.json"
  python3 -P "$T/build_world.py" "$WD/scen" "$WD/map_world.json" > "$WD/world1.log" 2>&1
  python3 -P "$T/fit_feed.py" "$WD/of" "$WD/map_world.json" > "$WD/fit.log" 2>&1 || echo "median err m 99" >> "$WD/fit.log"
  ERR=$(sed -n 's/^median err m \([0-9.]*\).*/\1/p' "$WD/fit.log" | tail -1)
  if python3 -c "import sys; sys.exit(0 if float('$ERR') < 12 else 1)"; then
    python3 -I "$T/place_feed_track.py" "$WD/feed_world.json" "$WD/of/xf_feed.json" "$WD/of/xf.json" "$WD/scen/track_local.json"
    python3 -P "$T/build_world.py" "$WD/scen" "$WD/map_world.json" > "$WD/world2.log" 2>&1
    WORLD="$WD/map_world.json"
  else
    echo "map fit too loose ($ERR m): flat world"; cp "$WD/of/xf_feed.json" "$WD/of/xf.json"
  fi
fi
mkdir -p "$OUT"
python3 -I "$B/build_replay.py" "$WD/of" "$WORLD" "$OUT" > "$WD/replay.log" 2>&1
cp "$WORLD" "$OUT/world.json"
rm -f "$OUT/track.npz"
echo "ok $KEY $(sed -n 's/^median err m \([0-9.]*\).*/map fit \1 m/p' "$WD/fit.log" 2>/dev/null | tail -1)"
