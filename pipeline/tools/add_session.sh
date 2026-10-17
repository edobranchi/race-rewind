#!/bin/sh
# Build one more session (qualifying, sprint...) of a weekend whose race is already built, reusing that race's 3D world.
# usage: add_session.sh <session_key> <race work dir (holds of/xf.json)> <race world.json> <work_dir> <out_dir>
# The download is reused when the build is re-run; the bulky per-car files are gzipped once the replay is built.
set -e
KEY=$1; RWD=$2; WORLD=$3; WD=$4; OUT=$5
T=$(dirname "$0"); B=$(dirname "$T")
mkdir -p "$WD"
[ -f "$WD/of/loc_done" ] || { python3 -I "$T/fetch_openf1.py" "$KEY" "$WD/of" > "$WD/fetch.log" 2>&1; touch "$WD/of/loc_done"; }
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$WD/of/sessions.json'))[0].get('is_cancelled') else 1)"; then
  echo "cancelled $KEY"; exit 3
fi
cp "$RWD/of/xf.json" "$WD/of/xf.json"
mkdir -p "$OUT"
python3 -I "$B/build_replay.py" "$WD/of" "$WORLD" "$OUT" > "$WD/replay.log" 2>&1
rm -f "$OUT/track.npz"
# keep this session's own track line and pit lane clear on the shared world too
python3 -P "$T/clear_world.py" "$WORLD" "$OUT/replay.json" >> "$WD/replay.log" 2>&1
gzip -f -1 "$WD"/of/loc_*.json "$WD"/of/car_*.json 2>/dev/null || true
echo "ok $KEY"
