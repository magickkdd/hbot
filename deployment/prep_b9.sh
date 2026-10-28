#!/bin/bash
# 拳击示范数据：source npz -> prepared(obs 1097 + target 225) -> sources.json/COMPLETE.json
set -u
REPO=/root/hbot/HoloMotion
SRC=/data/boxing-mocap/g1
OUT=$REPO/holoboxing/data/processed/b9-boxing
PY=/root/hbvenv/bin/python
cd "$REPO" || exit 1
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$REPO/holoboxing/evaluation/runtime" && mv compiled compiled.bak
cd "$REPO" || exit 1
"$PY" -m holoboxing.data.records \
  --assets "$REPO/holoboxing/assets" \
  --motion-directory "$SRC" \
  --output "$OUT/prepared" --workers 1 > /data/logs/b9_prepare.log 2>&1
rc=$?
cd "$REPO/holoboxing/evaluation/runtime" && mv compiled.bak compiled
echo "prepare_rc=$rc"
[ "$rc" -eq 0 ] || { tail -5 /data/logs/b9_prepare.log; exit "$rc"; }

"$PY" /data/build_sources.py "$OUT" cmu-boxing
