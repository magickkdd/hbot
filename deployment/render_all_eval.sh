#!/bin/bash
# 把 /data/eval-7657/native-*/match.json 渲染成 mp4（隔帧采样 + 双路并行）
set -u
OUT=/data/eval-7657
cd /root/hbot/HoloMotion || exit 1
export MUJOCO_GL=osmesa
export BOXING_ARENA_XML=/root/hbot/HoloMotion/thirdparties/HoloMotion_assets/holoboxing/arena/boxing.xml
export RENDER_STRIDE=${RENDER_STRIDE:-2}
FPS=$((50 / RENDER_STRIDE))
PY=/root/hbvenv/bin/python
mkdir -p "$OUT/videos"

one() {
  TAG=$1; NAME=$2; SIDE=$3
  M=$OUT/native-$TAG/match.json
  [ -f "$M" ] || { echo "NO_MATCH $TAG"; return; }
  FR=/tmp/fr_$TAG; rm -rf "$FR"; mkdir -p "$FR"
  if [ "$SIDE" = "0" ]; then WHO="red"; else WHO="blue"; fi
  echo "== render $TAG $(date)"
  "$PY" /data/render_replay.py "$M" "$FR" "B8@7657 $NAME($WHO) vs RoboStriker" \
    > /data/logs/render_$TAG.log 2>&1 || { echo "RENDER_FAIL $TAG"; return; }
  ffmpeg -y -loglevel error -framerate "$FPS" -i "$FR/f%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 \
    "$OUT/videos/$TAG.mp4" || { echo "FFMPEG_FAIL $TAG"; return; }
  rm -rf "$FR"
  echo "VIDEO_OK $TAG $(du -h "$OUT/videos/$TAG.mp4" | cut -f1)"
}

for NAME in br avg; do
  for SIDE in 0 1; do
    one "$NAME-side$SIDE" "$NAME" "$SIDE" &
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
  done
done
wait
echo "RENDER_ALL_DONE $(date)"
