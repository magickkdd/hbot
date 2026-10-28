#!/bin/bash
set -u
GAME=${1:?}
MATCH=${2:?}
LABEL=${3:?}
OUT=${4:?}
PY=/mnt/workspace/miniforge3/envs/hbot-eval/bin/python
FRAMES=/tmp/b5a_frames_$GAME
LOG=/mnt/workspace/logs/render_$GAME.log
cd /mnt/workspace/HoloMotion || exit 1
rm -rf "$FRAMES"
mkdir -p "$FRAMES"
echo "== render $GAME $(date) ==" >> "$LOG"
nice -n 10 "$PY" render_replay.py "$MATCH" "$FRAMES" "$LABEL" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "EGL_FAILED rc=$rc; retry osmesa" >> "$LOG"
  rm -rf "$FRAMES"
  mkdir -p "$FRAMES"
  MUJOCO_GL=osmesa nice -n 10 "$PY" render_replay.py "$MATCH" "$FRAMES" "$LABEL" >> "$LOG" 2>&1
  rc=$?
fi
[ "$rc" -eq 0 ] || { echo "RENDER_FAIL $GAME rc=$rc" >> "$LOG"; exit "$rc"; }
nice -n 10 ffmpeg -y -loglevel error -framerate 50 -i "$FRAMES/f%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT" >> "$LOG" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { echo "FFMPEG_FAIL $GAME rc=$rc" >> "$LOG"; exit "$rc"; }
rm -rf "$FRAMES"
ls -lh "$OUT" >> "$LOG"
echo "RENDER_DONE $GAME $(date)" >> "$LOG"
