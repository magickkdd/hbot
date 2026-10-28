#!/bin/bash
# 书生开发机 B7 SL 分块训练（19 epochs，每块 1800s，断点续传）
# 用法: nohup bash /data/run_sl.sh > /data/logs/sl_stdout.log 2>&1 &
set -u
BASE=/data/hbot
REPO=$BASE/HoloMotion
cd "$REPO" || exit 1
source /data/env.sh
mkdir -p /data/logs results/b7-sl
PY=$REPO/.venv-holoboxing/bin/python

if pgrep -f "[p]ython -m holoboxing.training.train_sl" >/dev/null; then echo "SL already running"; exit 0; fi
if ls results/b7-sl*/epoch_000019.pt >/dev/null 2>&1; then
  cp "$(ls -t results/b7-sl*/epoch_000019.pt | head -1)" results/b7-sl/epoch_000019.pt
  echo "SL already done"; exit 0
fi

while true; do
  CKPT=$(ls -t results/b7-sl*/last.pt results/b7-sl*/epoch_*.pt 2>/dev/null | head -1 || true)
  OUT="results/b7-sl-c$(date +%Y%m%d_%H%M%S)"
  if [ -n "$CKPT" ]; then START="--resume $CKPT"; else START=""; fi
  LOG="/data/logs/b7sl_$(date +%Y%m%d_%H%M%S).log"
  nice -n 10 "$PY" -m holoboxing.training.train_sl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --data "$HOLOBOXING_DATA" --prior-checkpoint results/b1-prior/best.pt \
    --config holoboxing/training/configs/sample.json \
    --output "$OUT" --epochs 19 --batch-size 512 --seed 43 \
    --device cuda:0 --max-runtime-seconds 1800 $START 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  if ls results/b7-sl*/epoch_000019.pt >/dev/null 2>&1; then
    cp "$(ls -t results/b7-sl*/epoch_000019.pt | head -1)" results/b7-sl/epoch_000019.pt
    echo "SL done"; break
  fi
  echo "SL chunk rc=$RC; sleep 10"; sleep 10
done
