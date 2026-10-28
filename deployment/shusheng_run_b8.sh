#!/bin/bash
# 书生开发机 B8 warmup+NFSP 分块训练（需要 b8 代码就绪；b8.json 不存在时退出）
# 用法: nohup bash /data/run_b8.sh > /data/logs/b8_stdout.log 2>&1 &
set -u
BASE=/data/hbot
REPO=$BASE/HoloMotion
cd "$REPO" || exit 1
source /data/env.sh
mkdir -p /data/logs results/b8-warmup results/b8-nfsp
PY=$REPO/.venv-holoboxing/bin/python
SL=results/b7-sl/epoch_000019.pt
[ -f "$SL" ] || { echo "B7 SL not done"; exit 2; }
[ -f holoboxing/training/configs/b8.json ] || { echo "B8 code missing (b8.json)"; exit 2; }
if pgrep -f "[p]ython -m holoboxing.training.train_rl" >/dev/null; then echo "RL already running"; exit 0; fi

while [ ! -f results/b8-warmup/model_200.pt ]; do
  if ls results/b8-warmup*/model_*.pt >/dev/null 2>&1; then
    CKPT=$(ls -t results/b8-warmup*/model_*.pt | head -1)
    START="--resume $CKPT"; OUT="results/b8-warmup-r$(date +%s)"
  else
    START="--supervised-checkpoint $SL"; OUT="results/b8-warmup"
  fi
  nice -n 10 "$PY" -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$HOLOBOXING_DATA" $START \
    --config holoboxing/training/configs/b8.json \
    --training-mode standing --output "$OUT" --num-envs 128 \
    --updates 200 --seed 43 --device cuda:0 --max-runtime-seconds 1800 \
    2>&1 | tee -a /data/logs/b8_warmup.log
  echo "warmup chunk rc=${PIPESTATUS[0]}"; sleep 5
done

while true; do
  LATEST=$(ls -t results/b8-nfsp*/model_*.pt 2>/dev/null | head -1 || true)
  if [ -n "$LATEST" ]; then
    IT=$(basename "$LATEST" | sed -E 's/model_([0-9]+)\.pt/\1/')
    [ "$IT" -ge 2000 ] && break
    START="--resume $LATEST"; OUT="results/b8-nfsp-r$(date +%s)"
  else
    START="--policy-checkpoint results/b8-warmup/model_200.pt"; OUT="results/b8-nfsp"
  fi
  nice -n 10 "$PY" -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$HOLOBOXING_DATA" $START \
    --config holoboxing/training/configs/b8.json \
    --training-mode selfplay --opponent-pool-capacity 8 \
    --opponent-refresh-updates 32 --output "$OUT" --num-envs 128 \
    --updates 200 --seed 43 --device cuda:0 --max-runtime-seconds 1800 \
    2>&1 | tee -a /data/logs/b8_nfsp.log
  echo "nfsp chunk rc=${PIPESTATUS[0]}"; sleep 5
done
echo B8_DONE
