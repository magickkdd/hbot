#!/bin/bash
# B9：拳击示范 BC 之后 -> 站立热身 32 次 -> 交互优化（短程验证）
set -u
REPO=/root/hbot/HoloMotion
cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
PY=/root/hbvenv/bin/python
AMP=$REPO/holoboxing/data/processed/b9-boxing/prepared

if [ ! -f results/b9-warmup/model_32.pt ]; then
  rm -rf results/b9-warmup
  nice -n 10 "$PY" -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$AMP" \
    --supervised-checkpoint results/b9-sl/epoch_000008.pt \
    --config holoboxing/training/configs/b8.json \
    --training-mode standing --output results/b9-warmup --num-envs 128 \
    --updates 32 --seed 43 --device cuda:0 --max-runtime-seconds 1800 \
    2>&1 | tee -a /data/logs/b9_warmup.log
fi
echo "WARMUP_DONE $(ls results/b9-warmup/model_*.pt 2>/dev/null | tail -1)"

nice -n 10 "$PY" -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$AMP" \
  --policy-checkpoint results/b9-warmup/model_32.pt \
  --config holoboxing/training/configs/b8.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 --nfsp \
  --nfsp-capacity 200000 --nfsp-batch 4096 --nfsp-lr 3e-4 --nfsp-interval 1 \
  --output results/b9-nfsp --num-envs 256 --updates 2000 --seed 43 --device cuda:0 \
  --save-reserve-seconds 300 --checkpoint-interval-seconds 600 \
  --max-runtime-seconds 14400 2>&1 | tee -a /data/logs/b9_nfsp.log
echo "B9_TRAIN_DONE"
