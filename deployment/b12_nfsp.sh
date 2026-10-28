#!/bin/bash
set -u
REPO=/root/hbot/HoloMotion
cd "$REPO" || exit 1
DATA=$REPO/holoboxing/data/processed/b12-boxing/prepared
rm -rf results/b12-nfsp
/root/hbvenv/bin/python -m holoboxing.training.train_rl \
  --assets "$REPO/holoboxing/assets" --lower "$REPO/models/holomotion-v2" --amp-motions "$DATA" \
  --policy-checkpoint results/b12-warm/model_32.pt \
  --config holoboxing/training/configs/b8.json \
  --training-mode selfplay --opponent-pool-capacity 8 --opponent-refresh-updates 32 \
  --nfsp --nfsp-capacity 200000 --nfsp-batch 4096 --nfsp-lr 3e-4 --nfsp-interval 1 \
  --output results/b12-nfsp --num-envs 256 --updates 2000 --seed 43 --device cuda:0 \
  --checkpoint-interval-seconds 1800 --save-reserve-seconds 600 --max-runtime-seconds 21600
