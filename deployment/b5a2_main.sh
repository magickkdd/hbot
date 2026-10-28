#!/bin/bash
set -u
cd /mnt/workspace/HoloMotion || exit 1
source env.sh
CKPT=results/b5a-selfplay/model_2560.pt
[ -f "$CKPT" ] || { echo NO_2560_CKPT; exit 1; }
[ -e results/b5a2-selfplay ] && { echo OUTPUT_EXISTS; exit 2; }
echo "== B5a2 4096 start $(date) =="
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --resume "$CKPT" --allow-resume-code-change \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 \
  --output results/b5a2-selfplay \
  --num-envs 128 --updates 1536 --seed 42 --device cuda:0 \
  --max-runtime-seconds 7200
echo "B5A2_RC=$? $(date)"
