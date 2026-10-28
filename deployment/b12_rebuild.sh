#!/bin/bash
set -u
REPO=/root/hbot/HoloMotion
cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
PY=/root/hbvenv/bin/python
DATA=$REPO/holoboxing/data/processed/b12-boxing/prepared

echo "== 1 数据 =="
bash /data/prep_b12.sh || { echo DATA_FAIL; exit 1; }
tail -1 /data/logs/b12_prep_run.log

echo "== 2 先验 =="
rm -rf /data/b12-prior-data results/b12-prior
"$PY" -m holoboxing.training.prepare_prior --source /data/lafan-pair --assets holoboxing/assets --output /data/b12-prior-data > /data/logs/b12_prep_prior.log 2>&1 || { echo PRIOR_DATA_FAIL; exit 1; }
"$PY" -m holoboxing.training.train_prior --data /data/b12-prior-data --dataset all --output results/b12-prior --epochs 200 --batch-size 256 --device cuda:0 > /data/logs/b12_prior.log 2>&1 || { echo PRIOR_FAIL; exit 1; }
grep -o '"position_m": [0-9.]*' /data/logs/b12_prior.log | tail -1

echo "== 3 模仿初始化（加大到 60 轮）=="
rm -rf results/b12-sl
"$PY" -m holoboxing.training.train_sl --assets holoboxing/assets --lower models/holomotion-v2 \
  --data "$DATA" --dataset lafan-fight --prior-checkpoint results/b12-prior/last.pt \
  --output results/b12-sl --epochs 60 --batch-size 256 --device cuda:0 > /data/logs/b12_sl.log 2>&1 || { echo SL_FAIL; exit 1; }
grep -o '"position_m": [0-9.]*' /data/logs/b12_sl.log | tail -1

echo "== 4 站立热身 32 =="
rm -rf results/b12-warm
nice -n 10 "$PY" -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" --amp-motions "$DATA" \
  --supervised-checkpoint results/b12-sl/epoch_000060.pt \
  --config holoboxing/training/configs/b8.json \
  --training-mode standing --output results/b12-warm --num-envs 128 \
  --updates 32 --seed 43 --device cuda:0 --max-runtime-seconds 1800 > /data/logs/b12_warm.log 2>&1 || { echo WARM_FAIL; exit 1; }
ls results/b12-warm/*.pt

echo "== 5 归档到 /data =="
tar czf /data/b12-artifacts.tgz -C "$REPO" results/b12-prior results/b12-sl results/b12-warm holoboxing/data/processed/b12-boxing 2>/dev/null
ls -lh /data/b12-artifacts.tgz | awk '{print $5}'
echo B12_REBUILD_DONE
