#!/bin/bash
# 顺序评测多个 checkpoint（禁止与训练并发）：每个 ckpt 导出 BR/AVG 各打 1 局 side0。
# 用法: eval_sweep.sh <outdir> <ckpt1> [ckpt2 ...]
set -u
# 硬门禁：训练在跑时绝不允许评测（mujoco-warp 编译曾把容器压挂）
if pgrep -f "training[.]train_rl" >/dev/null 2>&1; then
  echo "TRAINING_RUNNING_ABORT"; exit 1
fi
REPO=/root/hbot/HoloMotion; cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
export ROBOSTRIKER_CHECKPOINT=/root/hbot/model_3000.pt
export MUJOCO_GL=osmesa
PY=/root/hbvenv/bin/python
OUT=${1:?outdir}; shift
rm -rf "$OUT"; mkdir -p "$OUT"
export ROBOSTRIKER_SHA256=$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)
for CK in "$@"; do
  TAG=$(basename "$CK" .pt)
  echo "== $TAG $(date)"
  "$PY" -m holoboxing.training.export --checkpoint "$CK" --output "$OUT/$TAG-br.onnx" || { echo "EXPORT_FAIL $TAG"; continue; }
  "$PY" -m holoboxing.training.export --checkpoint "$CK" --policy average --output "$OUT/$TAG-avg.onnx" || { echo "EXPORT_FAIL $TAG avg"; continue; }
  for NAME in br avg; do
    "$PY" -m holoboxing.training.evaluate_native --candidate "$OUT/$TAG-$NAME.onnx" \
      --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" --opponent-sha256 "$ROBOSTRIKER_SHA256" \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --candidate-side 0 --output "$OUT/$TAG-$NAME-side0" --device cuda:0 || { echo "MATCH_FAIL $TAG $NAME"; continue; }
    "$PY" - "$OUT/$TAG-$NAME-side0/match.json" "$TAG" "$NAME" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
# side0 = 红方 = 我方（candidate 固定打 side0）
print("RESULT", sys.argv[2], sys.argv[3],
      "our_hits", s["hit_count"][0], "their_hits", s["hit_count"][1],
      "score", s["final_score"], s["end_reason"], "winner", s["winner"])
PY
  done
done
echo SWEEP_DONE
