#!/bin/bash
# 单 checkpoint 完整评测：双导出 + validate + native 4 局（BR/AVG × side0/1）
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
OUT=/data/eval-7657
CK=${1:-$(ls -t results/b8-nfsp-r*/model_*.pt | head -1)}
[ -f "$CK" ] || { echo "CKPT_MISSING"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
export ROBOSTRIKER_SHA256=$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)
echo "== ckpt=$CK  $(date)"
"$PY" -m holoboxing.training.export --checkpoint "$CK" --output "$OUT/br.onnx" || { echo EXPORT_BR_FAIL; exit 1; }
"$PY" -m holoboxing.training.export --checkpoint "$CK" --policy average --output "$OUT/avg.onnx" || { echo EXPORT_AVG_FAIL; exit 1; }
for NAME in br avg; do
  "$PY" -m holoboxing validate "$OUT/$NAME.onnx" --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --output "$OUT/$NAME-validation.json" --benchmark
done
for NAME in br avg; do
  for SIDE in 0 1; do
    echo "== match $NAME side$SIDE $(date)"
    "$PY" -m holoboxing.training.evaluate_native --candidate "$OUT/$NAME.onnx" \
      --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" --opponent-sha256 "$ROBOSTRIKER_SHA256" \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --candidate-side $SIDE --output "$OUT/native-$NAME-side$SIDE" --device cuda:0
    "$PY" - "$OUT/native-$NAME-side$SIDE/match.json" "$NAME" "$SIDE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); s=d["summary"]
print("RESULT", sys.argv[2], "side"+sys.argv[3], json.dumps(s, ensure_ascii=False)[:400])
PY
  done
done
echo "EVAL_DONE $(date)"
