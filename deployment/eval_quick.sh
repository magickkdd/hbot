#!/bin/bash
# 快速评测：导出 BR/AVG，各打 1 局（side0），只看命中
set -u
REPO=/root/hbot/HoloMotion; cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
export ROBOSTRIKER_CHECKPOINT=/root/hbot/model_3000.pt
export MUJOCO_GL=osmesa
PY=/root/hbvenv/bin/python
CK=${1:?ckpt}
OUT=${2:-/data/eval-quick}
rm -rf "$OUT"; mkdir -p "$OUT"
export ROBOSTRIKER_SHA256=$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)
echo "== ckpt=$CK $(date)"
"$PY" -m holoboxing.training.export --checkpoint "$CK" --output "$OUT/br.onnx" || exit 1
"$PY" -m holoboxing.training.export --checkpoint "$CK" --policy average --output "$OUT/avg.onnx" || exit 1
for NAME in br avg; do
  "$PY" -m holoboxing.training.evaluate_native --candidate "$OUT/$NAME.onnx" \
    --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" --opponent-sha256 "$ROBOSTRIKER_SHA256" \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --candidate-side 0 --output "$OUT/native-$NAME-side0" --device cuda:0 || exit 1
  "$PY" - "$OUT/native-$NAME-side0/match.json" "$NAME" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("RESULT", sys.argv[2], json.dumps(d["summary"], ensure_ascii=False)[:300])
PY
done
echo EVAL_QUICK_DONE
