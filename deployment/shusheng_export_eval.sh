#!/bin/bash
# NFSP 完成后：双导出 + validate + native 4+4 局
set -u
REPO=/root/hbot/HoloMotion; cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=/root/hbot/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/root/hbot/HoloMotion/models/holomotion-v2
export ROBOSTRIKER_CHECKPOINT=/root/hbot/model_3000.pt
PY=/root/hbvenv/bin/python
LATEST=$(ls -t results/b8-nfsp*/model_*.pt 2>/dev/null | head -1)
"$PY" -m holoboxing.training.export --checkpoint "$LATEST" --output results/boxing-b8-br.onnx
"$PY" -m holoboxing.training.export --checkpoint "$LATEST" --policy average --output results/boxing-b8-avg.onnx
export ROBOSTRIKER_SHA256=$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)
for NAME in boxing-b8-br boxing-b8-avg; do
  "$PY" -m holoboxing validate results/$NAME.onnx --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --output results/${NAME}-validation.json --benchmark
  for SIDE in 0 1; do
    "$PY" -m holoboxing.training.evaluate_native --candidate results/$NAME.onnx \
      --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" --opponent-sha256 "$ROBOSTRIKER_SHA256" \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --candidate-side $SIDE --output results/native-$NAME-side$SIDE-a --device cuda:0
    python3 deployment/fall_attrib.py results/native-$NAME-side$SIDE-a/match.json $SIDE || true
  done
done
echo EVAL_DONE
