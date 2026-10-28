#!/bin/bash
set -u
cd /mnt/workspace/HoloMotion || exit 1
source env.sh
CONDA=/mnt/workspace/miniforge3/bin/conda
export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
[ -f results/boxing-b5a.onnx ] || { echo NO_B5A_ONNX; exit 1; }
for SIDE in 0 1; do
  for R in a b; do
    echo "== native-b5a side$SIDE-$R start $(date +%H:%M:%S) =="
    "$CONDA" run -n hbot-eval python -m holoboxing.training.evaluate_native \
      --candidate results/boxing-b5a.onnx \
      --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
      --opponent-sha256 "$ROBOSTRIKER_SHA256" \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --candidate-side "$SIDE" --output "results/native-b5a-side$SIDE-$R" --device cuda:0
    echo "side${SIDE}_${R}_RC=$?"
  done
done
for SIDE in 0 1; do
  for R in a b; do
    D="results/native-b5a-side$SIDE-$R"
    echo "===== $D (us=$SIDE) ====="
    python3 deployment/fall_attrib.py "$D/match.json" "$SIDE"
  done
done
echo B5A_NATIVE_DONE
