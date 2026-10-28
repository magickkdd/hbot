#!/bin/bash
set -u
cd /mnt/workspace/HoloMotion || exit 1
source env.sh
CONDA=/mnt/workspace/miniforge3/bin/conda
D=results/b5a-selfplay
echo "== completion check =="
.venv-holoboxing/bin/python - <<'PY'
import json, os
d = 'results/b5a-selfplay'
p = os.path.join(d, 'complete.json')
if not os.path.exists(p):
    print('NO_COMPLETE')
    raise SystemExit(1)
c = json.load(open(p))
tr = c.get('training_result', {}) or {}
prog = tr.get('progress', {}) or {}
budget = tr.get('budget', {}) or {}
print('updates_completed=%s progress_completed=%s stop_reason=%s' % (
    c.get('updates_completed'), prog.get('completed_updates'),
    budget.get('stop_reason')))
if c.get('updates_completed') != 2560 or budget.get('stop_reason') != 'requested_progress_complete':
    print('NOT_READY')
    raise SystemExit(2)
PY
rc=$?
[ "$rc" -eq 0 ] || { echo "COMPLETION_CHECK_RC=$rc"; exit "$rc"; }
CKPT="$D/model_2560.pt"
[ -f "$CKPT" ] || { echo "NO_2560_CKPT"; exit 3; }
echo "== export =="
.venv-holoboxing/bin/python -m holoboxing.training.export --checkpoint "$CKPT" --output results/boxing-b5a.onnx || exit 4
echo "== validate =="
"$CONDA" run -n hbot-eval python -m holoboxing validate results/boxing-b5a.onnx \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --output results/b5a-validation.json --benchmark || exit 5
.venv-holoboxing/bin/python - <<'PY'
import json
v = json.load(open('results/b5a-validation.json'))
b = v.get('chain_benchmark', {}) or {}
print('VALIDATE_STATUS=%s PASSED=%s MEAN_MS=%s' % (
    v.get('status'), b.get('passed'), b.get('mean_ms')))
print('THRESHOLD_OK=%s' % (
    v.get('status') == 'PASS' and b.get('passed') is True
    and (b.get('mean_ms') or 999) <= 5.0))
PY
echo B5A_EXPORT_VALIDATE_DONE
