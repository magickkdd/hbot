#!/bin/bash
# 书生·端砚开发机 一键初始化（在 code-server 终端粘贴运行）
# 前提: /data/ 下已有 hbot-core.tgz hbot-debt.tgz（ModelScope 打包上传）
set -euo pipefail
BASE=/data/hbot
REPO=$BASE/HoloMotion
mkdir -p $BASE $BASE/logs
cd $BASE

echo "== 1/5 解包 =="
for T in /data/hbot-core.tgz /data/hbot-debt.tgz; do
  [ -f "$T" ] || { echo "MISSING $T"; exit 1; }
  tar xzf "$T" -C $BASE
done
ls $REPO/models/holomotion-v2/exported/model_10500.onnx >/dev/null && echo UNPACK_OK

echo "== 2/5 venv + torch cu128 =="
cd $REPO
if [ ! -x .venv-holoboxing/bin/python ]; then
  python3.11 -m venv .venv-holoboxing
  source .venv-holoboxing/bin/activate
  pip install -q --upgrade pip
  pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
  pip install -q -r holoboxing/training/requirements.txt
else
  echo "venv exists, skip install"
fi
.venv-holoboxing/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== 3/5 env.sh =="
cat > /data/env.sh <<EOF
export HOLOBOXING_ASSETS=$REPO/holoboxing/assets
export HOLOBOXING_LOWER=$REPO/models/holomotion-v2
export HOLOBOXING_DATA=$REPO/holoboxing/data/processed/b7-boxing/prepared
export ROBOSTRIKER_CHECKPOINT=$BASE/model_3000.pt
EOF
source /data/env.sh

echo "== 4/5 assets + prepared smoke =="
.venv-holoboxing/bin/python -m holoboxing assets --assets "$HOLOBOXING_ASSETS"
.venv-holoboxing/bin/python - <<'PY'
from holoboxing.training.supervised.data import Records
p = "/data/hbot/HoloMotion/holoboxing/data/processed/b7-boxing/prepared/manifest.json"
r = Records(p); v = Records(p, split="validation", allow_empty=True)
print("train", len(r.records), r.sample_count(), "| val", len(v.records), v.sample_count())
PY

echo "== 5/5 GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "SETUP_DONE — 停止/重启后只需: source /data/env.sh（venv 在 /data 持久）"
