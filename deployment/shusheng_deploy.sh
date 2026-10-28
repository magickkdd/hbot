#!/bin/bash
# 书生机一键部署 v2：解包代码 → 重放 B8 overlay → smoke（依赖已装好）
set -euo pipefail
BASE=/data/hbot; REPO=/root/hbot/HoloMotion
cd /data
echo "== 1/5 解包代码到本地盘 =="
mkdir -p /root/hbot
tar xzf /data/hbot-code.tgz -C /root/hbot
echo "== 2/5 重放 B8 overlay（防 tar 覆盖补丁）=="
tar xzf /data/b8-code.tgz -C /root
echo "== 3/5 venv 存档 =="
[ -f /data/venv-py312.tgz ] || tar czf /data/venv-py312.tgz -C /root hbvenv
ls -lh /data/venv-py312.tgz
echo "== 4/5 smoke =="
export HOLOBOXING_ASSETS=/root/hbot/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/root/hbot/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/root/hbot/HoloMotion/holoboxing/data/processed/b7-boxing/prepared
cd $REPO
/root/hbvenv/bin/python -m holoboxing assets --assets "$HOLOBOXING_ASSETS" | tail -1
/root/hbvenv/bin/python - <<'PY'
from holoboxing.training.supervised.data import Records
p = "/root/hbot/HoloMotion/holoboxing/data/processed/b7-boxing/prepared/manifest.json"
r = Records(p); v = Records(p, split="validation", allow_empty=True)
print("train", len(r.records), r.sample_count(), "| val", len(v.records), v.sample_count())
import holoboxing.training.nfsp as nfsp; print("nfsp import OK", nfsp.FORMAT)
PY
/root/hbvenv/bin/python -m holoboxing.training.train_rl --help 2>&1 | grep -c "\-\-nfsp" >/dev/null && echo NFLAG_OK
echo "== 5/5 数据本地化确认 =="
ls /root/hbot/HoloMotion/holoboxing/data/processed/b7-boxing/prepared/manifest.json && echo DEPLOY_DONE
