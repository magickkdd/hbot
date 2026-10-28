#!/usr/bin/env bash
# 在 ModelScope 实例上运行；把 Colab 训练所需的持久数据/控制器/资产/checkpoint 打成 tar。
# 生成后上传到私有 Google Drive: /MyDrive/hbot/hbot-colab-data.tgz
set -euo pipefail
cd /mnt/workspace
OUT=/mnt/workspace/hbot-colab-data.tgz

tar czf "$OUT" \
  --exclude='HoloMotion/holoboxing/data/processed/b7-boxing/robot_h5' \
  HoloMotion/holoboxing/data/processed/b7-boxing \
  HoloMotion/models/holomotion-v2 \
  HoloMotion/holoboxing/assets \
  HoloMotion/holoboxing/evaluation/runtime/compiled \
  HoloMotion/results/b1-prior \
  HoloMotion/results/boxing-b1-sl.onnx \
  HoloMotion/docs/b8 \
  HoloMotion/deployment/fall_attrib.py \
  model_3000.pt

sha256sum "$OUT" | tee /mnt/workspace/hbot-colab-data.sha256
ls -lh "$OUT"
echo "上传 $OUT 到私有 Drive: /MyDrive/hbot/hbot-colab-data.tgz"
echo "InterHuman/SMPL 数据只放私有 Drive 或加密包，不要公开分享。"
