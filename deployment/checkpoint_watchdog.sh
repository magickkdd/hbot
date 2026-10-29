#!/bin/bash
# v5：只归档当前训练目录，且每个目录只保留最新 1 份（单份 ~930MB，/data 配额 30G）
cd /root/hbot/HoloMotion || exit 1
while true; do
  for D in results/b12-04f; do
    [ -d "$D" ] || continue
    T="/data/checkpoints/latest/$(basename "$D")"
    mkdir -p "$T"
    rsync -a --include="*.pt" --include="*.json" --exclude="*" "$D/" "$T/" 2>/dev/null
  done
  cd /data/checkpoints/latest 2>/dev/null && for D in */; do
    ls -t "$D"model_*.pt "$D"epoch_*.pt "$D"last.pt 2>/dev/null | awk "NR>1" | xargs -r rm -f
  done
  cd /root/hbot/HoloMotion || exit 1
  sleep 300
done
