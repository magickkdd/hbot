#!/bin/bash
# 每5分钟: 滚动归档 checkpoint(每目录留最新2) + 额外维护单一最新副本 /data/checkpoints/nfsp-newest.pt
cd /root/hbot/HoloMotion || exit 1
while true; do
  NEWEST=""
  for D in results/b7-sl-* results/b7-sl results/b8-*; do
    [ -d "$D" ] || continue
    T="/data/checkpoints/latest/$(basename "$D")"
    mkdir -p "$T"
    rsync -a --include="*.pt" --include="*.json" --exclude="*" "$D/" "$T/" 2>/dev/null
    C=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
    if [ -n "$C" ]; then
      if [ -z "$NEWEST" ] || [ "$C" -nt "$NEWEST" ]; then
        NEWEST="$C"
      fi
    fi
  done
  if [ -n "$NEWEST" ]; then
    PREV=$(cat /data/checkpoints/.last-newest 2>/dev/null)
    if [ "$NEWEST" != "$PREV" ]; then
      mkdir -p /data/checkpoints
      rsync -a "$NEWEST" /data/checkpoints/nfsp-newest.pt
      echo "$NEWEST" > /data/checkpoints/.last-newest
      echo "$(date +%F\ %T) newest=$(basename "$NEWEST") -> nfsp-newest.pt" >> /data/logs/watchdog.log
    fi
  fi
  cd /data/checkpoints/latest 2>/dev/null && for D in */; do
    ls -t "$D"model_*.pt "$D"epoch_*.pt "$D"last.pt 2>/dev/null | awk "NR>2" | xargs -r rm -f
  done
  # chunk 目录每个约 2GB，只留最新 2 个（旧 chunk 的 checkpoint 已无续训价值，会吃满 30G 配额）
  ls -d b8-nfsp-r*/ 2>/dev/null | sort -r | awk "NR>2" | xargs -r rm -rf
  cd /root/hbot/HoloMotion || exit 1
  sleep 300
done
