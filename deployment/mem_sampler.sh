#!/bin/bash
# 采样看守：每 20s 记一次内存/cgroup 计数器/心跳，用来判断容器是"被 OOM 打死"还是"被平台重建"。
# 容器重建后 mem.log 里最后一条的时间戳 = 死亡时刻；oom_kill 计数若在此之前上升 → OOM 实锤。
# 用法: tmux new-session -d -s monitor "nohup bash /data/mem_sampler.sh >/dev/null 2>&1"
set -u
OUT=/data/logs/mem.log
CG=/sys/fs/cgroup
mkdir -p /data/logs
while true; do
  {
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    MEM=$(awk '/MemAvailable/{printf "%.1fG", $2/1048576}' /proc/meminfo)
    CUR=$(cat $CG/memory.current 2>/dev/null)
    MAX=$(cat $CG/memory.max 2>/dev/null)
    EV=$(cat $CG/memory.events 2>/dev/null | tr '\n' ' ')
    TOP=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>1 && NR<=4 {printf "%s=%.1fG ", $2, $1/1048576}')
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
    U=$(cut -d. -f1 /proc/uptime)
    echo "$TS hb avail=$MEM cg=${CUR:-?}/${MAX:-?} $EV top=$TOP gpu=$GPU uptime=${U}s"
  } >> "$OUT"
  # 只留最近 ~3000 行（约 16 小时），避免把 /data 配额吃掉
  [ "$(wc -l < "$OUT" 2>/dev/null || echo 0)" -gt 3000 ] && tail -2000 "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  sleep 20
done
