#!/usr/bin/env bash
# 本地实时查看书生A100训练进度
# 用法:  bash /home/czx/HBOT/deployment/watch_ssheng.sh          # 每15秒刷新
#        bash /home/czx/HBOT/deployment/watch_ssheng.sh 5        # 每5秒刷新
#        bash /home/czx/HBOT/deployment/watch_ssheng.sh once     # 只看一眼
# 退出:  Ctrl+C
set -e
MODE="${1:-15}"
snapshot() {
  bash /home/czx/HBOT/deployment/ssheng_exec.sh '
    cd /root/hbot/HoloMotion
    echo "--- 进程 (PID CPU% 启动 命令) ---"
    ps aux | grep -E "[t]rain_sl|[t]rain_mujoco|[r]un_all.sh" | grep -v grep | cut -c1-230
    echo "--- GPU ---"
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
    echo "--- 最新输出目录 ---"
    LATEST=$(ls -td results/b7-sl-* results/b8-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
      echo "$LATEST"
      echo "--- 最新epoch指标 ---"
      tail -1 "$LATEST/metrics.jsonl" 2>/dev/null | head -c 420; echo
    fi
    echo "--- checkpoint列表(最新5个) ---"
    ls -t results/b7-sl*/epoch_*.pt results/b8-warmup*/model_*.pt results/b8-nfsp*/model_*.pt 2>/dev/null | head -5
    echo "--- 阶段日志尾部 ---"
    for L in sl.log warmup.log nfsp.log; do
      [ -f /data/logs/$L ] && echo "[$L]" && tail -2 /data/logs/$L | cut -c1-220
    done
  ' 2>/dev/null || echo "(连接失败，下轮重试...)"
}
while true; do
  clear
  echo "===== 书生A100 训练进度  本地时间 $(date '+%m-%d %H:%M:%S') ====="
  snapshot
  [ "$MODE" = "once" ] && break
  sleep "$MODE"
done
