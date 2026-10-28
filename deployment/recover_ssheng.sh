#!/bin/bash
# 书生开发机 容器重建后一键恢复（v2，幂等）。网页终端只需: bash /data/recover.sh
# v2 固化了 2026-10-27 第 3 次重建踩到的三个坑：
#   A. /usr 被重置 → sshd/tmux/rsync 全丢（v1 只 WARN 不装，导致隧道活但 SSH 拒接、三链起不来）
#   B. hbot-code-lean.tgz 的包根是 holoboxing/ → v1 解到 /root/hbot，代码落在工作树之外
#   C. 任何 code 包都不含 holomotion/holosmpl/environments（底层控制器 python 包）
#      → ModuleNotFoundError: No module named 'holomotion'
# 修正：改用 /data/hbot-code-full.tgz（从已打补丁的活工作树打包，含全部代码）+ 先校验补丁再启链。
set -u
A=/data/hbot-access
REPO=/root/hbot/HoloMotion
PY=/root/hbvenv/bin/python
log() { echo "[recover] $*"; }

echo "== 1/8 基础依赖（apt→tuna）=="
need() { command -v "$1" >/dev/null 2>&1; }
if ! need sshd || ! need tmux || ! need rsync || [ ! -f /usr/lib/x86_64-linux-gnu/libOSMesa.so.8 ]; then
  . /etc/os-release
  mkdir -p /etc/apt/sources.list.d.disabled
  find /etc/apt/sources.list.d -maxdepth 1 -type f -exec mv {} /etc/apt/sources.list.d.disabled/ \; 2>/dev/null
  printf 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ %s main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ %s-updates main restricted universe multiverse\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ %s-security main restricted universe multiverse\n' \
    "$VERSION_CODENAME" "$VERSION_CODENAME" "$VERSION_CODENAME" > /etc/apt/sources.list
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || log "WARN apt update 失败（查白名单/镜像）"
  apt-get install -y -qq openssh-server tmux rsync libosmesa6 libglapi-mesa libgl1 libglx-mesa0 libglvnd0 libegl1 libglx0 libopengl0 || log "WARN apt install 失败"
fi
need tmux  || log "ERR 无 tmux，三链无法启动"
need rsync || log "WARN 无 rsync，watchdog 归档会失败"

echo "== 2/8 tailscale 隧道 =="
TS="$A/bin/tailscale"; [ -x "$A/bin/tailscale" ] || TS=$(command -v tailscale 2>/dev/null)
if [ -n "${TS:-}" ] && "$TS" --socket="$A/ts.sock" status >/dev/null 2>&1; then
  log "tailscaled 已在运行且已登录 → 不动它（避免打断现有 SSH）"
  "$TS" --socket="$A/ts.sock" serve --bg --tcp=2222 tcp://127.0.0.1:2223 2>/dev/null || true
  "$TS" --socket="$A/ts.sock" ip -4 2>/dev/null | tee "$A/receipt.txt"
elif [ -x "$A/bin/tailscaled" ]; then
  pgrep -x tailscaled >/dev/null 2>&1 && { kill "$(pgrep -x tailscaled | head -1)" 2>/dev/null; sleep 2; }
  nohup "$A/bin/tailscaled" --tun=userspace-networking --state="$A/tailscaled.state" --socket="$A/ts.sock" > "$A/tailscaled.log" 2>&1 &
  sleep 8
  "$TS" --socket="$A/ts.sock" up --hostname=ssheng-a100 --timeout=120s || log "WARN tailscale up 需人工授权（看上面 URL）"
  "$TS" --socket="$A/ts.sock" serve --bg --tcp=2222 tcp://127.0.0.1:2223 2>/dev/null
  "$TS" --socket="$A/ts.sock" ip -4 | tee "$A/receipt.txt"
else
  log "ERR $A/bin/tailscaled 缺失（从云盘重新上传 tailscale tgz）"
fi

echo "== 3/8 sshd（StrictModes 坑：authorized_keys 必须在 /root/.ssh）=="
if need sshd; then
  mkdir -p /root/.ssh /run/sshd && chmod 700 /root /root/.ssh
  cp "$A/authorized_key.pub" /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  grep -q '^AuthorizedKeysFile /root/.ssh/authorized_keys' "$A/sshd_config" 2>/dev/null || \
    sed -i 's|^AuthorizedKeysFile .*|AuthorizedKeysFile /root/.ssh/authorized_keys|' "$A/sshd_config"
  for P in $(pgrep -f "sshd -f $A/sshd_config" 2>/dev/null); do kill "$P" 2>/dev/null; done; sleep 1
  /usr/sbin/sshd -f "$A/sshd_config" && log "SSHD_UP"
fi

echo "== 4/8 venv =="
[ -x "$PY" ] || { log "untar venv-py312.tgz"; tar xzf /data/venv-py312.tgz -C /root; }

echo "== 5/8 代码（权威包 hbot-code-full.tgz）=="
mkdir -p /root/hbot "$REPO/results"
if [ ! -f /data/hbot-code-full.tgz ]; then
  log "WARN 无 code-full，退回 code-lean（注意包根是 holoboxing/，解到 \$REPO）"
  tar xzf /data/hbot-code-lean.tgz -C "$REPO"
else
  tar xzf /data/hbot-code-full.tgz -C /root/hbot
fi
[ -f "$REPO/holomotion/__init__.py" ] || { log "untar holomotion-pkg"; tar xzf /data/holomotion-pkg.tgz -C "$REPO"; }

echo "== 6/8 数据/资产（hbot-core.tgz：prepared 1228 文件 + models/holomotion-v2 + b1-prior）=="
if [ ! -f "$REPO/holoboxing/data/processed/b7-boxing/prepared/manifest.json" ]; then
  log "untar core"; tar xzf /data/hbot-core.tgz -C /root/hbot
fi
[ -f "$REPO/results/b1-prior/best.pt" ] || { log "WARN b1-prior 缺失，尝试从 checkpoints 归档补"; \
  cp "$(ls -t /data/checkpoints/latest/*/best.pt 2>/dev/null | head -1)" "$REPO/results/b1-prior/best.pt" 2>/dev/null || true; }

echo "== 7/8 manifest 修复 + 补丁校验 =="
cd "$REPO" || exit 1
if [ -f /data/fix_manifest.py ]; then
  N=$("$PY" -c "import json;print(len(json.load(open('holoboxing/data/processed/b7-boxing/prepared/manifest.json'))['records']))" 2>/dev/null || echo 0)
  if [ "$N" != "1166" ]; then
    log "manifest records=$N → 跑 fix_manifest"
    "$PY" /data/fix_manifest.py 2>&1 | tail -3
  else
    log "manifest 已是修复版 (1166)"
  fi
fi
if [ -f holoboxing/training/nfsp.py ]; then
  log "工作树已含 B8 补丁，跳过 overlay 重放"
elif [ -f /data/b8-overlay-replay.tgz ]; then
  log "重放 B8 overlay"; tar xzf /data/b8-overlay-replay.tgz -C "$REPO"
fi
find holoboxing -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
PATCH_OK=1
for chk in "holoboxing/training/nfsp.py:nfsp:3" "holoboxing/training/rsl_rl/algorithms/ppo.py:nfsp:6" \
           "holoboxing/training/rsl_rl/runners/on_policy_runner.py:nfsp:4" "holoboxing/training/train_rl.py:nfsp:5" \
           "holoboxing/training/export.py:average_policy:2" "holoboxing/training/supervised/data.py:_device_cache:2" \
           "holoboxing/data/interhuman.py:SKIPPED:1" "holoboxing/data/records.py:zero measured SL rows:1"; do
  F=${chk%%:*}; rest=${chk#*:}; K=${rest%%:*}; W=${rest##*:}
  G=$(grep -c "$K" "$F" 2>/dev/null || echo 0)
  [ "$G" -ge "$W" ] || { log "FAIL patch $F ($K=$G want>=$W)"; PATCH_OK=0; }
done
"$PY" -c "import holomotion.src.motion_tracking.link_observation, holoboxing.training.train_rl" 2>&1 | tail -2
"$PY" -m py_compile holoboxing/training/train_rl.py holoboxing/training/nfsp.py 2>&1 | tail -2
[ "$PATCH_OK" = "1" ] || log "ERR 补丁校验未过 → 不启动训练链（避免 while-true 空转）"

echo "== 8/8 checkpoint 回位 + 起三链 =="
restore_ckpt() { src=$1; dst=$2
  [ -e "$dst" ] && return 0; [ -e "$src" ] || return 0
  mkdir -p "$(dirname "$dst")" && cp "$src" "$dst" && log "restored $dst"; }
SL=$(ls -t /data/checkpoints/latest/b7-sl*/epoch_000019.pt 2>/dev/null | head -1)
[ -n "$SL" ] && restore_ckpt "$SL" "$REPO/results/b7-sl/epoch_000019.pt"
WU=$(ls -t /data/checkpoints/latest/b8-warmup*/model_200.pt 2>/dev/null | head -1)
[ -n "$WU" ] && restore_ckpt "$WU" "$REPO/results/b8-warmup/model_200.pt"
NF=$(ls -t /data/checkpoints/latest/b8-nfsp*/model_*.pt 2>/dev/null | head -1)
[ -n "$NF" ] && restore_ckpt "$NF" "$REPO/results/b8-nfsp-resume/$(basename "$NF")"
# 兜底：只有扁平 nfsp-latest.pt（run_all 每块会 cp 一份）时，按 ckpt 内部 iteration 命名成 model_<IT>.pt
# 否则 stage3 的 glob "model_*.pt" 看不见它 → 会 rm -rf 从零重跑（2026-10-27 已踩过）
if ! ls "$REPO"/results/b8-nfsp*/model_*.pt >/dev/null 2>&1 && [ -f /data/checkpoints/nfsp-latest.pt ]; then
  IT=$("$PY" -c "import torch;print(torch.load('/data/checkpoints/nfsp-latest.pt',map_location='cpu',weights_only=False).get('iteration',0))" 2>/dev/null)
  if [ -n "${IT:-}" ] && [ "$IT" -gt 0 ] 2>/dev/null; then
    mkdir -p "$REPO/results/b8-nfsp-resume"
    cp /data/checkpoints/nfsp-latest.pt "$REPO/results/b8-nfsp-resume/model_$IT.pt" && log "restored model_$IT.pt (from nfsp-latest.pt)"
  fi
fi
if [ "$PATCH_OK" = "1" ] && need tmux; then
  for S in run_all watchdog probes; do
    tmux has-session -t "$S" 2>/dev/null && { log "session $S 已在跑"; continue; }
    case $S in
      run_all)  tmux new-session -d -s run_all  "nohup bash /data/shusheng_run_all.sh > /data/logs/run_all.log 2>&1" ;;
      watchdog) tmux new-session -d -s watchdog "nohup bash /data/checkpoint_watchdog.sh > /dev/null 2>&1" ;;
      probes)   tmux new-session -d -s probes   "nohup bash /data/probe_auto.sh > /dev/null 2>&1" ;;
    esac
    log "started $S"
  done
fi
tmux ls 2>&1 | tr '\n' ' '; echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "RECOVER_DONE"
