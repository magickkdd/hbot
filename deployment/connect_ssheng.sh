#!/usr/bin/env bash
# 本地一键连接书生开发机（tailnet）
# 用法: bash /home/czx/HBOT/deployment/connect_ssheng.sh [100.x.x.x]
set -e
TS=~/.local/ts/tailscale
SOCK=$HOME/.local/share/ts/ts.sock
IP_FILE=/home/czx/HBOT/deployment/.ssheng_ip

IP="${1:-$(cat "$IP_FILE" 2>/dev/null || true)}"
if [ -z "$IP" ]; then
  echo "未找到书生机 IP。先运行: connect_ssheng.sh <100.x.x.x>（IP 由 ssheng_remote_access.sh 打印）"
  exit 1
fi
if ! $TS --socket=$SOCK status >/dev/null 2>&1; then
  nohup ~/.local/ts/tailscaled --tun=userspace-networking \
    --state=$HOME/.local/share/ts/ts.state --socket=$SOCK \
    > $HOME/.local/share/ts/tailscaled.log 2>&1 &
  sleep 4
fi
echo "$IP" > "$IP_FILE"
echo "连接 ssheng-a100 ($IP:2222) ..."
exec ssh -i ~/.ssh/id_ed25519 -p 2222 \
  -o ProxyCommand="$TS --socket=$SOCK nc %h 2222" \
  -o StrictHostKeyChecking=accept-new \
  root@"$IP"
