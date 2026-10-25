#!/usr/bin/env bash
# 本地一键连接 HBOT 的 ModelScope DSW 实例（tailnet）
# 用法:
#   bash /home/czx/HBOT/deployment/connect_hbot.sh            # 读取 .ts_ip 中记录的 IP
#   bash /home/czx/HBOT/deployment/connect_hbot.sh 100.x.x.x  # 显式指定 IP（首次/换 IP）
set -e
TS=~/.local/ts/tailscale
SOCK=$HOME/.local/share/ts/ts.sock
IP_FILE=/home/czx/HBOT/deployment/.ts_ip

IP="${1:-$(cat "$IP_FILE" 2>/dev/null || true)}"
if [ -z "$IP" ]; then
  echo "未找到实例 IP。先运行一次: connect_hbot.sh <实例100.x.x.x>，或把 IP 写入 $IP_FILE"
  exit 1
fi

# 1) 本地 userspace tailscaled（若未运行则拉起；已登录则直接复用）
if ! $TS --socket=$SOCK status >/dev/null 2>&1; then
  nohup ~/.local/ts/tailscaled --tun=userspace-networking \
    --state=$HOME/.local/share/ts/ts.state --socket=$SOCK \
    > $HOME/.local/share/ts/tailscaled.log 2>&1 &
  sleep 4
  $TS --socket=$SOCK up --hostname=czx-dev --timeout=30s 2>/dev/null || \
    $TS --socket=$SOCK up --hostname=czx-dev --timeout=30s
fi

# 2) 记录 IP 并连接
echo "$IP" > "$IP_FILE"
echo "连接 hbot-dsw ($IP:2222) ..."
exec ssh -i ~/.ssh/id_ed25519 -p 2222 \
  -o ProxyCommand="$TS --socket=$SOCK nc %h %p" \
  -o StrictHostKeyChecking=accept-new \
  root@"$IP"
