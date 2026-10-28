#!/bin/bash
# 书生·端砚开发机 远程接入（在 code-server 终端粘贴运行；幂等，每次启动开发机后重跑一遍即可）
# 做的事: sshd@127.0.0.1:2223(密钥持久化在 /data) + tailscale userspace(state 持久化在 /data)
#         + tailscale serve 2222 -> 2223；首次运行 tailscale up 会打印一个授权 URL，浏览器打开登录即可
set -u
A=/data/hbot-access
mkdir -p "$A" /data/logs

echo "== 1/4 sshd =="
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update -qq || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync tmux || true
fi
if ! command -v sshd >/dev/null 2>&1; then echo "ERR: no sshd and apt failed"; exit 1; fi
command -v rsync >/dev/null 2>&1 || { apt-get update -qq || true; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; }
command -v tmux  >/dev/null 2>&1 || { DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux; }

if [ ! -f "$A/ssh_host_ed25519_key" ]; then
  ssh-keygen -t ed25519 -f "$A/ssh_host_ed25519_key" -N '' -C ssheng-a100
fi
cat > "$A/authorized_key.pub" <<'EOF_PUB'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPeI8GbiuJFuXYajFlzIz+WxjR8+xXDHcFYBIOgtmhYb czx@DESKTOP-E93227A
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPaZlTK0OHU2Gk4cUPOzlCcS8sXpaUILYDLZArhrbT0m trae-kaggle
EOF_PUB
cat > "$A/sshd_config" <<EOF_SSHD
Port 2223
ListenAddress 127.0.0.1
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile $A/authorized_key.pub
Subsystem sftp internal-sftp
HostKey $A/ssh_host_ed25519_key
EOF_SSHD
mkdir -p /run/sshd
pkill -f 'sshd -f' 2>/dev/null
/usr/sbin/sshd -f "$A/sshd_config"
ss -ltn | grep 2223 && echo SSHD_UP

echo "== 2/4 tailscale =="
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
TSBIN=$(command -v tailscale)
TSDAEMON=$(command -v tailscaled)
pkill -9 tailscaled 2>/dev/null; sleep 1
nohup "$TSDAEMON" --tun=userspace-networking --state="$A/tailscaled.state" \
  --socket="$A/ts.sock" > "$A/tailscaled.log" 2>&1 &
sleep 4

echo "== 3/4 加入 tailnet =="
if "$TSBIN" --socket="$A/ts.sock" status >/dev/null 2>&1; then
  echo "already logged in"
else
  if [ -n "${TS_AUTHKEY:-}" ]; then
    "$TSBIN" --socket="$A/ts.sock" up --authkey="$TS_AUTHKEY" --hostname=ssheng-a100 --timeout=60s
  else
    echo "浏览器打开下面 URL 登录你的 Tailscale 账号并批准设备:"
    "$TSBIN" --socket="$A/ts.sock" up --hostname=ssheng-a100 --timeout=600s
  fi
fi

echo "== 4/4 serve + IP =="
"$TSBIN" --socket="$A/ts.sock" serve --bg --tcp=2222 tcp://127.0.0.1:2223 2>/dev/null || \
  "$TSBIN" --socket="$A/ts.sock" serve --bg 2222 127.0.0.1:2223
"$TSBIN" --socket="$A/ts.sock" ip -4 | tee "$A/receipt.txt"
"$TSBIN" --socket="$A/ts.sock" status | head -5
echo "REMOTE_ACCESS_DONE — 把上面 100.x.x.x 发给 agent"
