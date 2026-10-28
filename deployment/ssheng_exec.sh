#!/usr/bin/env bash
set -e
TS=~/.local/ts/tailscale
SOCK=$HOME/.local/share/ts/ts.sock
IP=$(cat /home/czx/HBOT/deployment/.ssheng_ip 2>/dev/null)
[ -z "$IP" ] && { echo "无 IP"; exit 1; }
if ! $TS --socket=$SOCK status >/dev/null 2>&1; then
  nohup ~/.local/ts/tailscaled --tun=userspace-networking --state=$HOME/.local/share/ts/ts.state --socket=$SOCK > $HOME/.local/share/ts/tailscaled.log 2>&1 &
  sleep 4
fi
exec_retry() {
  local n=0 rc=0
  while [ $n -lt "${SSHENG_RETRY:-3}" ]; do
    n=$((n+1))
    ssh -i ~/.ssh/id_ed25519 -p 2222 \
      -o ProxyCommand="$TS --socket=$SOCK nc $IP 2222" \
      -o StrictHostKeyChecking=accept-new -o ConnectTimeout="${SSHENG_CONNECT_TIMEOUT:-15}" \
      -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      root@$IP "$@" && return 0
    rc=$?
    [ $n -lt "${SSHENG_RETRY:-3}" ] && sleep "${SSHENG_RETRY_SLEEP:-6}"
  done
  return $rc
}
exec_retry "$@"
