#!/usr/bin/env bash
# HBOT ModelScope DSW bootstrap (persistent + idempotent)
# Preferred entrypoint after every boot (run in WebIDE terminal):
#   nohup bash /mnt/workspace/.remote-access/post_boot.sh \
#     > /mnt/workspace/.remote-access/post_boot.log 2>&1 &
set -x
P=/mnt/workspace/.remote-access
B=/mnt/workspace/.bootstrap
REPO=/mnt/workspace/HoloMotion
mkdir -p "$P/ssh-host-keys" "$B/bin" "$B/logs" "$B/apt"
chmod 700 "$P" 2>/dev/null || true

# 0) Persistent environment
if [ -f "$REPO/env.sh" ]; then . "$REPO/env.sh"; fi
if ! grep -qF "$REPO/env.sh" /root/.bashrc 2>/dev/null; then
  echo ". $REPO/env.sh" >> /root/.bashrc
fi

pkill -9 tailscaled 2>/dev/null
sleep 1

# 1) Authorized key (must match local ~/.ssh/id_ed25519.pub)
cat > "$P/authorized_key.pub" << 'EOF_PUB'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPeI8GbiuJFuXYajFlzIz+WxjR8+xXDHcFYBIOgtmhYb czx@DESKTOP-E93227A
EOF_PUB
chmod 600 "$P/authorized_key.pub"

# 2) sshd @127.0.0.1:2223; prefer cached debs when the system disk was reset
if ! command -v sshd >/dev/null 2>&1; then
  if ls "$B"/apt/openssh-server*.deb >/dev/null 2>&1; then
    dpkg -i "$B"/apt/openssh-server*.deb >/dev/null 2>&1 || true
  fi
fi
if ! command -v sshd >/dev/null 2>&1; then
  apt-get update -qq || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server
fi
if [ ! -f "$P/ssh-host-keys/ssh_host_ed25519_key" ]; then
  ssh-keygen -t ed25519 -f "$P/ssh-host-keys/ssh_host_ed25519_key" -N '' -C hbot-dsw
fi
cat > "$P/sshd_config" << 'EOF_SSHD'
Port 2223
ListenAddress 127.0.0.1
PermitRootLogin yes
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile /mnt/workspace/.remote-access/authorized_key.pub
Subsystem sftp internal-sftp
HostKey /mnt/workspace/.remote-access/ssh-host-keys/ssh_host_ed25519_key
EOF_SSHD
mkdir -p /run/sshd
pkill -f 'sshd -f' 2>/dev/null
/usr/sbin/sshd -f "$P/sshd_config"
ss -ltn | grep 2223 && echo SSHD_UP

# 3) tailscale: prefer persisted binaries; cache after any install
TSBIN=$(command -v tailscale 2>/dev/null || true)
if [ -z "$TSBIN" ] && [ -x "$B/bin/tailscale" ]; then
  cp -fL "$B/bin/tailscale" /usr/bin/tailscale
  [ -x "$B/bin/tailscaled" ] && cp -fL "$B/bin/tailscaled" /usr/sbin/tailscaled
  chmod +x /usr/bin/tailscale /usr/sbin/tailscaled 2>/dev/null || true
  TSBIN=/usr/bin/tailscale
fi
if [ -z "$TSBIN" ]; then
  curl -fsSL https://tailscale.com/install.sh | sh
  TSBIN=$(command -v tailscale || echo /usr/bin/tailscale)
fi
TSDAEMON=$(command -v tailscaled 2>/dev/null || true)
[ -z "$TSDAEMON" ] && TSDAEMON=/usr/sbin/tailscaled
if [ -x "$TSBIN" ]; then cp -fL "$TSBIN" "$B/bin/tailscale" 2>/dev/null || true; fi
if [ -x "$TSDAEMON" ]; then cp -fL "$TSDAEMON" "$B/bin/tailscaled" 2>/dev/null || true; fi

# 4) tmux/rsync: small, but cache debs so a network outage cannot block recovery
if ! command -v tmux >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  if ls "$B"/apt/*.deb >/dev/null 2>&1; then
    dpkg -i "$B"/apt/*.deb >/dev/null 2>&1 || true
  fi
  command -v tmux >/dev/null 2>&1 || { apt-get update -qq || true; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null 2>&1 || true; }
  command -v rsync >/dev/null 2>&1 || { apt-get update -qq || true; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null 2>&1 || true; }
fi

# 5) tailscaled userspace + persistent state
rm -f "$P/ts.sock" 2>/dev/null || true
"$TSDAEMON" --tun=userspace-networking --state="$P/tailscaled.state" \
  --socket="$P/ts.sock" > "$P/tailscaled.log" 2>&1 &
sleep 4
if [ -n "${TS_AUTHKEY:-}" ]; then
  "$TSBIN" --socket="$P/ts.sock" up --authkey="$TS_AUTHKEY" --hostname=hbot-dsw --timeout=60s 2>"$P/up_err.log" || \
    "$TSBIN" --socket="$P/ts.sock" up --hostname=hbot-dsw --timeout=60s >>"$P/up_err.log" 2>&1
else
  "$TSBIN" --socket="$P/ts.sock" up --hostname=hbot-dsw --timeout=60s >"$P/up_err.log" 2>&1
fi
UP_RC=$?
IP=$("$TSBIN" --socket="$P/ts.sock" ip -4 2>/dev/null)
echo "UP_RC=$UP_RC TS_IP=$IP"

if [ -n "$IP" ]; then
  "$TSBIN" --socket="$P/ts.sock" serve --bg --tcp=2222 tcp://127.0.0.1:2223 2>"$P/serve_err.log" || \
    "$TSBIN" --socket="$P/ts.sock" serve --bg 2222 127.0.0.1:2223 >>"$P/serve_err.log" 2>&1
  "$TSBIN" --socket="$P/ts.sock" serve status 2>&1
fi

# 6) Persistent state summary (no secrets)
{
  echo "RECEIPT: IP=$IP UP_RC=$UP_RC HOST=hbot-dsw $(date)"
  echo "repo=$(test -d "$REPO" && echo OK || echo MISSING)"
  echo "onnx_bytes=$(stat -c%s "$REPO/models/holomotion-v2/exported/model_10500.onnx" 2>/dev/null || echo MISSING)"
  echo "py311=$(test -x /mnt/workspace/miniforge3/envs/py311/bin/python && echo OK || echo MISSING)"
  echo "train_venv=$(test -x "$REPO/.venv-holoboxing/bin/python" && echo OK || echo MISSING)"
  echo "eval_env=$(test -x /mnt/workspace/miniforge3/envs/hbot-eval/bin/python && echo OK || echo MISSING)"
} > "$P/receipt.txt"
cat "$P/receipt.txt"
echo BOOTSTRAP_DONE
