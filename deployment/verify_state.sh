#!/bin/bash
# Verify persistent HBOT state after a reboot. Non-zero only on core failures.
set -u
REPO=/mnt/workspace/HoloMotion
B=/mnt/workspace/.bootstrap
. "$REPO/env.sh" 2>/dev/null || true
PASS=0
FAIL=0
pass() { echo "PASS $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL $1"; FAIL=$((FAIL + 1)); }

[ -d "$REPO" ] && pass "repo present" || fail "repo present"
[ -x "$REPO/.venv-holoboxing/bin/python" ] && pass "training venv" || fail "training venv"
[ -x /mnt/workspace/miniforge3/envs/py311/bin/python ] && pass "conda py311" || fail "conda py311"
[ -x /mnt/workspace/miniforge3/envs/hbot-eval/bin/python ] && pass "eval env" || fail "eval env"
[ -f "$REPO/env.sh" ] && pass "env.sh" || fail "env.sh"
command -v tmux >/dev/null 2>&1 && pass "tmux" || fail "tmux"
command -v rsync >/dev/null 2>&1 && pass "rsync" || fail "rsync"
[ -x /usr/bin/tailscale ] && pass "tailscale binary" || fail "tailscale binary"
[ -x /usr/sbin/sshd ] && pass "sshd binary" || fail "sshd binary"
ss -ltn 2>/dev/null | grep -q '127.0.0.1:2223' && pass "sshd listening 2223" || fail "sshd listening 2223"

ONNX="$REPO/models/holomotion-v2/exported/model_10500.onnx"
if [ -f "$ONNX" ] && [ "$(stat -c%s "$ONNX")" -eq 1647887681 ]; then
  pass "onnx size 1647887681"
else
  fail "onnx size 1647887681"
fi
if [ -f "$B/checksums/model_10500.onnx.sha256" ] && sha256sum -c "$B/checksums/model_10500.onnx.sha256" >/dev/null 2>&1; then
  pass "onnx sha256"
else
  fail "onnx sha256"
fi

if (cd "$REPO" && .venv-holoboxing/bin/python -m holoboxing assets --assets "$HOLOBOXING_ASSETS" >/tmp/hbot_assets.json 2>&1); then
  pass "assets manifest"
else
  fail "assets manifest"
fi
if (cd "$REPO" && .venv-holoboxing/bin/python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1); then
  pass "training torch cuda"
else
  fail "training torch cuda"
fi
if (cd "$REPO" && /mnt/workspace/miniforge3/envs/hbot-eval/bin/python -m holoboxing validate --help >/dev/null 2>&1); then
  pass "eval validate --help"
else
  fail "eval validate --help"
fi
if [ -d "$REPO/holoboxing/evaluation/runtime/compiled" ] && [ -n "$(find "$REPO/holoboxing/evaluation/runtime/compiled" -type f -print -quit 2>/dev/null)" ]; then
  pass "compiled observations"
else
  fail "compiled observations"
fi

for id in 1124 1843 4033; do
  if [ -f "$REPO/holoboxing/data/raw/InterHuman/motions/$id.pkl" ] && [ -f "$REPO/holoboxing/data/raw/InterHuman/annots/$id.txt" ]; then
    pass "InterHuman clip $id"
  else
    echo "INFO InterHuman clip $id missing"
  fi
done
for f in train val test; do
  if [ -f "$REPO/holoboxing/data/raw/InterHuman/split/$f.txt" ]; then
    pass "InterHuman split $f"
  else
    echo "INFO InterHuman split/$f.txt pending (user upload)"
  fi
done
[ -f "$REPO/holoboxing/data/raw/InterHuman/LICENSE.md" ] && pass "InterHuman LICENSE" || echo "INFO InterHuman LICENSE.md pending (user upload)"
RS_N=$(find "$REPO/holoboxing/data/raw/RoboStriker" -type f 2>/dev/null | wc -l)
echo "INFO RoboStriker files=$RS_N"
[ -f /mnt/workspace/licenses/neutral_smpl.pkl ] && pass "SMPL neutral" || echo "INFO SMPL neutral pending"

echo "--- disk ---"
df -h /mnt/workspace | tail -1
echo "VERIFY_PASS=$PASS VERIFY_FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "VERIFY_DONE (core failures: $FAIL)"
  exit 1
fi
echo "VERIFY_DONE"
