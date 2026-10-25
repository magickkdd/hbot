#!/bin/bash
# One-command restore after every instance boot (run in WebIDE terminal).
set -u
R=/mnt/workspace/.remote-access
B=/mnt/workspace/.bootstrap
mkdir -p "$B/logs"
grep -q HOLOBOXING_ASSETS /root/.bashrc 2>/dev/null || cat >> /root/.bashrc <<"ENVEOF"
export HOLOBOXING_ASSETS=/mnt/workspace/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/mnt/workspace/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/mnt/workspace/HoloMotion/holoboxing/data/processed/combined/prepared
ENVEOF
bash "$R/bootstrap_hbot.sh" > "$B/logs/bootstrap_run.log" 2>&1
brc=$?
bash "$B/tools/verify_state.sh" > "$B/logs/verify_state.log" 2>&1
vrc=$?
{
  echo "POST_BOOT $(date)"
  cat "$R/receipt.txt" 2>/dev/null
  echo "bootstrap_rc=$brc verify_rc=$vrc"
  tail -40 "$B/logs/verify_state.log"
  echo "POST_BOOT_DONE"
} | tee "$R/post_boot_receipt.txt"
