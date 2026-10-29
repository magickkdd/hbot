#!/bin/bash
# B12 阶段 04d（模仿锚定）：奖励与计分对齐后的交互优化（带自愈续跑）
#   1) 打补丁并自检（奖励项权重/阈值必须真的在文件里）
#   2) 从会出拳的 ckpt 起新对手池（初代快照就会打人），每 8 轮刷新对手
#   3) 进程死了自动从最新 ckpt 续跑，直到达到目标轮数
# 用法: bash b12_04b.sh <policy_ckpt> [target_updates]
set -u
REPO=/root/hbot/HoloMotion
BASE=${1:?需要起始 ckpt}
TARGET=${2:-2000}
OUT=${3:-results/b12-04d}
PY=/root/hbvenv/bin/python
DATA=$REPO/holoboxing/data/processed/b12-boxing/prepared

cd "$REPO" || exit 1

echo "== 0 补丁自检 =="
"$PY" /data/reapply_patches.py || { echo PATCH_FAIL; exit 1; }
"$PY" /data/apply_scored_hit.py "$REPO" || { echo HIT_PATCH_FAIL; exit 1; }
"$PY" -c "import ast,pathlib; [ast.parse(pathlib.Path(p).read_text()) for p in ('holoboxing/training/envs/mujoco_rewards.py','holoboxing/training/tasks/combat/mdp/rewards.py')]; print('COMPILE_OK')" || exit 1
grep -q '"scored_hit_reward"' holoboxing/training/envs/mujoco_rewards.py || { echo MISSING_HIT_TERM; exit 1; }
grep -q '"scored_hit_taken_reward"' holoboxing/training/envs/mujoco_rewards.py || { echo MISSING_TAKEN_TERM; exit 1; }
[ -f "$BASE" ] || { echo "CKPT_MISSING $BASE"; exit 1; }
echo "奖励项权重："
grep -A1 '"scored_hit_reward"\|"scored_hit_taken_reward"\|"impulse_strike_reward"' holoboxing/training/envs/mujoco_rewards.py | grep -E '^\s+-?[0-9]+\.0,' | tr -d ' '

run() {
  "$PY" -m holoboxing.training.train_rl \
    --assets "$REPO/holoboxing/assets" --lower "$REPO/models/holomotion-v2" --amp-motions "$DATA" \
    --config holoboxing/training/configs/b8-anchored.json \
    --training-mode selfplay --opponent-pool-capacity 8 --opponent-refresh-updates 8 \
    --nfsp --nfsp-capacity 200000 --nfsp-batch 4096 --nfsp-lr 3e-4 --nfsp-interval 1 \
    --output "$OUT" --num-envs 256 --updates "$1" --seed 47 --device cuda:0 \
    --checkpoint-interval-seconds 1800 --save-reserve-seconds 600 --max-runtime-seconds 21600 \
    "${@:2}"
}

echo "== 1 训练循环（$(date)）=="
while true; do
  CK=$(ls -t "$OUT"/model_*.pt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then
    echo "fresh from $BASE $(date)"
    run "$TARGET" --policy-checkpoint "$BASE"
  else
    IT=$("$PY" -c "
import json
try:
    print(json.loads(open('$OUT/metrics.jsonl').read().strip().split(chr(10))[-1])['iteration'])
except Exception:
    print(0)
")
    [ "$IT" -ge "$TARGET" ] && { echo "TARGET_REACHED $IT $(date)"; break; }
    REM=$((TARGET - IT))
    echo "resume $CK at $IT, remaining $REM $(date)"
    run "$REM" --resume "$CK" --allow-resume-code-change
  fi
  sleep 10
done

echo "== 2 存档（不打包，watchdog 已把最新 2 份同步到 /data）=="
ls -lh "$OUT"/model_*.pt | tail -3
echo B12_04B_DONE
