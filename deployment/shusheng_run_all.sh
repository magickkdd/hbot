REPO=/root/hbot/HoloMotion; cd "$REPO" || exit 1
export HOLOBOXING_ASSETS=/root/hbot/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/root/hbot/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/root/hbot/HoloMotion/holoboxing/data/processed/b7-boxing/prepared
mkdir -p /data/logs results results/b7-sl
PY=$REPO/.venv-holoboxing/bin/python; [ -x "$PY" ] || PY=/root/hbvenv/bin/python
stage1() { # B7 SL
  if ls results/b7-sl*/epoch_000019.pt >/dev/null 2>&1; then return 0; fi
  while true; do
    CKPT=$(ls -t results/b7-sl*/last.pt results/b7-sl*/epoch_*.pt 2>/dev/null | head -1 || true)
    OUT="results/b7-sl-c$(date +%Y%m%d_%H%M%S)"
    [ -n "$CKPT" ] && START="--resume $CKPT" || START=""
    nice -n 10 "$PY" -m holoboxing.training.train_sl \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --data "$HOLOBOXING_DATA" --prior-checkpoint results/b1-prior/best.pt \
      --config holoboxing/training/configs/b8.json \
      --output "$OUT" --epochs 19 --batch-size 512 --seed 43 \
      --device cuda:0 --max-runtime-seconds 14400 $START 2>&1 | tee -a /data/logs/sl.log
    if ls results/b7-sl*/epoch_000019.pt >/dev/null 2>&1; then
      mkdir -p results/b7-sl
      cp "$(ls -t results/b7-sl*/epoch_000019.pt | head -1)" results/b7-sl/epoch_000019.pt; mkdir -p /data/checkpoints && cp results/b7-sl/epoch_000019.pt /data/checkpoints/; return 0
    fi
    sleep 10
  done
}
stage2() { # warmup 200
  while [ ! -f results/b8-warmup/model_200.pt ]; do
    if ls results/b8-warmup*/model_*.pt >/dev/null 2>&1; then
      CKPT=$(ls -t results/b8-warmup*/model_*.pt | head -1); START="--resume $CKPT"; OUT="results/b8-warmup-r$(date +%s)"
    else
      rm -rf results/b8-warmup
      START="--supervised-checkpoint results/b7-sl/epoch_000019.pt"; OUT="results/b8-warmup"
    fi
    nice -n 10 "$PY" -m holoboxing.training.train_rl \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --amp-motions "$HOLOBOXING_DATA" $START \
      --config holoboxing/training/configs/b8.json \
      --training-mode standing --output "$OUT" --num-envs 128 \
      --updates 200 --seed 43 --device cuda:0 --max-runtime-seconds 3600 \
      2>&1 | tee -a /data/logs/warmup.log
    sleep 5
  done
}
mkdir -p /data/checkpoints && cp results/b8-warmup/model_200.pt /data/checkpoints/ 2>/dev/null
stage3() { # NFSP 2000
  while true; do
    LATEST=$(ls -t results/b8-nfsp*/model_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LATEST" ]; then
      IT=$(basename "$LATEST" | sed -E "s/model_([0-9]+)\.pt/\1/"); mkdir -p /data/checkpoints && cp "$LATEST" /data/checkpoints/nfsp-latest.pt; [ "$IT" -ge 20000 ] && break
      START="--resume $LATEST --allow-resume-code-change"; OUT="results/b8-nfsp-r$(date +%s)"
    else
      rm -rf results/b8-nfsp
      START="--policy-checkpoint results/b8-warmup/model_200.pt"; OUT="results/b8-nfsp"
    fi
    nice -n 10 "$PY" -m holoboxing.training.train_rl \
      --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
      --amp-motions "$HOLOBOXING_DATA" $START \
      --config holoboxing/training/configs/b8.json \
      --training-mode selfplay --opponent-pool-capacity 8 \
      --opponent-refresh-updates 32 --nfsp \
      --nfsp-capacity 200000 --nfsp-batch 4096 --nfsp-lr 3e-4 --nfsp-interval 1 \
      --output "$OUT" --num-envs 256 --updates 4000 --seed 43 --device cuda:0 \
      --save-reserve-seconds 1200 --checkpoint-interval-seconds 2150 \
      --max-runtime-seconds 21600 2>&1 | tee -a /data/logs/nfsp.log
    sleep 5
  done
  echo B8_DONE
}
stage4() { # 双导出 + validate + native 4+4 局 + 归档持久盘
  if [ -f results/boxing-b8-avg.onnx ] && ls results/native-boxing-b8-avg-side1-a/match.json >/dev/null 2>&1; then return 0; fi
  bash /data/shusheng_export_eval.sh > /data/logs/export_eval.log 2>&1
  BEST=$(ls -t results/b8-nfsp*/model_2000.pt 2>/dev/null | head -1)
  [ -n "$BEST" ] && cp "$BEST" results/b8-final-model_2000.pt
  tar czf /data/b8-final-$(date +%Y%m%d_%H%M%S).tgz \
    results/boxing-b8-br.onnx results/boxing-b8-avg.onnx \
    results/boxing-b8-br-validation.json results/boxing-b8-avg-validation.json \
    results/native-boxing-b8-br-side*-a results/native-boxing-b8-avg-side*-a \
    results/b8-final-model_2000.pt 2>/dev/null || true
  echo B8_ALL_DONE
}
stage1 && stage2 && stage3 && stage4
