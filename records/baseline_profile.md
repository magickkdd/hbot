# B1 基线画像（2026-10-25T03:44:14+08:00）

## 校验
- status=PASS, chain_benchmark.passed=true, mean_ms=4.79ms (A10, 阈值5.0)

## 对战画像（全部 0:0 / TIME / 零 hit-fall-KO）
- b1-match-self1: score [0, 0], end TIME, hits [0, 0], fps 49.8, latency 0.59ms
- b1-match-self2: score [0, 0], end TIME, hits [0, 0], fps 49.8, latency 0.59ms
- b1-match-self3: score [0, 0], end TIME, hits [0, 0], fps 49.8, latency 0.58ms
- b1-match-slvrl: score [0, 0], end TIME, hits [0, 0], fps 50.8, latency 0.58ms

## 判读
- 自对战 3 局结果完全一致 → 评测确定性正常
- SL vs RL 无交手：256-update 基线偏保守/防守，未主动出击；属基线真实水平，忠实记录
- 提交检查单第 8 项（fall≤2/局、无自身 KO）：满足（0/局）
