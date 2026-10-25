# 烟雾测量（2026-10-25T02:57:24+08:00）

- 训练 venv ，A10，16 envs（RL）/ batch 1024（prior）/ 512（SL）
- prior 100 epochs：runtime 48.0s，stop=requested_progress_complete（无早停）
- SL 2 epochs：见 records/boxing-smoke-sl.json
- warmup(standing) 4 updates：runtime 141.3s，每 update 均值 3.98s，编译+非训练开销 ≈ 125.4s（= budget.runtime_seconds − 末行 elapsed_seconds）
- selfplay 4 updates：runtime 79.0s，每 update 均值 3.16s，编译+非训练开销 ≈ 66.4s
- 同进程两次 RL 调用开销 125s → 66s：第二个 RL 进程开销明显更低（编译缓存或预热复用），保守排程仍按每次 RL 调用付一次开销计
- pool snapshots = 2，iterations = [0, 2]（interval=2/updates=4，与代码推断一致，主方案 §2.3 推断成立）
- validate：status=PASS，chain_benchmark.passed=true，mean_ms=4.92ms（A10，阈值 5.0）
- 自对战（4-update 烟雾模型）：0:0、end_reason=TIME、hit/fall/KO 全 0、本地平均时延 0.59ms、~51 FPS
