# HBOT 阶段 2 规格：完整基线（B1）

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.2 / 2026-10-24（v1.1 按代码复核修订；v1.2 二轮：预检预算 1.0h、standing/selfplay 分行外推、四段目录指针、续块 if 守卫、灰区统一、全段归档与 b1-timings、64 档补测命令） |
| 前置 | [PHASE1_SMOKE_TRAINING.md](PHASE1_SMOKE_TRAINING.md) **A 线**验收全绿，且 §5.2 归档已完成（`records/` 有烟雾实测数据） |
| 目标 | **官方参数一字不改**地完整复刻三阶段管线（prior 1000 → SL 19 → warmup 32 → selfplay 256），导出、校验、本地对战画像，并完成**首个平台提交** |
| 定位 | B1 是后续一切改进的对照组；本阶段不做任何 P1/P2 改动 |
| 预算 | B1 = 40% × 36h ≈ **14.4h**（分配表见 0.3，含 envs 预检；**每块启动前检查剩余预算，不足不启动**） |
| 硬约束 | ① 免费实例单次约 4h → 所有长跑**分块执行**（每块 ≤3h）；② 提交窗口北京时间 10:00–18:00，每天最多 3 次尝试；③ **仅 A 线**：SMPL+InterHuman prepared 就绪（B 线产物不进 B1，见 PHASE1 §5.4） |
| 上游文档 | [hbot_boxing_prelim_plan.md](hbot_boxing_prelim_plan.md) §5 / §7.2 / §7.3；[PHASE0_SETUP.md](PHASE0_SETUP.md) |

**执行方式**：agent 经 `ssh hbot` 执行；训练全部在 `tmux` 会话 `hbot` 内；**每个命令块自带 `cd /mnt/workspace/HoloMotion && source env.sh` 前缀**；训练命令一律用 `.venv-holoboxing/bin/python` 显式前缀（评测用 `conda run -n hbot-eval`）。命名规范：`results/b1-prior`、`b1-sl`、`b1-warmup`、`b1-selfplay[-rN]`、`boxing-b1.onnx`、`b1-validation.json`。

---

## 已核实的关键机制（本阶段分块执行的依据，出处见附录 D）

| 机制 | 行为 | 出处 |
|---|---|---|
| 时间预算 | 三个训练入口均支持 `--max-runtime-seconds`（含编译与保存的墙钟上限，0=关闭）；默认预留 600s 保存时间 | `training/time_budget.py:126-142` |
| 自动存档 | 预算内**每 300s** 存一次进度（RL 存 `model_<iteration>.pt`，prior 存 `last.pt`）；块结束（正常或到时）必存最终 checkpoint + `training-result.json` | `time_budget.py:18-19,94-101`；`on_policy_runner.py:545-559`；`train_prior.py:198-199,281` |
| 优雅停止 | SIGTERM/SIGINT 只"请求停止"，主循环在**update 边界**安全落盘；tmux 里 Ctrl+C 或 `kill -TERM` 都安全 | `time_budget.py:68-75,114-123` |
| 断点恢复 | `train_rl --resume <model_N.pt 文件>`；`--updates` 是**增量**；`--output` 必须是新空目录；**config 必须逐参数一致**（含 num-envs），否则报 `Resume format or configuration differs`；prior 恢复用 `last.pt` | `runners/resume.py:314-340`；`train_prior.py:86-118` |
| 完成判据 | `training-result.json` 的 `budget.stop_reason`：`requested_progress_complete` = 完成；`time_budget_reached` = 到块界 | `time_budget.py:103-112` |
| 异常快照 | 训练中抛异常写 `failure.pt`（仅诊断，**不可 resume**）；恢复回退到最近正常 checkpoint | `on_policy_runner.py:495-518` |
| 每步日志 | 每个 update 追加一行 JSON 到 `metrics.jsonl` | `on_policy_runner.py:527-544` |
| 运行完成标记 | `complete.json` 记 `updates_completed` | `training/scripts/train_mujoco.py:148-156` |
| 本地时延判据 | `chain_benchmark.passed` 的判据是 `mean_ms ≤ 5 ms`（`threshold_ms=5.0`），**严于官方 README 的 10 ms**；validate 本地模式的 `status=PASS` **不含**时延判定 | `evaluation/runtime/chain_benchmark.py`、`evaluation/validation.py` |

---

## 阶段总览

| 阶段 | 内容 | 产出 | 预算 |
|---|---|---|---|
| 0 | 前置核查、时长外推、分块排程、**envs 阶梯预检** | 排程表、envs 定档、记账 | 1.0h |
| 1 | Stage 1 先验正式（1000 epochs） | `b1-prior/best.pt` | 3.5h |
| 2 | Stage 2 SL 正式（19 epochs） | 最终目录 `epoch_000019.pt` + `b1-sl-final.txt` | 1.5h |
| 3 | Stage 3 热身正式（32 updates） | 最终目录 `model_32.pt` + `b1-warmup-final.txt` | 1.5h |
| 4 | Stage 3 自对弈正式（256 updates，官方复刻） | 最终块 `model_256.pt` | 5.5h |
| 5 | 导出 + validate/benchmark | `boxing-b1.onnx`、`b1-validation.json` | 1.0h（含阶段 6） |
| 6 | 本地对战画像 | 自对战 3 局 + SL vs RL、`baseline_profile.md` | — |
| 7 | 首个平台提交【用户手动】 | 平台验证成功 | — |
| 8 | 验收、记账、交接 | 验收表、records/ 归档 | 余量 0.4h |

---

## 阶段 0：前置核查与排程（agent）

### 0.1 门槛清单

- PHASE1 **A 线**验收表全绿；`records/` 已有：`boxing-smoke-validation.json`、各段 `training-result.json`、编译时长、每 update 均值、pool snapshots 结果；
- `ssh hbot` 可用；`cd /mnt/workspace/HoloMotion && source env.sh` 后**三个**变量非空；GPU 空闲（`nvidia-smi` 显存占用 ≈0）；
- A 线数据就绪：`holoboxing/data/processed/combined/COMPLETE.json` 存在。
- B 线数据（SMPL 未到）→ **B1 整体顺延**，不跑"假 B1"；等待期间可扩数据、预写阶段 3 文档。

### 0.2 时长外推工作表（用 PHASE1 归档数据填）

| 段 | 数据来源 | 外推结果 |
|---|---|---|
| prior 1000 epochs | PHASE1 实际完成轮数耗时 → `每轮耗时 × 1000`（早停则按实际轮数折算） | ___ h |
| SL 19 epochs | PHASE1 2 轮耗时 `× 9.5` | ___ h |
| RL 每 update·standing（正式档位） | **§0.4 standing 档实测**（16 envs 烟雾值不可直接外推，仅作 sanity） | ___ min |
| RL 每 update·selfplay（正式档位） | **§0.4 selfplay 档实测**（与 standing 分开取值，不共用） | ___ min |
| warmup 32 | `32 × standing每update + 编译` | ___ h |
| selfplay 256 | `256 × selfplay每update + 编译` | ___ h |

- 注意：PHASE1 的每 update 值来自 16 envs + interval=2 的烟雾，用于 128 envs + interval=256 会**低估**；§0.4 预检的 standing/selfplay 两档实测分别回填上表两行，并据此重算修订 0.3 分块计划（1-update 为粗估，warmup 首块后复核）。
- kernel 编译：待确认每个新进程是否重编译；排程按"每次 RL 调用付一次编译"保守计。

### 0.3 预算分配与分块计划（模板，按 0.2/0.4 实测修订）

| 段 | 预算上限 | 分块安排 |
|---|---|---|
| envs 预检（0.4） | 1.0h | 两模式各一次 1-update 探测，单次上限 1800s（含编译；两次上限合计 = 预算 1.0h） |
| prior | 3.5h | 首块 10800s；若需续块，续块 ≤1800s |
| SL | 1.5h | 单块 5400s |
| warmup | 1.5h | 单块 5400s（含编译） |
| selfplay | 5.5h | 块1 10800s + 块2 ≤9000s，**至多 2 块** |
| 导出+校验+对战 | 1.0h | — |
| 余量 | 0.4h | — |
| **合计** | **14.4h** | |

**两条硬门槛**：
1. **块前预算检查**：每块启动前核算「B1 已消耗 + 本块上限 ≤ 14.4h − 0.4h 余量」，不满足则**不启动**，停下来回报；
2. **触顶即停**：任何一段达到预算上限仍未完成 → 停止并回报，**不自动加块**（防止预算表形同虚设）。

记账：`instance_snapshot.md` 追加「B1 起点：日期 / 免费额度剩余」。

### 0.4 显存阶梯预检（一次性，为阶段 3/4 定死 envs 数）

> **重要**：`--resume` 要求 config 逐参数一致，**中途不能改 `--num-envs`**。128 是否可行必须现在定；不行就全程 64，并记录偏差①。16 envs 已由 PHASE1 覆盖，这里只测 128（64 仅在 128 失败时补测）。**standing 与 selfplay 各测一次**（初始化与开销不同）。

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
N=128
# 峰值显存后台采样（结束后取最大值）
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > /tmp/mem_$N.log 2>/dev/null &
MEMPID=$!

# ① standing 模式 1 update
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint results/boxing-smoke-sl/epoch_000002.pt \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/b1-envstest-$N-stand \
  --num-envs $N --updates 1 --seed 42 --device cuda:0 --max-runtime-seconds 1800

# ② selfplay 模式 1 update（用 PHASE1 的 smoke 模型开新阶段）
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint results/boxing-smoke-selfplay/model_4.pt \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 256 --output results/b1-envstest-$N-sp \
  --num-envs $N --updates 1 --seed 42 --device cuda:0 --max-runtime-seconds 1800

kill $MEMPID 2>/dev/null
sort -t, -k1 -n /tmp/mem_$N.log | tail -1   # 峰值显存（MiB）
for d in results/b1-envstest-$N-stand results/b1-envstest-$N-sp; do
  python -c "import json;r=json.load(open('$d/complete.json'));print('$d updates_completed =',r['updates_completed'])" 2>/dev/null || echo "$d 无 complete.json"
  ls $d/model_1.pt 2>/dev/null || echo "$d 无 model_1.pt"
done
# 每 update 实测（standing 与 selfplay 各自得出，分别回填 0.2 表两行）
for d in results/b1-envstest-$N-stand results/b1-envstest-$N-sp; do
  python - "$d" <<'EOF'
import json, sys
d = sys.argv[1]
r = json.load(open(d + '/training-result.json'))
rows = [json.loads(l) for l in open(d + '/metrics.jsonl')]
print(d, 'updates =', rows[-1]['iteration'],
      'elapsed =', rows[-1]['elapsed_seconds'],
      'runtime =', r['budget']['runtime_seconds'],
      '编译+开销 ≈', round(r['budget']['runtime_seconds'] - rows[-1]['elapsed_seconds'], 1))
EOF
done
```

- **判定（两档都满足才算 128 OK）**：`model_1.pt` 存在、`complete.json.updates_completed == 1`、`budget.stop_reason == "requested_progress_complete"`。**退出码 0 不算数**（时间预算可能在首个 update 前正常退出，只产出 `model_0.pt`）。
- 128 两档 OK → B1 全程 `--num-envs 128`（官方值）；任一档 FAIL → 用下方 64 档补测，双模式通过才定档 64，并记录「B1 偏差①：envs 128→64」；64 也 FAIL → 停止回报。
- 测试完成后清理（先确认清单再删，通配符只允许匹配 `b1-envstest-*`）：`ls -d results/b1-envstest-* && rm -rf results/b1-envstest-*`。
- 1-update 外推是**粗估**（不含长程效应与调度抖动）；warmup 首块结束后用实际 `elapsed_seconds ÷ updates` 复核 0.2 表并修订 0.3。

```bash
# 64 档补测（仅在 128 任一模式 FAIL 时执行；与 128 档命令相同，仅改 N 与目录名）
N=64
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5 > /tmp/mem_$N.log 2>/dev/null & MEMPID=$!
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint results/boxing-smoke-sl/epoch_000002.pt \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/b1-envstest-$N-stand \
  --num-envs $N --updates 1 --seed 42 --device cuda:0 --max-runtime-seconds 1800
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint results/boxing-smoke-selfplay/model_4.pt \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 256 --output results/b1-envstest-$N-sp \
  --num-envs $N --updates 1 --seed 42 --device cuda:0 --max-runtime-seconds 1800
kill $MEMPID 2>/dev/null
# 判定与每 update 实测：同上方 128 档流程
```

---

## 阶段 1：动作先验正式（1000 epochs）

### 1.1 运行

```bash
tmux new -s hbot 2>/dev/null || tmux attach -t hbot
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate

.venv-holoboxing/bin/python -m holoboxing.training.prepare_prior \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --assets "$HOLOBOXING_ASSETS" --output results/b1-reference

.venv-holoboxing/bin/python -m holoboxing.training.train_prior \
  --data results/b1-reference --output results/b1-prior \
  --hidden-dims 512 256 --batch-size 1024 --epochs 1000 \
  --seed 43 --device cuda:0 --max-runtime-seconds 10800
```

- 与官方唯一差异：`--max-runtime-seconds`（保险丝，不改变训练语义）。
- **验收**：最终目录（分块恢复时为 `-rN`）的 `best.pt` 与 `last.pt` 存在；日志显示 1000 轮完成或验证重构误差触发早停；把最终验证误差与实际完成轮数记入 `records/baseline_profile.md`。
- **登记指针**：`ls -dt results/b1-prior* | head -1 | tee results/b1-prior-final.txt`——后续阶段一律读本指针，不得硬编码首块目录。

### 1.2 中断恢复（若块到时未跑完）

恢复目标是 **`last.pt`**（最新训练状态；`best.pt` 是最佳验证轮快照，**不要**用它恢复，会回退轮次）：

```bash
LATEST=$(ls -t results/b1-prior*/last.pt 2>/dev/null | head -1)   # 跨续块目录找最新
.venv-holoboxing/bin/python -m holoboxing.training.train_prior \
  --data results/b1-reference --output results/b1-prior-r2 \
  --hidden-dims 512 256 --batch-size 1024 --epochs 1000 \
  --seed 43 --device cuda:0 --max-runtime-seconds 1800 \
  --resume "$LATEST"
```

- prior 的 resume 会校验 recipe/统计/设备一致并带回 best.pt；`--epochs` 目标必须 ≥ 已完成轮数（保持 1000）。
- 新块一律写**新输出目录**（`-r2`、`-r3`…），最终以最后一个目录里的 `best.pt` 为准；受 0.3 硬门槛约束，prior 续块预算合计 ≤3.5h。

---

## 阶段 2：监督学习正式（19 epochs）

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
.venv-holoboxing/bin/python -m holoboxing.training.train_sl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --data "$HOLOBOXING_DATA" --prior-checkpoint "$(cat results/b1-prior-final.txt)/best.pt" \
  --config holoboxing/training/configs/sample.json \
  --output results/b1-sl --epochs 19 --batch-size 512 \
  --seed 43 --device cuda:0 --max-runtime-seconds 5400
ls results/b1-sl/
```

- 产物命名 `epoch_{N:06d}.pt`，19 轮产出 `epoch_000019.pt`；记录最后一轮验证指标。
- **验收**：最终目录 `epoch_000019.pt` 存在；登记 `ls -dt results/b1-sl* | head -1 | tee results/b1-sl-final.txt`。

### 中断恢复（完整命令——batch/seed 必须与首块一致，默认值是 256/42，省略会校验失败）

```bash
LATEST=$(ls -t results/b1-sl*/epoch_*.pt | head -1)
.venv-holoboxing/bin/python -m holoboxing.training.train_sl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --data "$HOLOBOXING_DATA" --prior-checkpoint "$(cat results/b1-prior-final.txt)/best.pt" \
  --config holoboxing/training/configs/sample.json \
  --output results/b1-sl-r2 --epochs 19 --batch-size 512 \
  --seed 43 --device cuda:0 --max-runtime-seconds 5400 \
  --resume "$LATEST"
```

- `--epochs` 是**总计目标**（含已完成），保持 19；`--output` 换新目录。

---

## 阶段 3：站立热身正式（32 updates）

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint "$(cat results/b1-sl-final.txt)/epoch_000019.pt" \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/b1-warmup \
  --num-envs <0.4 定的值> --updates 32 --seed 42 --device cuda:0 \
  --max-runtime-seconds 5400
cat results/b1-warmup/training-result.json
```

- **验收（四项）**：最终目录 `model_32.pt` 存在；`budget.stop_reason == "requested_progress_complete"`；`complete.json` 的 `updates_completed == 32`；登记 `ls -dt results/b1-warmup* | head -1 | tee results/b1-warmup-final.txt`。

### 中断恢复（完整命令：复用首块全部参数，只改三处——start 参数换成 `--resume`、`--output` 换新目录、`--updates` 改为增量 `32-N`）

```bash
N=$(ls results/b1-warmup*/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
LATEST=$(ls -t results/b1-warmup*/model_*.pt | head -1)
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --resume "$LATEST" \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/b1-warmup-r2 \
  --num-envs <0.4 定的值> --updates $((32-N)) --seed 42 --device cuda:0 \
  --max-runtime-seconds 5400
```

> 注意：`--assets/--lower/--amp-motions/--config/--training-mode/--num-envs/--seed` 一个都不能省——`--amp-motions` 是必填项，且省略 `--config` 会退回默认配置直接触发 resume 校验失败。

---

## 阶段 4：自对弈正式（256 updates，官方复刻）

### 4.1 首块

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$(cat results/b1-warmup-final.txt)/model_32.pt" \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 256 --output results/b1-selfplay \
  --num-envs <0.4 定的值> --updates 256 --seed 42 --device cuda:0 \
  --max-runtime-seconds 10800
```

- 参数 = 官方教程原样（含 `interval=256/capacity=8`）。说明：**"256/256 只有 1 个对手快照"是由代码 `next_refresh = iteration + interval` 推断的**（PHASE1 只在 interval=2/updates=4 下实测过机制，未直接测 256/256）；B1 照官方复刻不改，跑完后在 4.3 用 `model_256.pt` 零成本复核该推断。P1 改进放阶段 3 文档。

### 4.2 分块监控与判读（自动跟踪最新块目录）

```bash
# 另开 tmux 窗口（Ctrl+B C）或本地另开 ssh：
watch -n 60 'D=$(ls -dt /mnt/workspace/HoloMotion/results/b1-selfplay* | head -1); \
  echo "DIR=$D"; ls -t $D/model_*.pt 2>/dev/null | head -3; \
  tail -1 $D/metrics.jsonl 2>/dev/null'
# 奖励趋势速览（字段名以实际为准，先看一行再取）：
python - <<'EOF'
import json, glob
d = sorted(glob.glob('results/b1-selfplay*/metrics.jsonl'))[-1]
rows = [json.loads(l) for l in open(d)]
print('keys:', [k for k in rows[-1]][:15]); print('updates:', rows[-1]['iteration'])
EOF
```

### 4.3 块间接续与最终产物登记

```bash
N=$(ls results/b1-selfplay*/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
LATEST=$(ls -t results/b1-selfplay*/model_*.pt | head -1)
echo "N=$N LATEST=$LATEST"
if [ "$N" -lt 256 ]; then
  # 下一块（增量 256-N，其余参数与首块完全一致；--updates 0 会被拒绝，必须有此守卫）
  .venv-holoboxing/bin/python -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$HOLOBOXING_DATA" \
    --resume "$LATEST" \
    --config holoboxing/training/configs/selfplay.json \
    --training-mode selfplay --opponent-pool-capacity 8 \
    --opponent-refresh-updates 256 --output results/b1-selfplay-r2 \
    --num-envs <0.4 定的值> --updates $((256-N)) --seed 42 --device cuda:0 \
    --max-runtime-seconds 10800
fi
# 完成后登记最终目录（无论几块；阶段 5/PHASE3 都读本指针）：
FINAL_DIR=$(ls -dt results/b1-selfplay* | head -1)
echo "$FINAL_DIR" | tee results/b1-final-dir.txt
```

- **验收（最终块，四项）**：`$FINAL_DIR/model_256.pt` 存在；`budget.stop_reason == "requested_progress_complete"`；`complete.json` 的 `updates_completed == 256`；`checkpoint_sha256` 记入 records。
- **零成本复核（推断验证）**：读 `model_256.pt` 的 `opponent_pool.snapshots`，应 == 1（与代码推断一致；不符 → 回填主方案 §2.3/§6 P1/附录 D）。
- 常见恢复报错 `Resume format or configuration differs` → 逐参数比对首块命令（config/num-envs/training-mode/capacity/interval/seed 一个都不能变）。
- 受 0.3 硬门槛约束：selfplay 至多 2 块；第二块结束仍未到 256 → 停止回报，不加第三块。

---

## 阶段 5：导出与校验

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
FINAL_DIR=$(cat results/b1-final-dir.txt)   # 或 ls -dt results/b1-selfplay* | head -1
ls "$FINAL_DIR/model_256.pt" || { echo "最终 checkpoint 缺失，回到 4.3"; exit 1; }
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$FINAL_DIR/model_256.pt" --output results/boxing-b1.onnx
ls -l results/boxing-b1.onnx               # 必须 < 50,000,000 字节

conda run -n hbot-eval python -m holoboxing validate results/boxing-b1.onnx \
  --output results/b1-validation.json --benchmark
python - <<'EOF'
import json
r = json.load(open('results/b1-validation.json'))
cb = r.get('chain_benchmark') or {}
print('status =', r.get('status'),
      '| chain_benchmark.passed =', cb.get('passed'),
      '| mean_ms =', cb.get('mean_ms'))
EOF
```

- **验收（两项同时）**：`status == "PASS"`；`chain_benchmark.passed == true`（即 `mean_ms ≤ 5`，本地唯一通过口径，严于官方 README 的 10ms）。`5 < mean_ms ≤ 10` **不通过本地验收**——如确需提交属例外路径（用户显式确认 + 平台云端实测，验收记录注明"灰区例外"，不计入全绿）；`> 10` 红且禁止提交。`mean_ms` 连同显卡型号记录归档。
- 归档（**本地 WSL 执行**；含全段 training-result 与目录指针——PHASE3 前置依赖）：

```bash
mkdir -p /home/czx/HBOT/records
scp hbot:/mnt/workspace/HoloMotion/results/b1-validation.json /home/czx/HBOT/records/
# 目录指针
for f in b1-final-dir.txt b1-prior-final.txt b1-sl-final.txt b1-warmup-final.txt; do
  scp "hbot:/mnt/workspace/HoloMotion/results/$f" /home/czx/HBOT/records/ 2>/dev/null \
    || echo "MISS: $f"
done
# 全段 training-result.json（含各段 runtime_seconds 实耗）
for d in $(ssh hbot 'cd /mnt/workspace/HoloMotion/results && ls -d b1-prior* b1-sl* b1-warmup* b1-selfplay*'); do
  scp "hbot:/mnt/workspace/HoloMotion/results/$d/training-result.json" "/home/czx/HBOT/records/$d.json" 2>/dev/null \
    || echo "MISS: $d"
done
# 最终 checkpoint 哈希
ssh hbot 'cd /mnt/workspace/HoloMotion && sha256sum "$(cat results/b1-final-dir.txt)/model_256.pt" results/boxing-b1.onnx' \
  | tee /home/czx/HBOT/records/b1-checkpoint-hashes.txt
# 用各段 runtime_seconds 汇总实际耗时到 records/b1-timings.md（PHASE3 成本估算输入）
```

---

## 阶段 6：本地对战画像

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
# 1) 基线自对战 3 局（match 无 seed 参数；每局必须新 --output-dir）
for i in 1 2 3; do
  conda run -n hbot-eval python -m holoboxing match \
    results/boxing-b1.onnx results/boxing-b1.onnx \
    --output-dir results/b1-match-self$i
done

# 2) SL vs RL 一局（先导出 SL）
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$(cat results/b1-sl-final.txt)/epoch_000019.pt" --output results/boxing-b1-sl.onnx
conda run -n hbot-eval python -m holoboxing match \
  results/boxing-b1-sl.onnx results/boxing-b1.onnx \
  --output-dir results/b1-match-slvrl

# 3) 摘要（先看结构再取字段）
python - <<'EOF'
import json, glob
for d in sorted(glob.glob('results/b1-match-*')):
    m = json.load(open(d + '/match.json'))
    print(d, {k: m[k] for k in list(m)[:10]})
EOF
```

- **画像清单（可量化项，写入 `records/baseline_profile.md`）**：每局 `hit_count / fall_count / knockdown_count / ko_count`、比分、`end_reason`；自对战 3 局结果是否完全一致（一致 = 评测确定性，属正常）。
- **倒地原因分类（脚滑/目标不可达/被击中）为可选项**：`match.json` 的 summary 不含原因标签；若 frames/contacts 数据足以人工判读则做二分类（倒地前短窗内是否被击中），否则仅记频次，完整归因需回放查看（租 Docker GPU，可选，不阻塞）。待确认 frames 字段可用性。
- **验收**：4 局 `match.json` + `replay.hdf5` 齐全，画像成文。

---

## 阶段 7：首个平台提交【用户手动，agent 备料】

### 7.1 提交前检查单（agent 逐项核对，全部 ✅ 才放行）

| # | 项 | 标准（可测） |
|---|---|---|
| 1 | validate | `b1-validation.json` 的 `status == PASS` **且** `chain_benchmark.passed == true` |
| 2 | 文件 | `boxing-b1.onnx` < 50,000,000 字节，sha256 已记录 |
| 3 | 时延 | `chain_benchmark.mean_ms ≤ 5`（即 `passed==true`）才放行；`5–10` 属例外路径：用户显式确认 + 平台实测并记录"灰区例外"；`>10` 禁止提交 |
| 4 | 窗口 | 当前北京时间在 10:00–18:00 内，且距 18:00 ≥2h（留验证排队） |
| 5 | 次数 | 当天尝试 <3，累计成功 <20 |
| 6 | 规则 | `#/details` 已截图存档（当天版本） |
| 7 | 旧模型 | 无（首个提交） |
| 8 | 画像 | 3 局自对战平均 `fall_count ≤ 2`/局且无自身 KO；超出则先归因再决定 |

### 7.2 操作步骤（用户）

1. 打开赛事官网模型提交入口，上传 `boxing-b1.onnx`（先 `scp` 到本地再上传，或按平台提供的通道）。
2. 等待平台验证完成；**当天一次成功后不再提交**。
3. 截图验证结果页面。

### 7.3 提交记录与失败排查（agent 建档）

`records/submissions.md` 追加：日期 / 文件名 + sha256 / 验证结果 / 当日尝试序号 / 累计成功数 / 备注。

验证失败时不猜测原因，按序排查：① 比对本地与平台侧 sha256（排除上传损坏）→ ② 按平台错误码定位（`MODEL_TOO_LARGE` / `FORMAT_CHECK_FAILED` / `CONTRACT_CHECK_FAILED` / 时延类）→ ③ 对照 `b1-validation.json` 的对应字段找差异 → ④ 当天剩余次数内重试 ≤1 次；仍失败则停止，次日再战。

---

## 阶段 8：验收、记账与交接

| 类别 | 检查项 | 通过标准 |
|---|---|---|
| 产物链 | prior/SL/warmup/selfplay | 四段 `training-result.json` 齐全；selfplay `stop_reason == "requested_progress_complete"` 且最终块 `model_256.pt` |
| 最终目录 | `results/b1-final-dir.txt` | 已登记，指向含 `model_256.pt` 的目录 |
| 偏差 | 与官方参数差异 | 仅 `--max-runtime-seconds`；若 envs 降档已记录为偏差① |
| 导出 | `boxing-b1.onnx` | <50MB、`status==PASS` 且 `chain_benchmark.passed==true`、`mean_ms` 已记录 |
| 对战 | 画像 | 4 局 + `baseline_profile.md` |
| 提交 | 平台状态 | 验证成功，模型已生效，`submissions.md` 已建档 |
| 预算 | B1 对账 | 实际消耗 vs 14.4h，记入 `instance_snapshot.md` |

**交付物**：远程 `results/b1-*` 全套 + 四个 `b1-*-final.txt` 指针；本地 `records/{b1-validation.json, baseline_profile.md, submissions.md, b1-*-final.txt, 各段 b1-*/training-result.json, b1-checkpoint-hashes.txt, b1-timings.md}` + 更新的 `instance_snapshot.md`。

**回填点**：`model_256.pt` 的 pool snapshots 复核结果；metrics.jsonl 字段名；分块实际次数；resume 是否一次通过——有出入即修订本文档与主方案附录 D。

**下一步**：阶段 3 改进与 A/B——见 [PHASE3_IMPROVEMENT.md](PHASE3_IMPROVEMENT.md)（P1：`capacity=8/interval=32` 自对弈，从 `b1` 的 actor 用 `--policy-checkpoint` 开新阶段续跑 + 与基线 A/B 对战、可选 P2/P1b 分支、循环赛选最终候选）。

---

## 附录 A：分块执行手册（selfplay 示例 walkthrough）

```text
块1: --policy-checkpoint "$(cat results/b1-warmup-final.txt)/model_32.pt" --output b1-selfplay --updates 256 --max-runtime-seconds 10800
  → 到时优雅停, 产 b1-selfplay/model_96.pt (N=96)
判读: training-result.json → budget.stop_reason == "time_budget_reached"
块2: --resume b1-selfplay/model_96.pt --output b1-selfplay-r2 --updates 160 (256-96) 其余参数同块1
  → 正常完成, 产 b1-selfplay-r2/model_256.pt
判读: stop_reason == "requested_progress_complete" ✅ 登记 FINAL_DIR, 结束
```

- 平台强制停止（非 SIGTERM）时：丢失自上次 update 边界存档以来的进度——**通常 ≤5 分钟，但若单个 update 本身超过 5 分钟则更多**（`budget.longest_update_seconds` 可查，可据此调大 `--checkpoint-interval-seconds`）；从最近 `model_<N>.pt` 恢复。`failure.pt` 只用于诊断，永远不要拿它 resume。
- **N=256（首块直接跑完）时跳过续块命令、直接登记 FINAL_DIR**：续块命令带 `if [ "$N" -lt 256 ]` 守卫——`--updates 0` 会被 `on_policy_runner.py:478` 拒绝。
- 每块开始前：`df -h /mnt/workspace` 确认磁盘 + **0.3 的块前预算检查**。

## 附录 B：产物与判读速查

| 文件 | 含义 | 用法 |
|---|---|---|
| `model_<N>.pt` | 第 N 个 update 后的可恢复 checkpoint | resume 输入 / 导出输入 |
| `training-result.json` | checkpoint 路径+sha256+bytes+progress+budget | `budget.stop_reason` 判完成；sha256 归档 |
| `complete.json` | `updates_completed` 等总结 | 阶段验收 |
| `metrics.jsonl` | 每 update 一行 JSON | 奖励/耗时趋势监控 |
| `environment.json` | 物理版本/接触参数记录 | 排查环境不一致 |
| `resume.json` | 本块从哪个 checkpoint 恢复 | 追溯链路 |
| `last.pt` / `best.pt`（prior） | 最新训练状态 / 最佳验证轮 | 恢复用 last.pt，导入用 best.pt |
| `failure.pt` | 异常诊断快照 | 只读不 resume |

## 附录 C：高频故障（承接 PHASE1 附录 A，新增 B1 专属）

| 现象 | 处理 |
|---|---|
| `Resume format or configuration differs` | 逐参数对齐首块命令；注意 `--num-envs`、config 路径、training-mode、capacity、interval、seed 全部一致 |
| 128 envs OOM | 已由 0.4 预检预防；若正式跑仍 OOM → 停止，全阶段改 64 **从头重跑**（resume 不能改 envs），记偏差① |
| selfplay 奖励长期不涨/持续下跌 | 记录 metrics 截图；若中间 checkpoint 存在可对比；B1 不调参，跑完为止（改进属阶段 3） |
| 实例 4h 被强停 | 重启实例 → PHASE0 §1.7 恢复连接 → `ssh hbot` → 按附录 A 从最近 checkpoint 续 |
| tmux 会话丢失 | **不要假设进程还在**：先 `tmux ls`、`ps -ef | grep train_rl`、`nvidia-smi` 三连确认；进程在 → 重新 attach 等待；进程不在 → 按附录 A 从最近 checkpoint 恢复（tmux server 被杀时子进程可能收到 SIGHUP 退出） |
| validate 本地 PASS 平台 FAIL | 按平台错误码逐项排查：先比对两端 sha256，再对照 `MODEL_TOO_LARGE` / `FORMAT_CHECK_FAILED` / `CONTRACT_CHECK_FAILED` / 时延（本地 5ms 判据已严于官方，时延类失败优先怀疑云端排队波动，次日重试一次） |

## 附录 D：机制出处（本仓库代码行号，2026-10-24 复核）

| 机制 | 文件:行 |
|---|---|
| 600s 保存预留 / 300s 存档间隔 | `training/time_budget.py:18-19` |
| 时间预算 CLI（add_budget_arguments） | `training/time_budget.py:126-142` |
| budget_from_arguments | `training/time_budget.py:145-156` |
| SIGTERM 优雅停 | `training/time_budget.py:68-75,114-123` |
| stop_reason 取值 | `training/time_budget.py:103-112` |
| RL 存档与最终落盘 | `training/rsl_rl/runners/on_policy_runner.py:545-559` |
| resume 校验 | `training/rsl_rl/runners/resume.py:314-340` |
| failure.pt | `training/rsl_rl/runners/on_policy_runner.py:495-518` |
| complete.json | `training/scripts/train_mujoco.py:148-156`（文件共 158 行） |
| prior 的 last.pt/best.pt | `training/train_prior.py:198-199,271,281` |
| chain_benchmark 5ms 判据 | `evaluation/runtime/chain_benchmark.py`（`threshold_ms=5.0`、`passed=mean<=5.0`） |
| validate 本地 PASS 不含时延 | `evaluation/validation.py`（仅 official 模式检查 `chain.passed`） |
