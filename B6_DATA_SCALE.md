# HBOT B6 执行规格：全量现有数据 × 多种子 prior/SL × 并行自对弈

> 版本 v1.0 · 2026-10-25
> 上游：[B5_EXECUTION.md](B5_EXECUTION.md)、[PHASE2_BASELINE.md](PHASE2_BASELINE.md)、[records/winrate.md](records/winrate.md)
> 定位：B5a 的“数据太小 + 直臂 reward hack”之后的下一代数据/先验工程；不改变比赛接口、低层控制器和观测契约。
> 原则：**先把现有数据全部用上，把 prior/SL 基础打牢，再做多种子 RL；训练不空转，CPU/GPU 用起来但不超配。**
>
> **【2026-10-25 晚更新】B6 v1 的 RL 多种子部分暂缓；实际执行以 [B7_FULL_DATA_SL.md](B7_FULL_DATA_SL.md) 为准：InterHuman 全量 7,810 只进 prior，拳击相关子集 1,411 进 SL，单 seed、无 RL，SL 后直接 native 验收。**

---

## 0. 目标与非目标

### 目标

1. **全量现有数据**：把本地 RoboStriker 全量（86 npz / 43 组）和现有 InterHuman 3 个 clip 全部纳入新 prepared 数据集；不再只用 4 组/3 clip 的官方示例。
2. **多种子 prior/SL**：prior 和 SL 各跑 3 个 seed，按验证误差选最优，建立比 B1（prior 290 epochs、SL 19 epochs、8 clips）更强的动作表示/初始化。
3. **多种子并行 RL**：从最优 SL 出发，至少 2 个 RL seed 并行训练；资源允许再开第 3 个；每个 seed 先到 1280 updates，按效果续到 2560/4096。
4. **效果验收**：validate 双门槛 + native 4 局 + `fall_attrib.py` 事件归因 + 拳型/收拳技术指标 + 视频抽帧；与 H2b/B5a 基线比较。
5. **算力透明**：记录 GPU util/显存/功耗/CPU 核数/每 update 时间；并行度按实测吞吐决定，不凭感觉堆进程。

### 非目标

- 不改观测/动作接口（1097 → 225）、不改冻结低层控制器、不改比赛规则。
- 不引入原始网络视频管线（video→SMPL→retarget 另立项目）。
- 不在本规格内改 reward 语义。B6a 固定当前 `b5-impulse` 奖励，仅把“数据/prior/SL”作为变量；如果效果仍被直臂 hack，再开 B6b（reward/技术门控）。
- 不覆盖 B1/B5a/B5a2 的原始结果；B6 是独立谱系/目录。

---

## 1. 启动条件与现状

| 项 | 要求/现状 |
|---|---|
| B5a2 4096 | 训练完成/已归档或明确停止；GPU 空闲后再开始 B6 数据准备 |
| 旧候选 | H2b 仍为 `final-candidate`；B5a/B5a2 只是实验，不替换 |
| 数据 | 实例现有 raw 仅 RoboStriker 4 组 + InterHuman 3 clip；本地 `incoming/robostriker` 有 86 npz（43 组） |
| 训练代码 | 分支 `b5-impulse`（impulse reward，commit 留档）；B6 不额外改 reward |
| 硬件 | A10 23GB、16 vCPU、60GB RAM；当前单 RL 进程约 1 核 / GPU 40% / 3.4GB / ~3.8s per update |
| 会话 | 单次最长 8h；关键产物写 `/mnt/workspace`；长任务 tmux/nohup |

**B6a 唯一变量 = 数据规模 + prior/SL 质量。** RL 超参固定：128 envs、seed 42/43、`selfplay.json`、capacity 8、refresh 32、`b5-impulse` 奖励。

---

## 2. 全量数据准备（新数据集，不覆盖旧 prepared）

### 2.1 数据盘点

| 数据 | 数量 | 用途 |
|---|---:|---|
| RoboStriker 全量 | 86 npz / 43 组（本地 `incoming/robostriker`） | 拳击 G1 轨迹；新 prepared 的主数据 |
| InterHuman 现有 | 3 clip：1124（右拳，train）、1843（拳击对打，validation）、4033（右拳+躲闪，test） | 人类拳击参考；4033 保持 test，不进训练 |
| 旧 prepared | 8 clip / 11,538 role frames / 11,460 samples | 只作 B1/B5 基线，B6 不使用 |
| 全量 InterHuman | 7,810 motions（只有 source-listing，未下载） | 不在 B6a 范围；可选 B6-extra |

### 2.2 推送与完整性校验

```bash
# 本地 WSL
cd /home/czx/HBOT
find incoming/robostriker -name '*.npz' -type f | wc -l          # 期望 86
rsync -av --include='*/' --include='*.npz' --include='download.json' --exclude='*' \
  incoming/robostriker/ hbot:/mnt/workspace/HoloMotion/holoboxing/data/raw/RoboStriker_full/
# 本地生成清单（相对路径，便于远端原样校验）
(cd incoming/robostriker && find . -name '*.npz' -type f -print0 | sort -z | xargs -0 sha256sum) > /tmp/rs_full.sha256
scp /tmp/rs_full.sha256 hbot:/tmp/rs_full.sha256
```

```bash
# 实例
cd /mnt/workspace/HoloMotion
find holoboxing/data/raw/RoboStriker_full -name '*.npz' | wc -l   # 86
(cd holoboxing/data/raw/RoboStriker_full && sha256sum -c /tmp/rs_full.sha256) | tee /tmp/rs_check.log
grep -q FAILED /tmp/rs_check.log && { echo RS_FULL_FAIL; exit 1; } || echo RS_FULL_OK
```

- 保持旧 `raw/RoboStriker` 和 `processed/combined` 原样；新数据只写 `RoboStriker_full` 与 `processed/full-YYYYMMDD`。
- 若传输中断，rsync 续传，不重新下载。

### 2.3 新 prepared 数据集

```bash
cd /mnt/workspace/HoloMotion && source env.sh
OUT=holoboxing/data/processed/full-$(date +%Y%m%d)
.venv-holoboxing/bin/python -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker_full \
  --source holoboxing/data/converted/InterHuman \
  --output "$OUT" --assets "$HOLOBOXING_ASSETS" \
  --test-fraction 0.1 --workers 4
.venv-holoboxing/bin/python -m holoboxing.data verify \
  --output "$OUT" --assets "$HOLOBOXING_ASSETS"
```

验收：
- `$OUT/COMPLETE.json` 存在且 `status=PASS`；无 `FAILED.json`、无 `.partial`；
- 记录 `source_clips`、`supervised_samples`、`robot_h5_clips/frames`；与旧 8-clip 数据对比；
- 官方/确定性 split：test 不参与 prior/SL/AMP；train/validation 分开；split 记录在 `sources.json`；
- 新建 `env-full.sh`（不覆盖 `env.sh`）：

```bash
cat > env-full.sh <<EOF
export HOLOBOXING_ASSETS=/mnt/workspace/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/mnt/workspace/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=$PWD/$OUT/prepared
export PIP_CACHE_DIR=/mnt/workspace/.cache/pip
EOF
source env-full.sh
echo "$HOLOBOXING_DATA" && test -d "$HOLOBOXING_DATA" && echo ENV_FULL_OK
```

- 所有 B6 命令使用 `source env-full.sh`；旧 `env.sh` 保留给 B1/B5 复现。
- 资源：prepare 使用 GPU 测量采集；`--workers 4` 先试，若 CPU 空闲再提高；只在这个阶段占 GPU，不与 RL 并行。
- 主实验保留 test 作为诚实留出；如果最终确认“所有数据都要进训练”，在选定配置后另开 **B6-final**：把 test 并入 train 重训，只用于最终提交/native 验收，不与 prior/SL 的留出指标混用。

---

## 3. 训练矩阵（多种子 / 多轮）

### 3.1 Stage A：prior 3 seeds

```bash
cd /mnt/workspace/HoloMotion && source env-full.sh
REF=results/b6-reference
.venv-holoboxing/bin/python -m holoboxing.training.prepare_prior \
  --source holoboxing/data/raw/RoboStriker_full \
  --source holoboxing/data/converted/InterHuman \
  --assets "$HOLOBOXING_ASSETS" --output "$REF"

for SEED in 43 44 45; do
  .venv-holoboxing/bin/python -m holoboxing.training.train_prior \
    --data "$REF" --output "results/b6-prior-s$SEED" \
    --hidden-dims 512 256 --batch-size 1024 --epochs 1000 \
    --seed "$SEED" --device cuda:0 --max-runtime-seconds 3600
done
```

- 记录每个 seed 的 `epochs_completed`、best validation reconstruction/position/rotation、`stop_reason`。
- 以 validation position/rotation 选 best prior；**不要**只看训练 loss。
- 如果 full data 导致 prior 反而变差：先查 split/数据质量/retarget，不进入 RL。
- prior 训练便宜；3 seeds 可串行，也可 2 路并行；不要为此占用 RL 并行额度。

### 3.2 Stage B：SL 3 seeds

```bash
cd /mnt/workspace/HoloMotion && source env-full.sh
PRIOR=$(ls -dt results/b6-prior-s*/best.pt | head -1)   # 或按 validation 选定的路径
for SEED in 43 44 45; do
  .venv-holoboxing/bin/python -m holoboxing.training.train_sl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --data "$HOLOBOXING_DATA" --prior-checkpoint "$PRIOR" \
    --config holoboxing/training/configs/sample.json \
    --output "results/b6-sl-s$SEED" --epochs 19 --batch-size 512 \
    --seed "$SEED" --device cuda:0 --max-runtime-seconds 3600
done
```

- 记录 19 轮 train/validation position/rotation/loss；与 B1 SL 的 train 5.2cm / val 20.1cm 对比。
- 选择 validation position/rotation 最优的 SL；如果 full data 的 validation 没改善，先修数据/表示，不盲目 RL。
- 选中的 SL 路径写入 `results/b6-sl-final.txt`。

### 3.2.1 SL-only 基线（先不加 RL，看模仿能不能打出拳）

SL 是监督学习/动作预测，不是“看视频学格斗”：用 prepared 的 `(观测, 未来目标姿态)` 对训练 planner，loss 是位置/旋转误差，decoder 和低层控制器冻结；没有 reward、没有自博弈、没有探索。它学的是“给定当前状态，下一帧目标姿态像参考动作的哪一段”，不是“什么时候打/打哪里”。

在进入 RL 前，先把最优 SL 导出并做一次完整验收：

```bash
cd /mnt/workspace/HoloMotion && source env-full.sh
SL=$(cat results/b6-sl-final.txt)
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$SL/epoch_000019.pt" --output results/boxing-b6-sl-only.onnx
# validate / native 4 局 / fall_attrib 用 §5.4 的命令，把 candidate 换成
# results/boxing-b6-sl-only.onnx，输出目录换成 native-b6-sl-only-*
```

- 现成参考：`results/boxing-b1-sl.onnx` 是 B1 的 SL-only 模型（8-clip 数据、19 epochs），可先拿它跑 4 局 native；B1 SL vs RL 已知 0:0，但 native 尚未测。
- 评估与 RL 同口径：validate 双门槛、native 4 局、HIT/倒地/胜负、归因、视频、肘 ROM/收拳循环。
- **决策**：
  - SL-only 能稳定打出 HIT、视频里有明确拳型、且通过 validate → 记录为独立基线；若满足 §5.3 采纳条件，可以不跑 RL 直接作为候选。
  - SL-only 仍是 0 HIT/只摆姿势 → 不意外；它证明“纯模仿不够”，再进入 RL。不要把 SL-only 失败当成数据没学好，它本来就不含交互/时机学习。
- 这一步很便宜（导出+4 局约 10–20 分钟），**不要跳过**；它是判断“要不要 RL”的直接证据。

### 3.3 Stage C：RL 多种子（并行）

固定：
- `b5-impulse` 代码/奖励（impulse weight 200、complete_punch 0.05、active_hit_shaping 1.5）；
- `selfplay.json`、128 envs、capacity 8、refresh 32、seed 42/43；
- 从最优 SL 出发，先 warmup 32，再 selfplay 1280（目标 1280 selfplay updates）。

**B6 是独立谱系**：不要用 `--resume` 接 H2b/B5a；数据/prior/SL identity 已变，从新 SL 开始；旧候选和旧结果保留不动。

每个 seed 的命令：

```bash
cd /mnt/workspace/HoloMotion && source env-full.sh
SL=$(cat results/b6-sl-final.txt)
SEED=42   # 另一个窗口用 43；第三个 44 仅在资源校准通过时开

.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint "$SL/epoch_000019.pt" \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output "results/b6-warmup-s$SEED" \
  --num-envs 128 --updates 32 --seed "$SEED" --device cuda:0 \
  --max-runtime-seconds 3600

.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "results/b6-warmup-s$SEED/model_32.pt" \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 --output "results/b6-selfplay-s$SEED" \
  --num-envs 128 --updates 1280 --seed "$SEED" --device cuda:0 \
  --max-runtime-seconds 7200
```

- 每个 seed 独立输出目录/日志；不共享任何文件。
- 续 2560（仅在 1280 生效且时间足够，新目录）：
  ```bash
  .venv-holoboxing/bin/python -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$HOLOBOXING_DATA" \
    --resume results/b6-selfplay-s42/model_1280.pt \
    --config holoboxing/training/configs/selfplay.json \
    --training-mode selfplay --opponent-pool-capacity 8 \
    --opponent-refresh-updates 32 --output results/b6-selfplay-s42-r2 \
    --num-envs 128 --updates 1280 --seed 42 --device cuda:0 \
    --max-runtime-seconds 7200
  ```
- 先只跑 1280 并验收；不要默认跑满 2560/4096。

---

## 4. CPU/GPU 利用规格（不浪费算力）

### 4.1 已知基线（2026-10-25 实测）

- 单 RL：1 个进程约占 **1 个 vCPU（~98.8%）**、GPU **39–47%**、显存 **3.4GB/23GB**、功耗 ~80W、SM 1695MHz、约 **3.8s/update**。
- 机器：16 vCPU、60GB RAM；训练是单线程 MuJoCo + 小 MLP，GPU 在等 CPU；**并行独立 seed 是提升利用率最安全的方式**，不要改 128 envs（会破坏可比性和 resume 口径）。

### 4.2 并行度校准（RL 前 10 分钟）

1. 启动 1 个 seed，等编译完成、`metrics.jsonl` 至少 50 行后，采样 5 分钟：
   ```bash
   nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 5 | tee results/b6-gpu-1run.log
   ps -o pid,etime,%cpu,%mem,nlwp -p <pid>
   # update 时间以 metrics.jsonl 的 iteration/elapsed_seconds 为准，不用启动首分钟
   ```
2. 启动第 2 个 seed，再采样 5 分钟；计算：
   - 单 run 的 update 时间 `t1`；双 run 的每个 run 平均 `t2`（同样从 metrics 计算）；
   - 聚合吞吐 `2/t2` 对比 `1/t1`；只有聚合提升 **≥1.5×** 才保留双 run。
3. 第三个 seed 仅在以下全部满足时开：
   - GPU util **< 90%**、显存 **< 18GB**、无 OOM；
   - 每个 run 的 update 时间 **≤ 5.5s**（相对单 run 慢不超过 ~45%）；
   - 聚合吞吐相对单 run **≥2.2×**。
4. 任一条件不满足 → 回退到 2 或 1 个 run；**不要 3 个都慢一半**。

### 4.3 不浪费算力的硬规则

- **prior/SL**：验证 plateau 就早停；不要为了“多训练几次”在同一个 8-clip 上死磕 epoch；B6 的“多”是多种子和全量数据，不是重复无效 epoch。
- **RL**：并行 seed 用满 GPU 余量；如果 GPU 长期 <60% 且 CPU 有余，增加第二个 seed；如果 GPU >95% 或 OOM，降并发。
- **数据准备/评估/渲染**：不与 RL 同时跑（prepare 用 GPU；native eval/ffmpeg 也会抢资源）；按队列串行，训练结束立即接 eval，避免 GPU 空转。
- **日志/checkpoint**：每 300s 自动存；8h 上限前预留 10 分钟保存；不把关键文件写 `/root`/`/tmp`。
- **失败止损**：单 run 出现 NaN/loss 爆炸/update 时间翻倍，先停该 run，不拖累其他 seed。

---

## 5. 监控、门控与验收

### 5.1 prior/SL 门控

| 阶段 | 通过标准 |
|---|---|
| prior | 3 seeds 全部有 best/last；validation position/rotation 有记录；最优不差于 B1（7.3cm/0.265rad）才进入 SL |
| SL | 19/19 epochs；validation position/rotation 有记录；最优不差于 B1（20.1cm/0.818rad）才进入 RL |

- 若 full data 反而更差：检查 split、retarget、帧率/角色对齐、数据质量；不要用“训练 loss 更低”掩盖 validation 变差。

### 5.2 RL 门控

- 门 1（Δ≈100）：impulse 字段存在且非零；无 NaN；shaping/vel 不崩；否则 kill 该 run。
- 门 2（Δ≈300）：nonzero% 有爬升；shaping/vel 不崩；零倒地趋势；否则停止，记录。
- 1280 完成：`complete.json`/`training-result.json` stop_reason 正确；导出 `boxing-b6-s42/s43.onnx`。
- validate：`status=PASS` 且 `chain_benchmark.passed=true`（mean_ms≤5.0）。
- native：每侧 2 局，共 4 局 + `fall_attrib.py`；记录我方/对方 HIT、倒地、W-L、causal KO；视频抽帧。
- **技术门控（新增）**：右/左肘 ROM、收拳-出拳循环次数、HIT 前是否有回收、`complete_punch_reward` 是否非零；直臂摆动的模型即使 HIT 高也不采纳。

### 5.3 采纳

必须同时满足：
1. validate 双门槛；
2. 我方 HIT > 对方 HIT；
3. 败局 = 0；
4. 我方倒地 = 0；
5. 技术门控通过（有收拳/出拳循环，不是固定伸臂）；
6. 与 H2b/B5a 基线对比不劣化。

任一不满足：保留 H2b，B6 模型只归档；不更新 `final-candidate`。

### 5.4 导出 / validate / native 命令

```bash
cd /mnt/workspace/HoloMotion && source env-full.sh
SEED=42
CKPT=results/b6-selfplay-s$SEED/model_1280.pt
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$CKPT" --output "results/boxing-b6-s$SEED.onnx"

/mnt/workspace/miniforge3/bin/conda run -n hbot-eval python -m holoboxing validate \
  "results/boxing-b6-s$SEED.onnx" \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --output "results/b6-validation-s$SEED.json" --benchmark

export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
for SIDE in 0 1; do for R in a b; do
  /mnt/workspace/miniforge3/bin/conda run -n hbot-eval python -m holoboxing.training.evaluate_native \
    --candidate "results/boxing-b6-s$SEED.onnx" \
    --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
    --opponent-sha256 "$ROBOSTRIKER_SHA256" \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --candidate-side "$SIDE" \
    --output "results/native-b6-s$SEED-side$SIDE-$R" --device cuda:0
done; done

for SIDE in 0 1; do for R in a b; do
  D="results/native-b6-s$SEED-side$SIDE-$R"
  echo "===== $D (us=$SIDE) ====="
  python3 deployment/fall_attrib.py "$D/match.json" "$SIDE"
done; done
```

- validate 的 `--assets/--lower` 必须显式传；本地直接 Python 评测没有 Docker 镜像的默认路径。
- native 每个 seed 4 局；两个 seed 的 eval 可在显存允许时并行，但不要和训练同时跑。
- 技术指标从 `match.json` 的 `dof_pos` 计算（右肘索引 25，已核实）：肘 ROM、收拳-出拳循环、HIT 前回撤比例；并抽帧看视频。

---

## 6. 时间预算与砍单顺序

| 阶段 | 估时 | 说明 |
|---|---:|---|
| B5a2 4096 收尾/归档 | 0.5h | 先释放 GPU |
| 全量数据推送 + prepare + verify | 1.0–1.5h | 86 npz 传输 + 43 组 prepare |
| prior 3 seeds | 0.5–1.0h | 全量数据更大，仍远小于 RL |
| SL 3 seeds | 0.3–0.5h | 19 epochs/seed |
| SL-only 基线（先不 RL） | 0.3–0.5h | 导出 + 4 局 + 归因 + 视频；必做 |
| RL 2 seeds ×1280（并行） | 1.5–2.0h | 按校准；单 run 1280≈81min |
| 2× native 4 局 + 归因 + validate | 0.7–1.0h | 可 2 个 eval 并行（显存允许） |
| 最优 seed 续 2560（可选） | 1.3–1.6h | 仅生效时 |
| 归档/winrate/视频 | 0.3h | — |
| 合计（不含 2560） | ~5–6.5h | 单场 8h 可容纳 |

砍单顺序（从下往上砍）：
1. 第三个 RL seed（保留 2 个）；
2. 2560 续跑（1280 已验收且时间不足时）；
3. 4096 续跑；
4. **不可砍**：全量数据 prepare/verify、prior/SL 最优、至少 1 个 RL seed 到 1280、validate + native + 技术门控、归档。

---

## 7. 产物与归档

| 类型 | 路径 |
|---|---|
| 新数据 | `holoboxing/data/processed/full-YYYYMMDD/{COMPLETE.json,sources.json,prepared/}` |
| 数据清单 | `records/b6-data-manifest.json` + `sha256` 清单 |
| prior/SL | `results/b6-prior-s*/`、`results/b6-sl-s*/`、`results/b6-sl-final.txt` |
| RL | `results/b6-warmup-s*/`、`results/b6-selfplay-s*/`，每 seed 独立 |
| 导出/校验 | `results/boxing-b6-s42.onnx`、`results/b6-validation-s42.json` |
| SL-only | `results/boxing-b6-sl-only.onnx`、`results/b6-sl-only-validation.json`、`results/native-b6-sl-only-*` |
| native | `results/native-b6-s42-side{0,1}-{a,b}/`、`match.json`、`result.json` |
| 归因/技术 | `records/b6-native-attrib.txt`、`records/b6-technique.json`、视频 |
| 资源 | `records/b6-gpu-1run.log`、`records/b6-timings.md` |
| 记账 | `records/winrate.md` 追加 B6 段；采纳才更新 `records/final-candidate.txt` |

所有命令/日志/输出目录新开，不覆盖 B1/B5 产物。关键文件同步到 `/mnt/workspace/.bootstrap/` 或本地 `records/`。

---

## 8. 风险与处置

| 风险 | 处置 |
|---|---|
| 全量 prepare 超时/失败 | 保留旧 prepared；先扩到 16/24 组验证，再全量；记录失败原因，不硬跑 |
| 86 npz 传输中断 | rsync 续传；sha256 清单校验；不重新下载 |
| GPU OOM/并行变慢 | 按 §4.2 降到 2/1；记录实际吞吐；不要 3 个慢进程 |
| prior/SL full data 验证变差 | 查 split/retarget/对齐；停止，不进入 RL |
| RL 直臂 hack 仍在 | B6a 只记录数据规模效果；按 B6b 单独做 reward/技术门控，不污染 B6a 变量 |
| 8h 到界 | 每块 `--max-runtime-seconds`；checkpoint + 新目录续块；按砍单顺序 |
| native 4 局仍靠对手自倒 | 事件归因判定；HIT 不达标不采纳；不要用胜负替代 KPI |

---

## 9. 执行顺序（复现清单）

1. 等 B5a2 4096 归档/停止，`nvidia-smi` 确认空闲。
2. 本地 rsync 86 npz → 实例 `raw/RoboStriker_full`；sha256 86/86；prepare/verify 新 `full-YYYYMMDD`；`COMPLETE.json=PASS`；建 `env-full.sh`。
3. prior 3 seeds（43/44/45）→ 记录 validation → 选 best。
4. SL 3 seeds（43/44/45）→ 记录 validation → 选 best；写 `b6-sl-final.txt`。
5. **SL-only 基线**：导出 best SL → validate → native 4 局 + 归因 + 视频；记录 HIT/拳型。满足采纳条件可停在这里；否则继续。
6. RL 资源校准：1 run → 2 runs（→ 3 个仅在阈值内）；每 seed warmup 32 + selfplay 1280；门 1/门 2 监控。
6. 每个 seed 导出/validate；native 4 局 + 归因 + 技术指标 + 视频。
7. 按 §5.3 判定；生效则续 2560（时间够）→ 再验收；不生效则保留 H2b，B6 归档。
8. 更新 `records/winrate.md`、`records/final-candidate.txt`（仅采纳）、`instance_snapshot.md` 的 B6 记账。

---

## 附录 A：资源记录模板

```text
B6 GPU/CPU 记录（日期）
单 run: GPU util __% / mem __GB / power __W / CPU __核 / update __s
2 runs: GPU __% / mem __GB / 每 run update __s / 聚合吞吐 __×
3 runs: 是否启用 __；原因 __
数据 prepare: GPU __% / workers __ / 用时 __min
native eval: 并发 __；每局 __min
```

## 附录 B：为什么用并行 seed 而不是加大 num_envs

- `num_envs` 是 resume/对比口径的一部分，改它会破坏与 H2b/B5a 的可比性，且单 env CPU 仿真已单线程；把 envs 翻倍通常让每 update 变慢，不保证墙钟收益。
- 多个独立 seed 使用不同随机性，既能提高 GPU 利用率，也能给“多训练几次”提供真正的方差估计；这是当前最稳的算力利用方式。
- 只有资源校准证明 GPU 长期空闲且单核未饱和时，才考虑调 envs/线程；本规格默认不改。
