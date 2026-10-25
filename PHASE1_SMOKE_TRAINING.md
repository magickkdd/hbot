# HBOT 阶段 1 规格：烟雾训练与导出验证

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.2 / 2026-10-24（v1.1 按代码复核修订；v1.2 二轮：耗时拆分算法修正、灰区与 5ms 统一、B 线三变量断言、归档清单含测量文件与失败暴露） |
| 前置 | [PHASE0_SETUP.md](PHASE0_SETUP.md) 阶段 0–2 全绿；阶段 3 数据就绪（A 线）或已登记 SMPL 阻塞（B 线） |
| 目标 | 用最小步数把 **prior → SL → RL → ONNX 导出 → validate/benchmark → 本地对战** 全链路在实例上跑通；产出第一个通过校验的 `boxing-smoke.onnx`；实测三项数据（对手池刷新语义、kernel 编译时长、每 update 耗时）供 PHASE2 排程 |
| 不追求 | 模型战力。本阶段产物只用于验证管线，不提交平台 |
| 预算 | A 线：目标 ≤3h，硬上限 3.5h（各命令 `--max-runtime-seconds` 上限合计 2h10m，另计数据准备/导出/校验/对战约 25min）；B 线再加 3.3B 与数据准备，目标 ≤3.5h。若总消耗超出主方案 B0（3.6h），从 B1 扣减并在 `instance_snapshot.md` 记账 |
| 上游文档 | [hbot_boxing_prelim_plan.md](hbot_boxing_prelim_plan.md) §4.4 / §5 / §7.1 |

**执行方式**：全部由 agent 经 `ssh hbot` 执行；训练一律放 `tmux` 会话 `hbot` 内。**每个命令块都自带 `cd /mnt/workspace/HoloMotion && source env.sh` 前缀**，不依赖前一 shell 的目录与环境。**A 线/B 线**：SMPL 已就绪走 A（完整数据）；SMPL 阻塞走 B（RoboStriker-only 降级，均为官方未文档化路径，冒烟本身就是验证）。

---

## 阶段总览

| 阶段 | 内容 | 产出 |
|---|---|---|
| 0 | 前置检查（env.sh 三变量、数据判定、额度基线） | 线路判定结论 |
| 1 | Stage 1 动作先验烟雾（100 epochs，上限 1800s） | `boxing-smoke-prior/{best.pt,last.pt}` |
| 2 | Stage 2 监督学习烟雾（2 epochs，上限 1200s；仅 A 线） | `boxing-smoke-sl/epoch_000002.pt` |
| 3 | Stage 3 RL 烟雾（warmup/selfplay 各 4 updates，上限各 2400s；**对手池语义实测**） | `model_4.pt` ×2、snapshots 记录 |
| 4 | 导出 + validate/benchmark + 自对战 | `boxing-smoke.onnx`、`boxing-smoke-validation.json`、`match.json` |
| 5 | 验收（A/B 分列）、记账、归档、发现回填 | 验收表、`records/` 归档 |

---

## 阶段 0：前置检查（agent）

### 0.1 环境变量文件（重要修正）

PHASE0 把三变量写进了 `~/.bashrc`，但 `ssh hbot '<命令>'` 的非交互 shell 可能不加载它。统一改用可 source 的 `env.sh`，本阶段所有远程命令前先 source：

```bash
ssh hbot
cd /mnt/workspace/HoloMotion
cat > env.sh <<'EOF'
export HOLOBOXING_ASSETS=/mnt/workspace/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/mnt/workspace/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/mnt/workspace/HoloMotion/holoboxing/data/processed/combined/prepared
EOF
source env.sh
[ -n "$HOLOBOXING_ASSETS" ] && [ -n "$HOLOBOXING_LOWER" ] && [ -n "$HOLOBOXING_DATA" ] \
  && echo ENV_OK || echo ENV_MISSING
```

- **验收**：输出 `ENV_OK`（三个变量全部非空）。B 线稍后会把 `HOLOBOXING_DATA` 临时指向别的目录（见 0.2），切换后**必须重跑本断言**。

### 0.2 数据就绪判定与线路选择

```bash
cd /mnt/workspace/HoloMotion && source env.sh
ls holoboxing/data/processed/combined/COMPLETE.json 2>/dev/null && echo LINE_A_READY
ls holoboxing/data/raw/RoboStriker | head -5
nvidia-smi --query-gpu=name,memory.used --format=csv
```

- 有 `COMPLETE.json` → **A 线**（`HOLOBOXING_DATA` 保持默认）。
- 没有 → **B 线**，需先补一步 RoboStriker-only 的 AMP 数据准备（`train_rl --amp-motions` 必填）：

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
python -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --output holoboxing/data/processed/robostriker-only \
  --assets "$HOLOBOXING_ASSETS" --test-fraction 0.1 --workers 2
python -m holoboxing.data verify \
  --output holoboxing/data/processed/robostriker-only --assets "$HOLOBOXING_ASSETS"
# 切换 env.sh 中的 HOLOBOXING_DATA 指向 robostriker-only 后，重跑 0.1 的完整三变量断言
sed -i 's#processed/combined/prepared#processed/robostriker-only/prepared#' env.sh && source env.sh
[ -n "$HOLOBOXING_ASSETS" ] && [ -n "$HOLOBOXING_LOWER" ] && [ -n "$HOLOBOXING_DATA" ] \
  && echo ENV_OK || echo ENV_MISSING
```

- **验收（B 线）**：`robostriker-only/COMPLETE.json` 存在、无 `FAILED.json`；`ENV_OK`。
- GPU 显存占用应接近 0（无残留训练进程；有则 `tmux ls` 检查处理）。

### 0.3 额度基线

记录当前免费额度剩余（用户口述或页面截图）到 `deployment/instance_snapshot.md`，标注「阶段 1 烟雾起点」。

---

## 阶段 1：动作先验烟雾

### 1.1 生成先验训练窗口

```bash
tmux new -s hbot 2>/dev/null || tmux attach -t hbot
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate

# A 线（双 source，官方命令）
python -m holoboxing.training.prepare_prior \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --assets "$HOLOBOXING_ASSETS" --output results/boxing-smoke-reference

# B 线：只传 RoboStriker 一个 --source，其余相同
```

- **验收**：`results/boxing-smoke-reference/` 非空、无 FAILED 类文件。

### 1.2 训练先验（烟雾规模）

```bash
python -m holoboxing.training.train_prior \
  --data results/boxing-smoke-reference --output results/boxing-smoke-prior \
  --hidden-dims 512 256 --batch-size 1024 --epochs 100 \
  --seed 43 --device cuda:0 --max-runtime-seconds 1800
ls results/boxing-smoke-prior/
```

- 与官方差异仅两处（属允许的烟雾差异，见 §5 回填规则）：`--epochs 100`（官方 1000，留给完整基线）、加 `--max-runtime-seconds`（三个训练入口均支持）。
- 产物：`best.pt`（最佳验证轮）与 `last.pt`（最新训练状态，断点恢复用；`train_prior.py:199,281`）。
- **验收**：`best.pt` 与 `last.pt` 存在；记录**实际完成的 epochs 数**（早停可能 <100）与最终验证重构误差。
- 失败处理：CUDA OOM → `--batch-size 512` 重跑（新 `--output` 目录）；数据相关报错 → 回 PHASE0 阶段 3。

---

## 阶段 2：监督学习烟雾（仅 A 线；B 线跳到 3.3B）

```bash
python -m holoboxing.training.train_sl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --data "$HOLOBOXING_DATA" --prior-checkpoint results/boxing-smoke-prior/best.pt \
  --config holoboxing/training/configs/sample.json \
  --output results/boxing-smoke-sl --epochs 2 --batch-size 512 \
  --seed 43 --device cuda:0 --max-runtime-seconds 1200
ls results/boxing-smoke-sl/
```

- `--epochs` 语义是**总计目标轮数**（含已完成）；命名规则 `epoch_{已完成轮数:06d}.pt`，2 轮产出 `epoch_000002.pt`。
- **验收**：`epoch_000002.pt` 存在（即 2 轮全部完成）。

---

## 阶段 3：RL 烟雾

### 3.1 站立热身（4 updates，含编译时长测量）

```bash
START_TS=$(date +%s)   # 记录启动时刻，用于编译时长拆分
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint results/boxing-smoke-sl/epoch_000002.pt \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/boxing-smoke-warmup \
  --num-envs 16 --updates 4 --seed 42 --device cuda:0 \
  --max-runtime-seconds 2400
echo "wall=$(( $(date +%s) - START_TS ))s"
ls results/boxing-smoke-warmup/ && cat results/boxing-smoke-warmup/training-result.json
```

- 首次运行会编译 GPU 物理 kernel，**10–30 分钟属正常**。
- **耗时拆分（PHASE2 排程依赖，算法以 metrics 为准）**：`metrics.jsonl` 每行含累计 `elapsed_seconds`（纯训练时间，不含编译/保存）。
  - `每 update 均值 = metrics.jsonl 末行 elapsed_seconds ÷ 完成 update 数`（已含首个 update，**不要**另行扣减首 update——那样会系统性低估）；
  - `编译+非训练开销 ≈ budget.runtime_seconds − 末行 elapsed_seconds`；
  - 如需单独记录"首个 update 完成时刻"，必须在训练进行中另开 shell 执行 `stat -c %Y metrics.jsonl` 抓取（文件最终 mtime 是最后一次写入，不能事后当首行时间用）；此项可选，上面两条已满足排程需要。
  - 待确认：kernel 编译是否每个新进程都重做（保守起见，排程按"每次 RL 调用都付一次编译"计）。
- **验收**：`model_4.pt` 存在；`complete.json` 的 `updates_completed == 4`；`budget.stop_reason == "requested_progress_complete"`。

### 3.2 自对弈（4 updates）+ 对手池语义实测（本阶段核心验证点）

```bash
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint results/boxing-smoke-warmup/model_4.pt \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 4 \
  --opponent-refresh-updates 2 --output results/boxing-smoke-selfplay \
  --num-envs 16 --updates 4 --seed 42 --device cuda:0 \
  --max-runtime-seconds 2400
```

- **验收 a**：`results/boxing-smoke-selfplay/model_4.pt` + `complete.json`（`updates_completed == 4`）存在。
- **验收 b（关键）**：`interval=2、updates=4`（iteration 0 与 2 各刷新一次）应得到 **snapshots == 2**：

```bash
python - <<'EOF'
import torch
ck = torch.load('results/boxing-smoke-selfplay/model_4.pt',
                map_location='cpu', weights_only=False)
pool = ck.get('opponent_pool')
if pool is None:
    print('opponent_pool 缺失，顶层键:', list(ck.keys()))
else:
    print('snapshots =', len(pool['snapshots']),
          'iterations =', [s.get('iteration') for s in pool['snapshots']])
EOF
```

- 该结果检验的是**刷新机制**（`next_refresh = iteration + interval`）在 `interval=2` 下的行为；据此**推断** `interval=256/updates=256` 时只有 1 个快照（该组合未直接实测，PHASE2 跑完 B1 后用 `model_256.pt` 零成本复核，见 PHASE2 §4.3）。若 snapshots==2 且 iterations==[0,2] → 推断成立；**不符 → 立即回填主方案 §2.3/§6 P1/附录 D**。

### 3.3B B 线：prior-only 直连 RL（无 SL，验证降级路径入口）

```bash
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --prior-checkpoint results/boxing-smoke-prior/best.pt \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/boxing-smoke-prioronly \
  --num-envs 16 --updates 4 --seed 42 --device cuda:0 \
  --max-runtime-seconds 1800
```

- **验收**：`model_4.pt` + `complete.json`（`updates_completed == 4`）存在。
- **结论口径（重要）**：本步只能证明「`--prior-checkpoint` 入口存在且 4-update standing 管线可跑」；**不能**证明完整训练稳定性或战力。主方案 §4.2/§9 的降级路径只有在后续跑通 prior-only selfplay 并做一次小规模 A/B 后，才可将预案从"待验证"升级为"可用"；本阶段仅在 `instance_snapshot.md` 登记「入口验证通过」。

---

## 阶段 4：导出与校验

### 4.1 导出 ONNX（A/B 线 checkpoint 分开取）

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
# A 线：
CKPT=results/boxing-smoke-selfplay/model_4.pt
# B 线（跳过了 3.2，没有 selfplay 产物，用 3.3B 的 prior-only standing 产物）：
# CKPT=results/boxing-smoke-prioronly/model_4.pt
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$CKPT" --output results/boxing-smoke.onnx
ls -l results/boxing-smoke.onnx
```

- **验收**：文件生成，< 50,000,000 字节；导出过程会在 CPU 上比对训练模型与 ONNX 输出（不一致会报错，报错即失败处理入口）。

### 4.2 直接 Python 校验 + 测时（无 Docker 路径）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
conda run -n hbot-eval python -m holoboxing validate results/boxing-smoke.onnx \
  --output results/boxing-smoke-validation.json --benchmark
python - <<'EOF'
import json
r = json.load(open('results/boxing-smoke-validation.json'))
print('status =', r.get('status'))
cb = r.get('chain_benchmark') or {}
print('chain_benchmark.passed =', cb.get('passed'),
      'mean_ms =', cb.get('mean_ms'), 'threshold_ms =', cb.get('threshold_ms'))
EOF
```

- **验收（必须同时满足）**：
  1. `status == "PASS"`（本地模式接口探测通过——**注意该字段不包含时延判定**，`validation.py` 只在 official 模式才因时延降级）；
  2. `chain_benchmark.passed == true`，其内部判据是 `mean_ms ≤ 5 ms`（`chain_benchmark.py` 中 `threshold_ms=5.0`，**严于官方 README 的 10 ms**）；
  3. `mean_ms` 已记录归档。
- 区间解读（与验收口径一致，无灰区放行）：`mean_ms ≤ 5`（即 `passed==true`）是**唯一**本地通过口径；`5 < mean_ms ≤ 10` **不通过本地验收**（本地代码判据严于官方 README 的 10ms）——如确需提交属例外路径：用户显式确认风险、依赖平台云端实测，并在验收记录注明"灰区例外"，**不得计入"验收全绿"**；`> 10` 红，禁止提交。
- 其他字段：`actual_fps=0` 表示未测量而非性能为零；`NOT_EVALUATED` 表示未做官方硬件验收；本地卡非 RTX 5090，数值仅参考。

### 4.3 自对战一局

```bash
cd /mnt/workspace/HoloMotion && source env.sh
conda run -n hbot-eval python -m holoboxing match \
  results/boxing-smoke.onnx results/boxing-smoke.onnx \
  --output-dir results/boxing-smoke-match
ls results/boxing-smoke-match/
python - <<'EOF'
import json
m = json.load(open('results/boxing-smoke-match/match.json'))
print({k: m[k] for k in list(m)[:12]})   # 先看结构，再定位比分/事件字段
EOF
```

- **验收**：`match.json` + `replay.hdf5` 生成；从 `match.json` 记录比分、倒地、KO 事件数到 `instance_snapshot.md`。
- 回放观看：实例上无 `viewer/dist`，本阶段只做 JSON 分析；需要看动画时租 1–2h Docker GPU 用官方 `run.sh --view`（可选，不阻塞）。

---

## 阶段 5：验收、记账与归档

### 5.1 验收表（按线路分列）

| 类别 | 检查项 | A 线标准 | B 线标准 |
|---|---|---|---|
| 管线 | prior | `best.pt`+`last.pt` 存在，完成轮数与验证误差已记录 | 同左 |
| 管线 | SL | `epoch_000002.pt` 存在 | 跳过（登记阻塞） |
| 管线 | warmup | `complete.json.updates_completed==4` 且 `stop_reason==requested_progress_complete` | 跳过 |
| 管线 | selfplay / prior-only | 同上字段标准（selfplay） | 同上字段标准（prioronly） |
| 导出 | `boxing-smoke.onnx` | 大小合规，导出比对通过 | 同左 |
| 校验 | validate + benchmark | `status==PASS` **且** `chain_benchmark.passed==true`，`mean_ms` 已记录（5–10ms 灰区=不通过；例外须用户确认并注明，不算全绿） | 同左 |
| 对战 | 自对战 1 局 | `match.json`/`replay.hdf5` 生成，事件已摘录 | 同左 |
| 实测① | 对手池 snapshots | 数值与 iterations 已记录；与代码推断一致或已触发回填 | 仅推断未实测，登记 |
| 实测② | kernel 编译时长 | 已按 3.1 方法拆分记录 | 同左 |
| 记账 | 额度 | 阶段消耗已记入 `instance_snapshot.md` | 同左 |

### 5.2 归档清单（先远程落盘测量结论，再本地归档）

```bash
# ① 远程：把测量结论落成文件（编译拆分、每 update 均值、pool snapshots——PHASE2 §0.2 直接依赖）
ssh hbot 'cat > /mnt/workspace/HoloMotion/results/smoke-measurements.md <<EOF
# 烟雾测量（$(date -Iseconds)）
编译+非训练开销 ≈ ___ s（= budget.runtime_seconds − 末行 elapsed_seconds）
每 update 均值 ≈ ___ s（= 末行 elapsed_seconds ÷ 完成 update 数）
pool snapshots = ___，iterations = [___]
EOF'
```

```bash
# ② 本地 WSL 执行：归档（只跳过"确认不存在"的目录；真实 scp 失败必须暴露）
mkdir -p /home/czx/HBOT/records
scp hbot:/mnt/workspace/HoloMotion/results/boxing-smoke-validation.json /home/czx/HBOT/records/
scp hbot:/mnt/workspace/HoloMotion/results/boxing-smoke-match/match.json /home/czx/HBOT/records/boxing-smoke-match.json
scp hbot:/mnt/workspace/HoloMotion/results/smoke-measurements.md /home/czx/HBOT/records/
for d in boxing-smoke-prior boxing-smoke-sl boxing-smoke-warmup boxing-smoke-selfplay boxing-smoke-prioronly; do
  if ssh hbot "[ -d /mnt/workspace/HoloMotion/results/$d ]"; then
    scp "hbot:/mnt/workspace/HoloMotion/results/$d/training-result.json" "/home/czx/HBOT/records/$d.json" \
      || echo "SCP_FAIL: $d"
  else
    echo "SKIP(不存在): $d"
  fi
done
ls -l /home/czx/HBOT/records/   # 核对归档清单与数量
```

（B 线不存在的目录按 SKIP 跳过；出现 `SCP_FAIL` 必须处理后才能进入 PHASE2。）

### 5.3 发现回填清单

**回填规则**：仅当**命令行参数名、参数语义、默认取值或运行行为**与官方教程/代码不符时回填；以下属允许的烟雾差异，不触发回填：`--epochs/--updates` 缩小、`--num-envs` 缩小、`--opponent-refresh-updates/--opponent-pool-capacity` 调整、`--output` 目录名、`--max-runtime-seconds` 等预算参数、B 线单 source。

1. snapshots ≠ 2 或 iterations ≠ [0,2] → 修主方案 §2.3、§6 P1、附录 D；
2. prior-only 入口不可用（报参数/格式错误）→ 修主方案 §4.2、§9 的预案表述；
3. 命令行为与官方不符（按上述规则判定）→ 以 `--help` 实测为准修订本文档与主方案。

### 5.4 下一步

- **A 线**：验收全绿 → 进入 [PHASE2_BASELINE.md](PHASE2_BASELINE.md)（完整基线 B1）。
- **B 线**：验收通过也**不得进入 B1**（B1 要求完整数据的官方复刻）；唯一出口是登记 SMPL 阻塞、等待许可后补 PHASE0 §3.4，转 A 线再进 B1。等待期间可做：扩数据下载、预写阶段 3 文档。

---

## 附录 A：高频故障

| 现象 | 处理 |
|---|---|
| CUDA OOM | `--num-envs 8` 或 `--batch-size 512`；换新 `--output` 目录重跑 |
| `Output already exists` | 官方要求新目录；`--output results/<新名>` |
| 首次 RL 卡住无输出 | kernel 编译中，等；看 `training-result.json` 而非进程退出码 |
| 训练被实例 4h 上限杀掉 | tmux 内跑 + `--max-runtime-seconds` 留余量；重启后 `--resume` 续跑同阶段 |
| `--resume` 与 `--policy-checkpoint` 混用报错 | 二者互斥：续跑同阶段用前者，开新阶段用后者 |
| validate 报缺 assets/lower | `source env.sh` 后再跑；或显式加 `--assets/--lower` |
| torch.load 读 checkpoint 报格式错 | 先打印顶层 keys 定位；必要时查 `holoboxing/training/checkpoint_io.py` |
| 新 shell 里 `python -m holoboxing.training.*` 报缺包 | 未激活训练 venv；用 `.venv-holoboxing/bin/python -m ...` 显式前缀 |

## 附录 B：与上游文档的回填对照

| 本文档步骤 | 上游受影响位置 |
|---|---|
| 3.2 对手池实测（interval=2） | 主方案 §2.3、§6 P1、附录 D |
| 3.3B prior-only 入口验证 | 主方案 §4.2、§9 |
| 4.2 时延口径（代码 5ms vs README 10ms、本地 PASS 不含时延） | 主方案 §5.4、§7.1、附录 D |
| 0.1 env.sh 修正 | PHASE0 §2.2（bashrc 非交互不加载的补充说明） |
