# HBOT 阶段 3 规格：改进与 A/B（B2）

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.1 / 2026-10-24（第二轮代码复核修订：P1 归因口径与 current-best 指针、续块 if 守卫、灰区与 5ms 统一、P2 patch 流程与 config 检查、随机性核查局、冻结线量化、P1b 完整命令、候选清单机制） |
| 前置 | [PHASE2_BASELINE.md](PHASE2_BASELINE.md) 验收全绿，且 §5 归档已完成：`records/` 有 B1 四段 `training-result.json`（含实耗）、`b1-*-final.txt` 指针、`b1-checkpoint-hashes.txt`、`b1-timings.md` |
| 目标 | 用 B2 预算产出一个**经本地 A/B 验证不弱于基线**的改进模型（首选 P1：`capacity=8/interval=32`），选出最终候选 checkpoint，并完成 2–3 次以收集平台回放为目的的提交 |
| 预算 | B2 = 30% × 36h ≈ **10.8h**。主线（P1 + A/B + 循环赛 + 提交）≈ B1 selfplay 实耗 + 1.5h；P2 / P1b / 机制诊断为**门控可选分支**（0.4 预算门 + 0.1 冻结线） |
| 硬约束 | ① **12/1 之后不改代码**——P2 代码修改必须在此之前完成并冻结；② 提交窗口 10:00–18:00、每天 ≤3 次尝试、累计成功 ≤20；③ 本阶段提交只为收集回放，排名动作留给最后阶段 |
| 上游文档 | [hbot_boxing_prelim_plan.md](hbot_boxing_prelim_plan.md) §6（P1/P2/P3）、§7.3、§7.4；[PHASE2_BASELINE.md](PHASE2_BASELINE.md)（分块/恢复/预算门机制沿用其修正版） |

**执行方式**：agent 经 `ssh hbot`；训练在 `tmux` 会话 `hbot`；命令块自带 `cd /mnt/workspace/HoloMotion && source env.sh` 前缀；训练用 `.venv-holoboxing/bin/python`，评测用 `conda run -n hbot-eval`。命名规范：`results/p1-selfplay[-rN]`、`boxing-p1.onnx`、`p1-validation.json`；P2/P1b 同理。

---

## 已核实的关键机制（出处见附录 D）

| 机制 | 行为 | 出处 |
|---|---|---|
| 新阶段续训 | `--policy-checkpoint <model_256.pt>` 从保存的 actor+归一化器开**新课程阶段**：重建 critics/优化器/探索（重置为 config 的 `init_noise_std`）/对手池；只校验 format、resources、policy_spec，**不比对奖励代码** → P2 改奖励权重后仍可用既有 actor 起训 | `on_policy_runner.py` `initialize_from_policy`；training README |
| updates 合法性 | `--updates ≤ 0` 直接报错 → 续块命令**必须**带 `if N<256` 守卫 | `on_policy_runner.py:478` |
| interval 数学 | 256 updates 内快照数：`256→1；64→4；32→8；16→16` | `opponent_pool.py:56-70`（推断 + PHASE1 机制实测 + PHASE2 复核） |
| 奖励权重位置 | 12 项权重硬编码在 `mujoco_rewards.py` `terms()`；配置级旋钮只有 `task_reward_weight 0.8 / style_reward_weight 0.2`（`training.json:41-43`）；实际生效配置写入**输出目录 `config.json`**（`train_mujoco.py:79`），`environment.json` 里没有 | 已核 |
| 代码修改与 resume | resume 的 identity 校验含代码哈希：同码内正常 resume；跨码改动需 `--allow-resume-code-change` | `runners/resume.py:314-340` |
| 评测确定性 | `match` 用固定 `initial_state`，无 seed 参数 → 同一有序对局可复现；A/B 双侧换序 + 核查局验证 | `evaluation/match.py:47` |
| 时延口径 | 本地验收唯一通过口径 = `chain_benchmark.passed == true`（即 `mean_ms ≤ 5`）；5–10ms **不算通过**，提交属例外路径（用户显式确认 + 平台实测并记录） | `chain_benchmark.py` |

---

## 阶段总览

| 阶段 | 内容 | 产出 | 预算 |
|---|---|---|---|
| 0 | 前置 + 实验设计定稿（归因口径、判定规则、冻结线、候选清单初始化） | 规则确认、`candidates.txt`、B2 起点记账 | 0h |
| 1 | P1 训练（interval=32，256 updates，从 B1 最终 actor 新阶段） | `p1-final-dir.txt` + `model_256.pt` | ≈B1 selfplay 实耗 |
| 2 | P1 验收 + 对手池快照复核（应 =8） | 记录 | 0h |
| 3 | A/B：核查局 + 双侧对局 + 判定 + `current-best.txt` | `winrate.md`、`current-best.txt` | ≤1h |
| 4 | （可选，门控）P2 奖励调整一轮 | `p2-*` + patch | ≈阶段 1 |
| 5 | （可选，门控）P1b（capacity=16/interval=16） | `p1b-*` | ≈阶段 1 |
| 6 | 本地循环赛（candidates.txt 全员）+ 最终候选登记 | 胜率表、`final-candidate.txt` | ≤0.5h |
| 7 | 平台提交 2–3 次【用户手动】 | `submissions.md`、平台回放分析 | 0h |
| 8 | 验收、记账、交接 | 验收表 | 余量 |

---

## 阶段 0：前置与实验设计（agent）

### 0.1 门槛清单

- PHASE2 验收表全绿且 §5 归档完成（见"前置"）；`results/b1-final-dir.txt` 存在且指向含 `model_256.pt` 的目录；
- `instance_snapshot.md` 记「B2 起点：日期 / 免费额度剩余」；
- 初始化候选清单：`echo results/boxing-b1.onnx > results/candidates.txt`（此后每导出一个新模型就 `echo results/boxing-xxx.onnx >> results/candidates.txt`，阶段 8 校验闭环）。

### 0.2 P1 参数与归因口径

| 参数 | B1 | P1 |
|---|---|---|
| `--opponent-refresh-updates` | 256 | **32** |
| `--opponent-pool-capacity` | 8 | 8（不变） |
| `--updates` / `--num-envs` / `--seed` / config | 256 / 同 / 42 / selfplay.json | 相同 |
| 起点 actor | `b1-warmup/model_32.pt`（SL 后仅 32 updates） | **B1 最终 `model_256.pt`** |
| 学习状态 | — | `--policy-checkpoint` **重建** critics/优化器/探索/对手池 |

**归因口径（重要）**：P1 **不是**单因子实验——相对 B1 同时存在"起点 actor 更强 + 新阶段重置 + interval=32"三个差异。本地 A/B 只回答**产品级问题：P1 产物是否弱于 B1**；不证明"8 个对手快照带来提升"。若需要机制层因果解释，有一个**可选诊断分支**（0.4 预算门）：`B1 最终 actor + interval=256` 同流程跑 256 updates 作对照——仅当要解释机制时才跑，不影响产品决策。

### 0.3 A/B 判定规则（先写死，防止事后挑规则）

- **核查局先行**：正式判定前，先跑一局与正式局完全相同输入的重复局（见 §3 命令），比对两局 `match.json` 是否完全一致：
  - 一致 → 确定性假设成立，按双侧 2 局制继续；
  - 不一致 → **停止判定**，回填本文档附录 A 与主方案附录 D（"match 确定性"条目），改为每侧 3 局按胜场判定（命令模板见 §3 末尾）。
- **【2026-10-25 实测回填】**：核查局已实际推翻确定性假设——`ab-p1-red` 与 `-red-repeat` 的 summary 巧合相同（0:0/TIME/零 hit），但 frames 层不等（物理浮点非确定性）。**本阶段所有 A/B 直接采用每侧 3 局制**，不再逐次重跑核查局；证据：`records/winrate.md` 与 `results/ab-p1-red*`。
- 双侧 2 局制判定：
  1. P1 两局全胜 → **采纳**；
  2. 两局全负 → **弃**（记录后不重试）；
  3. 1:1 → 比两局净胜分合计；再平 → 比 fall_count 合计（少者胜）；仍平 → **弃**（基线已验证，保守优先）。
- 可选分支（P2/P1b）的 A/B 对手 = `current-best.txt` 指向的模型，规则同上。

### 0.4 预算门控与冻结线（可计算）

```bash
# 远程/本地皆可，数据来源：instance_snapshot.md 与 records/b1-timings.md
剩余额度H=___        # 36 − 累计已耗
需保留H=2.5          # PHASE4 最终复跑与验证
分支成本H=___        # B1 selfplay 实耗（b1-timings.md）+ 0.5h A/B
```

- **分支开启条件（两条同时满足）**：`剩余额度H − 需保留H ≥ 分支成本H` **且** 今天不晚于 **11/24**（12/1 前一周）。
- 任一不满足 → 关闭全部可选分支（P2/P1b/机制诊断），只保主线与收尾。
- 主线自身的块前预算检查沿用 PHASE2 §0.3 两条硬门槛（块前核算、触顶即停）。

---

## 阶段 1：P1 训练（interval=32）

```bash
tmux new -s hbot 2>/dev/null || tmux attach -t hbot
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
B1_FINAL=$(cat results/b1-final-dir.txt)
ls "$B1_FINAL/model_256.pt" || { echo "B1 最终 checkpoint 缺失"; exit 1; }
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$B1_FINAL/model_256.pt" \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 --output results/p1-selfplay \
  --num-envs <B1 定档值> --updates 256 --seed 42 --device cuda:0 \
  --max-runtime-seconds 10800
```

**分块续跑（自带 `if N<256` 守卫——`--updates 0` 会被 `on_policy_runner.py:478` 拒绝）：**

```bash
N=$(ls results/p1-selfplay*/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
LATEST=$(ls -t results/p1-selfplay*/model_*.pt | head -1)
echo "N=$N LATEST=$LATEST"
if [ "$N" -lt 256 ]; then
  .venv-holoboxing/bin/python -m holoboxing.training.train_rl \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --amp-motions "$HOLOBOXING_DATA" \
    --resume "$LATEST" \
    --config holoboxing/training/configs/selfplay.json \
    --training-mode selfplay --opponent-pool-capacity 8 \
    --opponent-refresh-updates 32 --output results/p1-selfplay-r2 \
    --num-envs <B1 定档值> --updates $((256-N)) --seed 42 --device cuda:0 \
    --max-runtime-seconds 10800
fi
# 完成后登记（无论几块）：
P1_FINAL=$(ls -dt results/p1-selfplay* | head -1); echo "$P1_FINAL" | tee results/p1-final-dir.txt
```

- resume 与首块都必须保持 `--opponent-refresh-updates 32`（config 一致性）；至多 2 块。
- 监控沿用 PHASE2 §4.2（目录通配换成 `p1-selfplay*`）。
- 预期现象：新阶段重建 critics，**训练早期指标短暂劣化属正常**；持续 >50 updates 无恢复再排查。
- **验收**：`$P1_FINAL/model_256.pt`；`budget.stop_reason == "requested_progress_complete"`；`complete.json.updates_completed == 256`。

## 阶段 2：P1 快照复核（零成本，检验机制推断）

```bash
python - <<'EOF'
import torch, glob
d = open('results/p1-final-dir.txt').read().strip() + '/model_256.pt'
ck = torch.load(d, map_location='cpu', weights_only=False)
pool = ck['opponent_pool']
print('snapshots =', len(pool['snapshots']),
      'iterations =', [s['iteration'] for s in pool['snapshots']])
EOF
```

- **验收**：`snapshots == 8` 且 `iterations == [0,32,64,…,224]`。不符 → 回填主方案 §2.3/§6 P1/附录 D。

## 阶段 3：A/B 对战（核查局 → 正式双局 → 判定 → current-best）

```bash
cd /mnt/workspace/HoloMotion && source env.sh && source .venv-holoboxing/bin/activate
P1_FINAL=$(cat results/p1-final-dir.txt)
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$P1_FINAL/model_256.pt" --output results/boxing-p1.onnx
echo results/boxing-p1.onnx >> results/candidates.txt
conda run -n hbot-eval python -m holoboxing validate results/boxing-p1.onnx \
  --output results/p1-validation.json --benchmark

# ① 随机性核查局（输入与 ② 的第一局完全相同，仅换输出目录）
conda run -n hbot-eval python -m holoboxing match \
  results/boxing-p1.onnx results/boxing-b1.onnx --output-dir results/ab-p1-red-repeat

# ② 正式双侧两局
conda run -n hbot-eval python -m holoboxing match \
  results/boxing-p1.onnx results/boxing-b1.onnx --output-dir results/ab-p1-red
conda run -n hbot-eval python -m holoboxing match \
  results/boxing-b1.onnx results/boxing-p1.onnx --output-dir results/ab-p1-blue

# ③ 核查局比对 + 摘要
python - <<'EOF'
import json, glob
a = json.load(open('results/ab-p1-red/match.json'))
b = json.load(open('results/ab-p1-red-repeat/match.json'))
print('确定性: ' + ('一致' if a == b else '不同→停止判定，改每侧3局，回填附录A'))
for d in sorted(glob.glob('results/ab-p1-*')):
    if d.endswith('-repeat'): continue
    m = json.load(open(d + '/match.json'))
    print(d, {k: m[k] for k in list(m)[:10]})
EOF
```

- 若核查局不一致，改每侧 3 局：

```bash
for i in 1 2 3; do
  conda run -n hbot-eval python -m holoboxing match \
    results/boxing-p1.onnx results/boxing-b1.onnx --output-dir results/ab3-red-$i
  conda run -n hbot-eval python -m holoboxing match \
    results/boxing-b1.onnx results/boxing-p1.onnx --output-dir results/ab3-blue-$i
done
# 判定：每侧胜场多者胜；同分回到净胜分 → fall_count → 弃。
```

- **验收**：`p1-validation.json` 双条件通过（`status==PASS` 且 `chain_benchmark.passed==true`；5–10ms 灰区=不通过，例外须用户确认并注明）；判定结论写入 `records/winrate.md`。
- **判定后更新 current-best 指针**（P2/P1b 的起点依据）：

```bash
if <P1 采纳>; then cat results/p1-final-dir.txt > results/current-best.txt
else cat results/b1-final-dir.txt > results/current-best.txt; fi
ls "$(cat results/current-best.txt)/model_256.pt" || { echo "current-best 指针失效"; exit 1; }
```

## 阶段 4（可选，门控）：P2 奖励调整一轮

> 一次只改一个变量；两个级别二选一，默认推荐 a（不改码，无冻结线风险）。

**a) config 级（无代码修改）**：复制 `selfplay.json` 为 `selfplay-p2.json`，加字段（示例 `"task_reward_weight": 0.9, "style_reward_weight": 0.1`）。**生效确认**：先跑 1-update 烟雾（`--updates 1 --output results/p2-cfg-smoke`），读其**输出目录的 `config.json`**（`train_mujoco.py:79` 写入完整 TrainConfig）核对两个字段；`environment.json` 里没有配置，不要用它判读。该 run 的 resume 必须始终用同一 config 文件（config 进 resume 校验）。

**b) 代码级（单权重，走修改管理）**：编辑 `holoboxing/training/envs/mujoco_rewards.py` `terms()` 的一个权重（`offensive_hit_reward_punch` 50.0→60.0，:144-146；或 `defensive_hit_penalty_punch` -8.0→-12.0，:159-161）。流程：

1. 编辑代码；
2. **改完、commit 前**（远程执行）：`git diff > /tmp/p2-reward.patch`；
3. `git checkout -b p2-reward && git add -A && git commit -m "P2: offensive 50->60"`；
4. 本地 WSL：`scp hbot:/tmp/p2-reward.patch /home/czx/HBOT/records/ && sha256sum /home/czx/HBOT/records/p2-reward.patch`，哈希记入 `winrate.md`；
5. 12/1 前冻结；run 内再改码需 `--allow-resume-code-change` 并记录原因。

**P2 训练**（起点 = `current-best.txt`；已核实 `initialize_from_policy` 不校验奖励代码）：

```bash
BEST=$(cat results/current-best.txt)
ls "$BEST/model_256.pt" || { echo "current-best 指针失效"; exit 1; }
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$BEST/model_256.pt" \
  --config <selfplay.json 或 selfplay-p2.json> \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 --output results/p2-selfplay \
  --num-envs <B1 定档值> --updates 256 --seed 42 --device cuda:0 \
  --max-runtime-seconds 10800
# 分块续跑：同阶段 1 的 if N<256 模板（目录通配 p2-selfplay*，interval/config 保持一致）
# 完成后：登记 p2-final-dir.txt；导出 boxing-p2.onnx；>> candidates.txt；A/B 对 current-best；更新 current-best
```

- 验收同阶段 1/3；结论（含 patch 哈希，若走 b）写 `winrate.md`。

## 阶段 5（可选，门控）：P1b（capacity=16/interval=16）

```bash
BEST=$(cat results/current-best.txt)   # 应指向 P1（若 P1 被弃则本分支不应开启）
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$BEST/model_256.pt" \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 16 \
  --opponent-refresh-updates 16 --output results/p1b-selfplay \
  --num-envs <B1 定档值> --updates 256 --seed 42 --device cuda:0 \
  --max-runtime-seconds 10800
```

- 分块续跑同阶段 1 模板（`p1b-selfplay*`，`--opponent-pool-capacity 16 --opponent-refresh-updates 16` 保持）。
- 验收：`model_256.pt` 完整；快照复核应 =16（iterations 0,16,…,240）；导出 `boxing-p1b.onnx` → `candidates.txt`；A/B 对 current-best；判定后**更新 current-best**。
- 归因注记：P1b 再次经历新阶段重置，与 P1 的比较同为产品级结论。

## 阶段 6：本地循环赛与最终候选

```bash
cd /mnt/workspace/HoloMotion && source env.sh
MODELS=$(cat results/candidates.txt)   # 自动包含全部已导出候选（b1/p1[/p1b/p2]）
echo "$MODELS" | tr ' ' '\n'
i=0
for A in $MODELS; do for B in $MODELS; do
  [ "$A" = "$B" ] && continue
  i=$((i+1))
  conda run -n hbot-eval python -m holoboxing match "$A" "$B" \
    --output-dir results/rr-$i
done; done
python - <<'EOF'
import json, glob
for d in sorted(glob.glob('results/rr-*')):
    m = json.load(open(d + '/match.json'))
    print(d, {k: m[k] for k in list(m)[:8]})   # 人工判胜后填 winrate.md
EOF
```

- **胜率表**：`records/winrate.md` 逐局记比分/fall/KO 与胜者；循环赛自动覆盖"每个候选 vs B1"的双侧交手（检查单 #3 的证据来源）。
- **最终候选** = 循环赛第一；同分按"对基线两局净胜分"。登记（可执行版）：

```bash
CANDIDATE=results/boxing-p1.onnx        # 按循环赛冠军填
[ -f "$CANDIDATE" ] || { echo "候选缺失: $CANDIDATE"; exit 1; }
echo "$(realpath "$CANDIDATE") $(sha256sum "$CANDIDATE" | cut -d' ' -f1)" | tee results/final-candidate.txt
```

- 若冠军是 `boxing-b1.onnx`：最终候选 = 基线，阶段 7 不提交新模型（保留 B1 平台模型），结论记"改进未奏效，证据归档"。

## 阶段 7：平台提交 2–3 次【用户手动，agent 备料】

提交对象 = 最终候选 ONNX（非基线时）。检查单（按候选场景重写，不沿用基线版）：

| # | 项 | 标准（可测） |
|---|---|---|
| 1 | validate | `status==PASS` 且 `chain_benchmark.passed==true`（≤5ms；5–10 属例外路径：用户显式确认 + 平台实测，记录"灰区例外"） |
| 2 | 文件 | <50MB，sha256 已记录 |
| 3 | 候选证据 | 候选 vs B1 双侧对局（循环赛自动覆盖）结果已归档 `winrate.md` |
| 4 | 窗口 | 10:00–18:00 且距 18:00 ≥2h |
| 5 | 次数 | 当天尝试 <3，累计成功 <20 |
| 6 | 规则 | `#/details` 当天截图存档 |
| 7 | 画像 | 候选的 A/B 局无自身 KO、平均 fall_count ≤2/局 |

- 节奏：上午窗口；验证成功当天不再提交；总量 ≤3 次。
- **回放收集（提交的真正目的）**：平台对战公布后下载/查看回放，把对手风格（贴身/游走/重拳型）与我方失分模式记入 `records/platform_replays.md`。
- 规则提醒：平台用**最新验证成功**的模型；失败/验证中不替换旧模型。

## 阶段 8：验收、记账与交接

| 类别 | 检查项 | 通过标准 |
|---|---|---|
| P1 | 训练 + 快照复核 | `p1-final-dir.txt` 指向完整 `model_256.pt`；snapshots==8 且 iterations 正确（或已回填） |
| A/B | 核查局 + 判定 | 确定性核查已做；双侧对局结论已归档 `winrate.md` |
| 指针 | `current-best.txt` / `final-candidate.txt` | 均存在、指向的 checkpoint/onnx 存在、sha256 已记 |
| 可选分支 | 门控记录 | 每个分支的预算门/冻结线判定（开/不开 + 数值依据）已记录 |
| 候选闭环 | `candidates.txt` | 每个 `results/boxing-p*.onnx` 都出现在 `winrate.md`（防止漏赛） |
| 提交 | `submissions.md` | 每次提交的验证结果与回放分析归档 |
| 预算 | B2 对账 | 实耗 vs 10.8h 记入 `instance_snapshot.md` |

**回填点**：核查局推翻确定性假设 → 本文 0.3/附录 A + 主方案附录 D；P1 快照数与推断不符 → 主方案 §2.3/§6 P1/附录 D；`--policy-checkpoint` 跨奖励代码行为与核实不符 → 附录 D。

**下一步**：[PHASE4 冻结与最终提交]（最终复跑/换 seed、12/2 前完成最终验证、12/3 只确认），届时另立文档。

---

## 附录 A：A/B 与循环赛方法论（确定性评测下）

1. `match` 固定 `initial_state`（`match.py:47`），无 seed → 同一有序对局完全可复现；信息量来自模型差异与红蓝侧序，不是重复次数。
2. 方法论保险 = **核查局**：正式判定前用相同输入跑重复局，`match.json` 全等则确定性成立；不等则整体切换为每侧 3 局胜场制并回填。**【2026-10-25 实测：不等（frames 层物理浮点非确定性；summary 层巧合相同）→ 本阶段已整体切换为 3 局制】**。
3. 每个有序对打 1 局；A/B 双侧 2 局；循环赛全覆盖有序对（含 vs B1）。
4. 真实对手多样性来自平台对战（对手未知）——这是阶段 7 收集回放的原因；本地 A/B 只回答"有没有变弱"。
5. 归因边界：所有"从更强 actor 起新阶段"的比较都是**产品级**结论；机制归因需要同起点对照（0.2 诊断分支）。

## 附录 B：代码修改管理规范（P2 代码级）

1. 顺序：改码 → `git diff > /tmp/p2-reward.patch`（**改后、commit 前**，改前 diff 为空）→ 分支 + commit → `scp` 回本地 `records/` → 记 sha256；
2. patch 哈希记入 `winrate.md`，保证 A/B 结论可归因到具体 diff；
3. 12/1 冻结线后不再改码；必须 hotfix 则视为新分支重新走 A/B；
4. resume 与代码一致性：同码内正常 resume；跨码用 `--allow-resume-code-change` 并另行记录；
5. `--policy-checkpoint` 起训不受奖励代码变化影响（已核实），但 resources/policy_spec 必须一致——不要动 assets 与先验结构。

## 附录 C：高频故障（本阶段专属）

| 现象 | 处理 |
|---|---|
| `Policy fine-tuning format/resources/frozen representation differs` | 依次核对：checkpoint 是否本管线产物、`$HOLOBOXING_ASSETS` 与 B1 同源、先验结构未被改动 |
| `--updates 0` 报错 | 续块命令缺 `if N<256` 守卫；N=256 时直接登记 final-dir |
| P1 训练早期奖励下滑 | 新阶段重置 critics 属预期；>50 updates 不恢复再查 metrics |
| P1 两局大比分负且频繁倒地 | 查 p1 `metrics.jsonl` 是否后期崩坏；不自动重试（预算），记录后回退基线 |
| config 级 P2 疑似未生效 | 读该 run **输出目录 `config.json`**（非 environment.json）核对字段 |
| 核查局结果不一致 | 停止判定 → 每侧 3 局制 → 回填附录 A 与主方案 |
| A/B 1:1 且净胜分同 | 逐级 tiebreak（净胜分→fall→弃） |

## 附录 D：机制出处（2026-10-24 复核）

| 机制 | 文件:行 |
|---|---|
| `--policy-checkpoint` 新阶段语义与校验项 | `training/rsl_rl/runners/on_policy_runner.py`（`initialize_from_policy`）、`training/README.md:35-37` |
| `--updates ≤ 0` 拒绝 | `training/rsl_rl/runners/on_policy_runner.py:478` |
| 对手池刷新与快照数 | `training/opponent_pool.py:56-70` |
| 奖励权重硬编码位置 | `training/envs/mujoco_rewards.py:144-146`、`:159-161`；`configs/training.json:41-43` |
| 输出目录 config.json | `training/scripts/train_mujoco.py:79` |
| resume 代码一致性校验 | `training/rsl_rl/runners/resume.py:314-340` |
| match 固定初态/无 seed | `evaluation/match.py:47` |
| chain_benchmark 5ms 判据 | `evaluation/runtime/chain_benchmark.py`（`threshold_ms=5.0`） |
