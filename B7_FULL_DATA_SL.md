# HBOT B7 执行规格：全量 InterHuman × RoboStriker × 单种子 prior/SL（无 RL）

> **本文件是 [B8_ROBOSTRIKER_REPRO.md](B8_ROBOSTRIKER_REPRO.md) 的数据/初始化前置，不是论文方法复现的主规格。** 论文 Stage I→II→III(a)→III(b) 的执行以 B8 主规格为准。
>
> 版本 v1.0 · 2026-10-25
> 上游：[B6_DATA_SCALE.md](B6_DATA_SCALE.md)（B6 v1，数据规模方向）、[PHASE2_BASELINE.md](PHASE2_BASELINE.md)、[records/winrate.md](records/winrate.md)
> 定位：把现有数据从「3 个 InterHuman clip + 4 组 RoboStriker」扩到「InterHuman 全量 7,810 + RoboStriker 全量 43 组」，用全量数据训练 prior，用拳击相关子集训练 SL，**本轮不做 RL**，SL 训练完直接验收。
> 原则：数据只增不覆盖；全量数据优先保证 prior；SL 只喂拳击相关片段，避免通用交互动作稀释；官方 test 永远留出；所有结论以 native 对战与 validate 为准。

---

## 0. 本轮决策（2026-10-25，用户确认）

| 决策 | 选择 |
|---|---|
| InterHuman 全量范围 | **全量 7,810 进 prior；拳击相关子集进 SL/验收** |
| 官方 test | **保留不训练**（test 只做最终验收） |
| 训练规模 | **1 个 prior seed + 1 个 SL seed，SL 19 epochs** |
| 4096 模型 | **顺带验收 B5a2 `model_4096.pt`**（export/validate/native） |
| RL | 本轮**不跑**；SL-only 直接验收 |

非目标：
- 不改观测/动作接口（1097 → 225）、不改冻结低层控制器、不改比赛规则、不改 reward。
- 不再扩 selfplay、不做多种子 RL、不做课程/奖励实验（B5/B6 v1 已有证据归档）。
- 不把 InterHuman 原始数据拷出实例或再分发。

---

## 1. 现状与关键事实

| 项 | 现状（2026-10-25 23:30） |
|---|---|
| B5a2 4096 | 已跑完 4096/4096，`results/b5a2-selfplay/model_4096.pt`；export+validate **PASS / 4.84ms**；native 4 局进行中 |
| RoboStriker | 本地全量 86 npz / 43 组；已推到实例 `raw/RoboStriker_full`（86/86） |
| InterHuman | 实例原只有 3 clip；官方 Google Drive 在本机与实例均不可达 |
| InterHuman 全量来源 | HF 镜像 `Hang19/Interhuman` 的 `motions.zip`（561MB）+ `annots.zip`（3.4MB）；**逐文件 md5+size 与官方 listing 完全一致：motions 7,810/7,810、annots 7,897/7,897** |
| 官方 split | train 6,021 / val 580 / test 1,177 / 未标注 32（沿用现有 `raw/InterHuman/split`） |
| 额度 | 总剩余约 14h；当前场约 5h（19:18 开机，8h 场约 03:18 结束），之后还有一场 8h |
| 实测 retarget | 热态 3 clip / 3,418 帧 = **26s**（≈190 帧/s，含 setup）；全量 2.27M 帧 ≈ **3.0–3.5h 串行** |
| 实测 prepare | B1 8 clip / ~9.2k sim 帧 ≈ 200s（≈46 sim steps/s，4 workers 口径） |
| B1 SL 事实 | SL 只用 InterHuman；B1 训练集只有 **1 个 train clip / 430 样本**，validation 6,172 样本；19 epochs 仅 36s |
| B1 prior 事实 | 290 epochs / 83.6s；validation position 7.3cm / rotation 0.265rad |

**结构性结论**：B1 的 SL 几乎没有训练数据。本轮全量拳击子集把 SL 训练样本从 430 提到约 80 万，是数量级变化；这是「SL 能不能打出拳」的第一优先变量。

---

## 2. 数据获取与校验

### 2.1 镜像来源与许可

- 官方 InterHuman 仅 Google Drive；实例/本地均不可达。使用 HF 镜像 `Hang19/Interhuman`：
  - `motions.zip` 561MB（7,810 个 `motions/<id>.pkl`）
  - `annots.zip` 3.4MB（7,897 个 `annots/<id>.txt`）
- **来源证明**：两包逐文件与官方 `source-listing.json` 的 md5+size 全部一致（motions 7,810/7,810、annots 7,897/7,897）。
- **许可**：镜像标 MIT 不代表原始数据许可；InterHuman 原始许可仍为非商业研究、禁止再分发。只保留官方 `LICENSE.md`，数据只存在实例内，不拷出、不入 Git。

### 2.2 传输与落盘（已完成/进行中）

本地把 `motions.zip` 按 128MB 分成 5 块，5 路并行 rsync 到实例；`annots.zip` 单独传。传输中断用 `--partial` 续传。

```bash
# 实例
cd /mnt/workspace/HoloMotion/holoboxing/data/raw/InterHuman_full
cat motions.zip.part-* > motions.zip
sha256sum motions.zip        # 期望 1e499699a6a3685cc9277045a4e2b01d5e7d3cfe682c6545c7175daa2595d06d
unzip -q motions.zip -d .
unzip -q annots.zip -d .
rm -f motions.zip annots.zip motions.zip.part-*
cp ../InterHuman/source-listing.json .
cp -r ../InterHuman/split .
cp ../InterHuman/LICENSE.md .
find motions -name '*.pkl' | wc -l    # 7,810
find annots -name '*.txt' | wc -l     # 7,897
```

### 2.3 生成官方格式 receipt（`download.json`）

`retarget_dataset` 读取 `download.json`（`holoboxing_download_v1`、`selection_mode=clip_ids`），并对每个 motion 和 split 文件做 sha256 校验。用脚本生成：

- 逐文件用官方 listing 校验 md5+size；
- 计算 sha256；
- `requested_clip_ids` = 全部 7,810 个 id；
- `source_motion_files=7810`、`selected_motion_files=7810`、`bounded_selection=false`；
- 包含 motions、annots、split、LICENSE 的完整文件清单。

**验收**：`downloaded_records(raw/InterHuman_full)` 返回 7,810 条，split 分布为 train 6,021 / validation 580 / test 1,177 / unknown 32；无缺失 annotation。

---

## 3. 子集划分与用途

| 用途 | 数据 | 规模 |
|---|---|---:|
| prior 全量 | RoboStriker 86 npz + InterHuman 全量 7,810 | 43 组 + 7,810 clip |
| SL / 验收 | InterHuman 拳击相关子集 | **1,411 clip** |
| prior 通用补充 | 其余 InterHuman | 6,399 clip |
| 留出 | 官方 test（含 4033） | 1,177 clip（不训练） |

**拳击子集定义**（annots 文本关键词并集；12 个 annotation 无对应 motion，已剔除）：

```
punch | strike | box(ing) | fight | spar | dodge | evade | avoid | slip |
jab | hook | uppercut | cross
```

- 计数：base(punch/strike/box/fight) 970；加 dodge 1,077；加拳型词 1,423；**与 motions 求交后 1,411**。
- 覆盖直拳/摆拳/勾拳/上勾拳/闪避；不含踢、摔、拥抱等通用交互。
- 子集按官方 split 划分；官方 test 片段从 SL 中排除。

---

## 4. 全量 retarget（4 次迭代，分块可续）

### 4.1 为什么分块

- 官方 `retarget_dataset` 串行处理，首次编译约 80s，之后约 190 帧/s；全量 7,810 约 3–3.5h。
- 单进程一旦中断（会话到界/失败）只能整批重跑；分块后每块独立输出，失败只重跑该块。
- 两个子集分开：拳击子集优先完成，供 prepare/SL 使用；通用子集服务 prior。

### 4.2 分块

- `boxing`：1,411 clip（4 iterations）。
- `generic-0..3`：6,399 clip 均分 4 块（4 iterations）。
- 每块目录结构：`motions/<id>.pkl`（硬链接，不复制数据）、`split/`、`LICENSE.md`、块级 `download.json`（只含本块文件 + split）。
- 4 次迭代统一保留，与 B1/现有 3 clip 一致；实测 2 次迭代平均关节差 0.2–0.9°，但个别帧最大 33°，不降质量。

### 4.3 命令与合并

```bash
cd /mnt/workspace/HoloMotion && source env.sh
# 拳击块
.venv-holoboxing-retarget/bin/python -m holoboxing.data retarget \
  --source holoboxing/data/raw/InterHuman_chunks/boxing \
  --output holoboxing/data/converted/InterHuman_b7_boxing \
  --smpl-model /mnt/workspace/licenses/neutral_smpl.pkl \
  --assets "$HOLOBOXING_ASSETS" --iterations 4 --device cuda:0

# 通用块（4 块，按显存/GPU 实测决定并行度）
for K in 0 1 2 3; do
  .venv-holoboxing-retarget/bin/python -m holoboxing.data retarget \
    --source holoboxing/data/raw/InterHuman_chunks/generic-$K \
    --output holoboxing/data/converted/InterHuman_b7_generic-$K \
    --smpl-model /mnt/workspace/licenses/neutral_smpl.pkl \
    --assets "$HOLOBOXING_ASSETS" --iterations 4 --device cuda:0 &
done; wait
```

合并脚本把各块 `motions.json` 的 records 拼成 `converted/InterHuman_full`，npz 硬链接过去；顶层 `motions.json` 的 `source_download_sha256` 指向全量 receipt，`bounded_conversion=false`。

### 4.4 验收

- 每块无 `FAILED.json`；records 数 = 块内 clip 数。
- 合并后 7,810 条，npz 文件名唯一；`source_group` 唯一。
- `pair_scaling.maximum_pair_xy_error_m` 全部 < 1e-4。
- **确定性抽查**：新转换的 `interhuman-1124.npz`、`1843.npz`、`4033.npz` 与现有 `converted/InterHuman/` 同名文件逐数组一致（同 SMPL、同 assets、同 4 iterations）。
- 任一失败：记录 clip id 与错误，不阻塞其余块；失败 clip 不进 prior/SL。

---

## 5. 全量 prior

### 5.1 数据

```bash
cd /mnt/workspace/HoloMotion && source env.sh
REF=results/b7-reference
.venv-holoboxing/bin/python -m holoboxing.training.prepare_prior \
  --source holoboxing/data/raw/RoboStriker_full \
  --source holoboxing/data/converted/InterHuman_full \
  --assets "$HOLOBOXING_ASSETS" --output "$REF" \
  --validation-fraction 0.1 --test-fraction 0.1 --seed 42
```

- 官方 split 优先：InterHuman train/val/test 按官方；RoboStriker 无官方 split，按组随机 10%/10%。
- 期望 reference 样本：InterHuman ~4.5M + RoboStriker ~0.9M（以 manifest 实测为准）。
- 验收：manifest 无重复 clip；`excluded_groups` 含官方 test；`held_out_test_groups` 非空。

### 5.2 训练

```bash
.venv-holoboxing/bin/python -m holoboxing.training.train_prior \
  --data results/b7-reference --output results/b7-prior \
  --hidden-dims 512 256 --batch-size 1024 --epochs 1000 \
  --seed 43 --device cuda:0 --max-runtime-seconds 7200
```

- 单 seed；2h 硬上限；validation plateau 早停。
- **门控**：validation position/rotation 不差于 B1（7.3cm / 0.265rad）才进入 SL；变差先查 split/retarget/对齐，不硬跑。
- 记录 `epochs_completed`、best/last、stop_reason、每 epoch 时间。

---

## 6. 拳击 prepared 数据集与 SL

### 6.1 prepared（只含拳击子集）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
OUT=holoboxing/data/processed/b7-boxing
.venv-holoboxing/bin/python -m holoboxing.data prepare \
  --source holoboxing/data/converted/InterHuman_b7_boxing \
  --output "$OUT" --assets "$HOLOBOXING_ASSETS" \
  --test-fraction 0 --workers 4 --seed 42
.venv-holoboxing/bin/python -m holoboxing.data verify \
  --output "$OUT" --assets "$HOLOBOXING_ASSETS"
```

- `--test-fraction 0`：官方 test 已足够；不再随机割 test，尽量保留训练样本。
- **验收**：`COMPLETE.json.status=PASS`；无 `FAILED.json`/`.partial`；prepared 中无 `split=test`；记录 `supervised_samples`、train/val clip 数。
- `prepare` 无断点续跑；若会话到界被 kill，换新输出目录重跑（旧目录保留为 `.failed-YYYYMMDD-HHMM`）。
- 资源：4 workers 先试；若 GPU 被 generic retarget 拖慢到聚合吞吐下降，则串行执行。

### 6.2 SL（19 epochs，1 seed）

```bash
cd /mnt/workspace/HoloMotion && source env-b7.sh   # HOLOBOXING_DATA=$OUT/prepared
.venv-holoboxing/bin/python -m holoboxing.training.train_sl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --data "$HOLOBOXING_DATA" --prior-checkpoint results/b7-prior/best.pt \
  --config holoboxing/training/configs/sample.json \
  --output results/b7-sl --epochs 19 --batch-size 512 \
  --seed 43 --device cuda:0 --max-runtime-seconds 10800
```

- 只训练 train split；validation 每 epoch 评估。
- **门控**：19/19 完成；记录 train/validation position/rotation/loss；与 B1 SL（train 5.2cm / val 20.1cm）对比。
- 若 19 epochs 内 validation 连续 3 epoch 变差：记录，不早停（本轮只跑一个 seed，需要完整曲线）。
- 3h 硬上限；超时按最后 checkpoint 继续（`--resume`）。

---

## 7. 导出与验收（SL-only）

```bash
cd /mnt/workspace/HoloMotion && source env-b7.sh
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint results/b7-sl/epoch_000019.pt \
  --output results/boxing-b7-sl-only.onnx

/mnt/workspace/miniforge3/envs/hbot-eval/bin/python -m holoboxing validate \
  results/boxing-b7-sl-only.onnx \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --output results/b7-sl-validation.json --benchmark
```

- validate 双门槛：`status=PASS` 且 `chain_benchmark.passed=true`（mean_ms ≤ 5.0）。
- native 4 局（对官方原生 RoboStriker，`/mnt/workspace/model_3000.pt`）：

```bash
export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
for SIDE in 0 1; do for R in a b; do
  /mnt/workspace/miniforge3/envs/hbot-eval/bin/python -m holoboxing.training.evaluate_native \
    --candidate results/boxing-b7-sl-only.onnx \
    --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" --opponent-sha256 "$ROBOSTRIKER_SHA256" \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --candidate-side "$SIDE" --output "results/native-b7-sl-side$SIDE-$R" --device cuda:0
done; done
```

- 归因/技术指标/视频：`deployment/fall_attrib.py` + `deployment/render_b5a_game.sh`；技术指标从 `match.json` 的 `dof_pos` 计算右肘 ROM、收拳-出拳循环、HIT 前回撤比例。
- **对照**：
  - B1 SL-only（`results/boxing-b1-sl.onnx`）同口径 native 4 局 —— 直接回答「扩数据有没有用」；
  - B5a2 4096（RL impulse）native 4 局 —— 当前最强 RL 对照；
  - H2b 已有 native 双 KO 记录。
- 判定（只记录，不替换 `final-candidate`，除非满足全部条件）：
  1. validate 双门槛；
  2. 我方 HIT > 对方 HIT；
  3. 败局 = 0；4. 我方倒地 = 0；
  5. 技术门控（有收拳/出拳循环，不是固定伸臂）；
  6. 不劣于 H2b/B5a2。
- 预期管理：SL-only 不含交互/时机学习，**可能仍 0 HIT/只摆姿势**；本轮目的是测「全量数据 + 拳击子集」对模仿质量的上限，不用胜负替代 KPI。

---

## 8. B5a2 4096 验收

- 已导出 `results/boxing-b5a2-4096.onnx`；validate **PASS / 4.84ms**。
- native 4 局 + `fall_attrib.py` + 视频，与 B7 SL、H2b 同口径。
- 产物：`results/native-b5a2-4096-side{0,1}-{a,b}/`、`records/b5a2-native-attrib.txt`。
- 不因「是 RL 模型」直接替换 final-candidate；仍按 §7 判定条件。

---

## 9. 资源预算与时间线

| 阶段 | 估时 | 说明 |
|---|---:|---|
| 传输 + unzip + receipt | 0.5–1.0h | 已进行中；5 路分块 |
| 拳击 retarget 1,411 | 0.7–1.0h | ≈190 帧/s |
| 通用 retarget 6,399 | 2.0–2.8h | 4 块；按 GPU 实测并行 |
| 拳击 prepare 1,411 | 1.5–3.0h | 4 workers；与 generic retarget 重叠需实测 |
| 全量 prior | 1.5–2.0h | 2h 硬上限 |
| SL 19 epochs | 1.5–2.0h | 拳击子集约 80 万样本 |
| B7 验收（validate+native 4 局+归因+视频） | 0.7–1.0h | 可与 B1 SL 对照并行 |
| B5a2 native 4 局 | 0.5h | 已开始 |
| **合计** | **9–12h** | 剩余约 14h；留 2–3h 缓冲 |

**并行策略**：
- 传输期间跑 B5a2 native（已执行）——网络/GPU 不冲突。
- 拳击 retarget 完成后，generic retarget 与拳击 prepare 可尝试重叠；每 10 分钟看 GPU util 与各自 update/帧吞吐，**聚合吞吐下降就改串行**。
- prior 与 prepare 尽量不同时跑（都是 GPU 重负载）；prior 等 generic retarget + prepare 完成后再启动。

**砍单顺序**（从下往上）：
1. B5a2/B7 的视频渲染与额外重复局；
2. B7 native 每侧 b 局（保留 a 局，每侧 1 局）；
3. SL epochs 19 → 12（保留完整曲线记录）；
4. B1 SL-only 对照（若时间不够，只做 B7）；
5. **不可砍**：全量 receipt/校验、拳击 retarget、拳击 prepare/verify、全量 prior、SL 19 epochs（或至少 12）、validate 双门槛、至少 2 局 native、归因/技术指标。

---

## 10. 风险与处置

| 风险 | 处置 |
|---|---|
| 镜像来源/许可 | 已逐文件 md5 对齐官方；保留官方 LICENSE；只在实例内使用，不拷出/不入 Git |
| 传输慢/中断 | 128MB 分块 + `--partial` 续传；单块失败只重传该块 |
| retarget 单 clip 失败（gender/短片段/数据异常） | 分块隔离；`FAILED.json` 记录；跳过该 clip，不阻塞其余 |
| prepare 被会话到界 kill | 无断点；换新目录重跑；优先在会话前半段启动 |
| prior validation 变差 | 停止，先查 split/retarget；不进入 SL |
| SL 19 epochs 超 3h | `--max-runtime-seconds` 保存最后 checkpoint；下一场 `--resume` |
| SL-only 仍 0 HIT | 预期结果之一；记录技术指标与视频，按证据决定是否再开 RL |
| GPU 并行互相拖慢 | 以聚合吞吐为准；不满足就串行；不靠「多进程看起来忙」 |
| 磁盘 | prepared H5 ~数 GB、reference ~数 GB；NAS 余量 993T，无风险 |

---

## 11. 产物清单

| 类型 | 路径 |
|---|---|
| 全量 raw | `holoboxing/data/raw/InterHuman_full/{motions,annots,split,LICENSE.md,source-listing.json,download.json}` |
| 分块 raw | `holoboxing/data/raw/InterHuman_chunks/{boxing,generic-0..3}` |
| 转换 | `converted/InterHuman_b7_boxing`、`converted/InterHuman_b7_generic-0..3`、合并 `converted/InterHuman_full` |
| prepared | `processed/b7-boxing/{COMPLETE.json,sources.json,prepared/}` |
| prior | `results/b7-reference/`、`results/b7-prior/` |
| SL | `results/b7-sl/` |
| 导出/校验 | `results/boxing-b7-sl-only.onnx`、`results/b7-sl-validation.json` |
| native | `results/native-b7-sl-side{0,1}-{a,b}/`、`results/native-b1-sl-side{0,1}-{a,b}/` |
| B5a2 | `results/boxing-b5a2-4096.onnx`、`results/b5a2-validation.json`、`results/native-b5a2-4096-side{0,1}-{a,b}/` |
| 归因/技术 | `records/b7-native-attrib.txt`、`records/b7-technique.json`、视频 |
| 记账 | `records/winrate.md` 追加 B7 段；`deployment/instance_snapshot.md` 记额度/耗时 |

所有新目录独立，不覆盖 B1/B5/B6 产物；`final-candidate` 仅在满足 §7 判定时更新。

---

## 12. 执行顺序（复现清单）

1. 传输 `motions.zip`/`annots.zip` → 实例；合并分块、unzip；校验 7,810/7,897；生成 receipt；`downloaded_records` 返回 7,810。
2. 生成拳击/通用分块目录与块级 receipt（motions 用硬链接）。
3. 拳击 retarget 1,411（4 iter）→ `converted/InterHuman_b7_boxing`；确定性抽查 1124/1843/4033。
4. 拳击 prepare + verify → `processed/b7-boxing`；`COMPLETE.json=PASS`；建 `env-b7.sh`。
5. 通用 retarget 4 块 → 合并 `converted/InterHuman_full`（7,810）；QC。
6. `prepare_prior`（RoboStriker_full + InterHuman_full）→ `results/b7-reference`。
7. `train_prior` 单 seed（2h cap）→ 门控对比 B1（7.3cm/0.265rad）。
8. `train_sl` 19 epochs 单 seed → 记录 train/val position/rotation；对比 B1（5.2cm/20.1cm）。
9. 导出 SL-only ONNX → validate 双门槛 → native 4 局 + 归因 + 技术指标 + 视频。
10. 同口径跑 B1 SL-only native 对照；汇总 B5a2 4096、H2b 对照。
11. 按 §7 判定；更新 `records/winrate.md`、`deployment/instance_snapshot.md`；不满足则保留 H2b，B7 归档。

---

## 13. 验收表

| 类别 | 检查项 | 通过标准 |
|---|---|---|
| 数据 | InterHuman 全量 | motions 7,810 + annots 7,897，md5 全对，receipt 可被 `downloaded_records` 读取 |
| 数据 | RoboStriker 全量 | 86 npz / 43 组，sha256 清单一致 |
| 数据 | retarget | 合并 7,810 条，无 FAILED，pair XY 误差 < 1e-4，3 clip 抽查一致 |
| 数据 | prepared | `COMPLETE.json=PASS`；无 test 进入训练；记录 samples/clips |
| 训练 | prior | 单 seed；validation 不差于 B1（7.3cm/0.265rad）；有 best/last |
| 训练 | SL | 19/19；train/val position/rotation 有完整曲线；与 B1 对比已记录 |
| 模型 | validate | `status=PASS` 且 `chain_benchmark.passed=true` |
| 对战 | native | 每侧 ≥1 局；记录 HIT/倒地/KO/胜负 + 归因 + 技术指标 + 视频 |
| 对照 | B1 SL-only / B5a2 / H2b | 同口径结果已记录 |
| 记账 | 额度/耗时/失败 | `instance_snapshot.md` + `winrate.md` 追加 |

---

## 14. 执行记录（2026-10-26）

- **InterHuman 全量已就位**：`raw/InterHuman_full` 7,810 motions + 7,897 annots；`motions.zip` sha256 `1e499699…` 与本地一致；逐文件 md5 全对；`download.json` 可被 `downloaded_records` 读取，split 为 train 6,022 / val 580 / test 1,177 / unknown 31。
- **RoboStriker 全量**：`raw/RoboStriker_full` 86 npz / 43 组。
- **retarget 调度修正**：3 路并行把 A10 打到 94% util，聚合吞吐反而低于单路；改为**单路先跑拳击 1,411**（约 35–45min），通用 6,399 留到下一场串行/双路。已中止的 3 个部分输出目录改名 `.aborted-*` 保留，不复用。
- **B5a2 4096**：native 4 局 0 命中；side0 0:11 双败、side1 1000:1 双胜（对手自倒）。不替换 final-candidate；记录见 `records/winrate.md`。
- **prepare 第一次中止（02:05）**：官方实现遇到 `Source produced zero measured SL rows; refusing to drop its AMP motion` 会整批 abort；已完成 **715/1,238** clip，compiled 包自动恢复。
- **补丁**：`records.py::_prepare_group` 改为逐 clip try/except，只跳过 zero-row clip，其他错误仍抛出；patch 留档 `records/b7-prepare-skip.patch`。补跑 523 个缺失 clip，已识别两个坏 clip：`interhuman-1630`、`interhuman-2829`（均 0 条有效 SL 行），跳过并从 prepared/sources 中排除。
- **补跑吞吐**：约 18.5 帧/s（明显低于主批的 62 帧/s，剩余 clip 更长/更难）；预计本场 03:10 左右完成，若跨会话则保留已完成部分、下一场补剩余。
- **合并约束**：`PreparedDataset.source_metadata` 要求 `sources.json` 的 records 与 `manifest.json` 的 records **逐条同序、sha256/group/split 一致**，且 `COMPLETE.json` 的 `sources_sha256` 匹配 `sources.json`；坏 clip 必须从两者同时排除，不能只删 manifest。合并脚本：`/mnt/workspace/logs/b7_merge_prepared.py`（有 missing manifest 时）或 `b7_merge_snapshot.py`（只有部分 clip 文件时）。
- **后续**：合并 → verify → 若仍有未完成 clip，下一场补跑 → 通用 retarget → 合并 7,810 → `prepare_prior` → prior → SL（19 epochs）→ 验收 → B8a（LS-NFSP；平均策略与 best-response 都导出）。
