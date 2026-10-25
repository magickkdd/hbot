# H-Bot 拳击赛道初赛方案

**目标：2026-12-03 初赛截止前进入排行榜前 16，晋级决赛。**
**范围：只覆盖初赛；决赛（RDK S600）等官方规则公布后再补。**
**版本：v1.2 / 2026-10-24（v1.1 按官方代码与赛事页核查修订；v1.2 补本地时延判据 5ms 口径）**

---

## 0. 结论摘要

- 本赛题不是从零训练人形机器人，而是训练一个 **50Hz 的高层拳击规划器**，输出骨盆、双腕、双踝的未来位姿，交给固定的 HoloMotion V2 下层控制器执行。
- 官方参考实现已经开源了 RoboStriker 风格的三阶段管线：**动作先验 → 监督学习 → PPO + AMP + 自对弈**。训练使用 MuJoCo Warp，不需要 Isaac Sim，24GB 单卡可以从 16 envs 起步、逐步扩到 128。
- 你的算力足以完成初赛，但必须**先基线、后改进**：36 小时 NVIDIA 额度优先保证一次完整基线 + 一次 RL 改进 + 最终复跑，不做早期架构创新。
- 稳定进前 16 的关键不是模型多大，而是：**接口一次跑通、基线稳定、每天按规则提交、最后一天有验证成功的最强模型**。

---

## 1. 赛题与规则

### 1.1 两层控制与接口

| 项目 | 内容 |
|---|---|
| 高层策略 | 参赛队训练并提交；输入 `obs[1,1097]`，输出 `target_pose[1,225]` |
| 下层控制器 | HoloMotion V2，官方固定，不可替换或改参数 |
| 控制频率 | 高层 50Hz（每 20ms 一次决策），物理 200Hz |
| 机器人 | Unitree G1，29 自由度，戴拳套 |
| 场地 | 5m × 5m 拳台，单回合最多 60 秒 |

**输入 `obs[1,1097]`：**

| 切片 | 含义 |
|---|---|
| `[0:1080]` | 10 帧自身状态，每帧 108 维，间隔 20ms，由旧到新 |
| `[1080:1095]` | 对手 5 个身体点在己方当前骨盆坐标系下的位置 |
| `[1095]` | 剩余回合时间（秒） |
| `[1096]` | 己方比分减对方比分 |

每帧 108 维自身状态：骨盆角速度 3 + 5 个身体点位姿 45 + 5 个点线速度 15 + 上一帧位姿 45。

**输出 `target_pose[1,225]`：**

- reshape 为 `[1,5,5,9]`：5 个未来时刻 × 5 个身体点 × 9 维位姿。
- 未来时刻：当前观测后 20、40、60、80、100ms。
- 身体点：骨盆、左腕、右腕、左踝、右踝。
- 9 维：3 维位置 + 6 维旋转（旋转矩阵第 1、3 列，即 X、Z 轴；Y = Z × X）。
- 所有目标都在己方当前骨盆坐标系下。

### 1.2 关键规则

| 规则 | 判定 |
|---|---|
| 有效命中 | 手部接触对方头部或躯干；脚击不计分 |
| 冲量要求 | 每只手最近 20ms 内法向冲量 ≥ 5 N·s，得 1 分 |
| 重复命中 | 左右手分别计算；同一只手两次计分至少间隔 200ms；不需要先分离 |
| 倒地 | 骨盆或躯干原点距拳台地面高度**严格低于 0.30m**；仅倾斜不算 |
| 击倒 KO | 连续倒地满 3 秒，对手 +1000 分并结束比赛 |
| 胜负 | 时间结束比分高者胜；同分平局；双方同时 KO 平局 |

### 1.3 排行榜与提交限制

- 排行榜使用 Elo 等级分，**每天重置为 1500**，按当日对战结果更新。
- **最后一日排行榜前 16 名晋级决赛**。
- Elo 变化不受 KO 奖励分或单场比分差影响；也就是说，**1:0 赢和 1000:0 赢在排名上没有区别**。
- 每天提交窗口：北京时间 **10:00–18:00**；上传和验证成功都必须在 18:00 前完成。
- 每队每天最多尝试 **3 次**；一次验证成功后当天不再接受新提交；累计成功提交上限 **20 次**。
- 平台使用队伍**最新验证成功**的模型；失败或仍在验证的新模型不会替换之前成功的模型。
- 本地对战不更新官方排行榜；官方成绩以平台组织的比赛为准。

**规则来源（2026-10-24 核查）：**

- 场地 60 秒、5m × 5m、戴拳套：赛事详情页 + 官方资源 `arena/boxing.xml`（base 2.5×2.5，墙在 ±2.5）+ `29dof/*boxing_gloves*.xml`。
- 命中/冲量/冷却/倒地/KO：官方仓库 `holoboxing/README.md` 的 “Match rules”。
- Elo 每日重置 1500、最后一日前 16、提交窗口 10:00–18:00、每天 3 次、累计 20 次成功：赛事详情页 `#/details`；**官方仓库里没有这些内容**。
- 页面没有标注规则版本号或更新时间；提交前必须重新核对并截图存档，避免平台改规则后方案失效。

### 1.4 赛程

| 日期 | 事项 |
|---|---|
| 2026-10-22 | 报名开启，初赛赛题发布 |
| 2026-12-03 | 初赛提交截止 |
| 2026-12-10 | 公布初赛结果，决赛开启 |
| 2027-01-07 | 决赛提交截止 |
| 2027-01-14 | 最终获奖结果公布 |

> 决赛在 RDK S600 设备上做闭环评测是官方宣传口径，最终以决赛公告为准。本方案暂不展开。

---

## 2. 前沿调研

### 2.1 技术地图

| 方向 | 代表工作 | 核心结论 | 对本赛题的用法 |
|---|---|---|---|
| 全身运动跟踪 | HoloMotion-1/V2、BeyondMimic、KungfuBot、SONIC | 大规模动作数据 + 强跟踪控制器，能做出高动态动作 | 下层控制器已固定；重点是用高质量拳击动作做参考 |
| 潜空间运动先验 | PULSE、CALM、ASE | 把高维动作压到低维潜空间，高层只输出潜变量 | 官方 32 维潜变量 + 解码器就是这类思路 |
| 竞技自对弈 | RoboStriker、SMPLOlympics、NFSP/PSRO/AlphaStar League | 直接在关节空间自对弈会失衡；在潜空间自对弈更稳 | 官方参考实现的核心；也是拉开差距的关键 |
| 奖励设计 | AMP、RoboStriker 命中/防守/终结奖励、SMPLOlympics | 强动作先验 + 简单任务奖励，往往比堆复杂奖励更有效 | 奖励要贴合裁判规则，不能只看比分 |
| 端侧部署 | ONNX Runtime、RDK S600、ASAP | 决策模型很小，真正瓶颈是下层控制器和算子兼容性 | 初赛保证 ONNX 结构合法、完整链平均 ≤10ms（RTX 5090） |

### 2.2 RoboStriker：当前最直接的 SOTA

RoboStriker（arXiv:2608.16195，官方教程引用版）把问题拆成三层：

1. **动作库**：先在人体动作数据上训练一个动作跟踪器，学会拳击、闪避、步伐等物理可行动作。
2. **潜空间**：把动作蒸馏到一个单位超球面的潜空间，高层策略只在这个低维空间里探索。这样就不会探索到“关节乱动、身体崩溃”的区域。
3. **策略进化**：先用 AMP 动作先验热身，再做潜空间神经虚拟自对弈（LS-NFSP），让双方在潜空间里共同进化。

论文的核心结论：

- 直接在原始关节空间做多智能体探索，会长期停留在“保持平衡”阶段，学不出战术。
- 潜空间自对弈能显著减少摔倒，打出进攻、防守、反击等战术行为。
- 该框架在仿真中胜率和打击效率优于原始动作空间和 AMP 基线，并完成了真机验证。

### 2.3 官方参考实现

官方把 RoboStriker 简化成一条可复现管线：

| 阶段 | 内容 | 产出 |
|---|---|---|
| Stage 1 | 动作先验：32 维单位向量 + 解码器 `32→256→512→225` | `best.pt` |
| Stage 2 | 监督学习：Actor `[256,128]`，从自身 + 对手观测预测未来位姿 | `epoch_000019.pt` |
| Stage 3 | PPO + AMP：先对站立对手热身 32 updates，再做 256 updates 自对弈 | `model_256.pt` |
| 导出 | 把 Actor、归一化、冻结解码器导出为 ONNX | `boxing.onnx` |

参考配置：128 个并行环境，对手池容量 8，`--opponent-refresh-updates 256`。
**实际行为**：`iteration` 从 0 开始，`next_refresh = iteration + interval`；在 256 updates 预算下，只有 iteration 0 生成 1 个初始快照，iteration 1–255 不会刷新，池子实际不会填满。

### 2.4 其他可借鉴工作

- **PULSE**：用变分信息瓶颈学习通用人形动作潜空间，并用本体感觉先验提高采样效率。它是“32 维潜变量 + 解码器”思路的来源之一。
- **SMPLOlympics**：已经实现拳击等竞技体育环境，结论是“强动作先验 + 简单奖励 + 自对弈”就能产生类人行为，不需要过度设计奖励。
- **BeyondMimic**：高质量动作跟踪 + 引导扩散，能做出跳跃、旋转、冲刺等高动态动作。可以用来生成更丰富的拳击/闪避参考动作。
- **KungfuBot**：面向武术等高动态技能的双层优化跟踪，强调动作物理可行性和自适应跟踪容差。对“拳击动作参考”的质量有借鉴意义。
- **ASAP**：真机部署时用残差动作模型补偿仿真与现实差异。初赛不需要，决赛若上真机再考虑。
- **NFSP / PSRO / AlphaStar League**：竞技自对弈的通用思路；核心是保留历史对手、按难度采样、避免只克制定一个对手。

### 2.5 对我们最重要的 5 条启示

1. **潜空间是物理可行性的护栏**：不要直接输出 225 维关节/位姿从头探索，先用动作先验约束。
2. **动作先验比奖励工程更重要**：先把参考动作和跟踪质量做好，再谈奖励权重。
3. **对手池是稳定性的关键**：基线在 256 updates 内实际只有 1 个初始对手快照，等于固定对手；要让历史对手真正生效，必须缩短刷新周期。
4. **官方 Actor 丢掉了时间和比分**：代码里只取 `obs[:1095]`，没有用最后两维；当前 63 维状态已包含线速度，旧 9 帧和上一帧位姿被置零。可改进的是时间/比分和 10 帧历史。
5. **Elo 只看胜负**：训练目标应该是提高胜率，而不是刷命中数或净胜分。

---

## 3. 总体方案

### 3.1 路线选择

**主线：官方参考管线为基线，只做“RL 阶段可复用”的局部改进。**

理由：

- 官方管线已经打通数据、训练、导出、校验、对战、回放，最省时间。
- 你的 NVIDIA 额度只有 36 小时，不适合从零搭新框架或多轮架构搜索。
- 初赛目标是前 16，不是冠军；稳定性优先于新颖性。

**改进优先级：**

| 优先级 | 改进 | 需要重训 | 风险 | 说明 |
|---|---|---|---|---|
| P0 | 官方三阶段基线 | 全部 | 低 | 必须完成 |
| P1 | 缩短对手刷新周期（先 8/32，再 16/16）+ PFSP（可选） | 仅 RL | 低 | 基线 256 updates 实际只有 1 个快照 |
| P2 | 奖励权重对齐裁判 | 仅 RL | 中 | 可和 P1 合并 |
| P3 | 使用时间/比分 + 10 帧历史 | SL + RL | 中高 | 第二次完整训练再做；线速度已在用 |
| P4 | 更大潜空间/残差头/新网络 | 全部 | 高 | 本地循环赛对基线胜率 ≥60%（≥20 场）且剩余 GPU ≥10h 才考虑 |

**不做清单：**

- 不做直接 225 维从头 RL。
- 不做大 Transformer/视觉模型。
- 不引入 Isaac Sim 或额外仿真栈。
- 不为了“看起来先进”而改动作空间。
- 不在 12 月 1 日之后改代码。

### 3.2 三阶段训练

```
Stage 1  动作先验      32 维潜空间 + 解码器
Stage 2  监督学习      Actor 从观测预测未来目标位姿
Stage 3  PPO + AMP    站立热身 → 自对弈
导出      ONNX         只含 Actor + 归一化 + 冻结解码器
```

### 3.3 模型与接口

- 基线保持官方结构：Actor 输入 1097，实际取 1095 维因果观测；输出 32 维单位向量；解码器输出 225。
- 导出时保留观测归一化和冻结解码器；ONNX 必须满足 `[1,1097] → [1,225]`、float32、单文件 <50,000,000 字节。
- 不使用随机算子、Dropout；允许符号 batch 维，但必须支持 batch=1；TopK 只能沿最后一维、该维宽度已知，K 为 1–128 常量。

---

## 4. 数据方案

### 4.1 数据来源

| 数据 | 用途 | 许可 |
|---|---|---|
| RoboStriker 单机器人 G1 轨迹 | 动作先验、AMP 风格奖励 | 官方提供，随仓库下载 |
| InterHuman 双人动作 | 动作先验训练 + 风格参考 + SL 的配对观测与目标 | 非商业研究许可，禁止再分发 |
| SMPL 中性人体模型 | 把人体动作重定向到 G1 | 学术非商业许可，需注册申请 |

### 4.2 为什么需要 InterHuman/SMPL

- 这不是比赛官方要求，而是**官方参考训练管线里 SL 阶段的需要**。
- SL 要让高层策略学会“看到自己 + 对手的观测，预测未来目标位姿”。InterHuman 提供双人配对动作，才能构造对手观测和监督目标。
- RoboStriker 数据是单机器人轨迹，只能用于动作先验和 AMP，不能单独跑官方 SL。
- 如果不申请：代码里有 `--prior-checkpoint` 入口，可从 `best.pt` 直接初始化 latent RL（官方教程未文档化）。注意 `train_rl` 的 `--amp-motions` **必填**，需要先用 RoboStriker-only 跑 `holoboxing.data prepare` 生成 AMP 参考——这条组合路径官方同样未文档化，必须先做 4 updates 烟雾测试再依赖；且训练更不稳、更吃 GPU，不适合“稳定进前 16”的目标。优先拿许可。

### 4.3 申请与下载

1. **SMPL**：在 `https://smpl.is.tue.mpg.de/` 注册，下载 neutral SMPL 模型；仅用于非商业研究，不可再分发。
2. **InterHuman**：按 `https://github.com/tr3e/InterGen#interhuman-dataset` 的许可下载；仅非商业使用。
3. **RoboStriker**：官方脚本下载，先下 4 组做烟雾测试，正式训练再扩。

### 4.4 数据准备流程

```bash
# 1) 下载 RoboStriker 轨迹（先少量）
python -m holoboxing.data download robostriker \
  --output holoboxing/data/raw/RoboStriker --limit-groups 4 --workers 2

# 2) 下载 InterHuman 片段
python -m holoboxing.data download interhuman \
  --output holoboxing/data/raw/InterHuman \
  --clip-id 1124 --clip-id 1843 --clip-id 4033 --workers 2

# 3) 重定向（单独环境，需要 SMPL neutral 模型）
python3.11 -m venv .venv-holoboxing-retarget
.venv-holoboxing-retarget/bin/python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-holoboxing-retarget/bin/python -m pip install \
  -r holoboxing/data/retarget-requirements.txt
.venv-holoboxing-retarget/bin/python -m holoboxing.data retarget \
  --source holoboxing/data/raw/InterHuman \
  --output holoboxing/data/converted/InterHuman \
  --smpl-model /path/to/licensed/neutral-smpl.pkl \
  --assets "$HOLOBOXING_ASSETS" --device cuda:0 --iterations 4

# 4) 合并并生成 1097 观测 / 225 目标
python -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/combined \
  --assets "$HOLOBOXING_ASSETS" --test-fraction 0.1 --workers 2
python -m holoboxing.data verify \
  --output holoboxing/data/processed/combined --assets "$HOLOBOXING_ASSETS"

export HOLOBOXING_DATA="$PWD/holoboxing/data/processed/combined/prepared"
```

### 4.5 数据策略

- 第一轮：官方示例规模（4 组 RoboStriker + 3 个 InterHuman 片段），先跑通全链路。
- 第二轮：扩大 RoboStriker 组数和 InterHuman 片段数，优先覆盖直拳、摆拳、闪避、步伐、贴身。
- 保持官方 split，验证集不参与训练和统计。
- 数据资产和检查点一起记录哈希，避免复现错位。

---

## 5. 训练方案

### 5.1 Stage 1：动作先验

```bash
python -m holoboxing.training.prepare_prior \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --assets "$HOLOBOXING_ASSETS" --output results/boxing-reference

python -m holoboxing.training.train_prior \
  --data results/boxing-reference --output results/boxing-prior \
  --hidden-dims 512 256 --batch-size 1024 --epochs 1000 \
  --seed 43 --device cuda:0
```

- 编码器把 225 维参考目标压成 32 维单位向量；解码器 `32→256→512→225`。
- 只用训练集动作和训练集统计；早停看验证重构误差。
- 产出 `results/boxing-prior/best.pt`。

### 5.2 Stage 2：监督学习

```bash
python -m holoboxing.training.train_sl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --data "$HOLOBOXING_DATA" --prior-checkpoint results/boxing-prior/best.pt \
  --config holoboxing/training/configs/sample.json \
  --output results/boxing-sl --epochs 19 --batch-size 512 \
  --seed 43 --device cuda:0
```

- Actor `[256,128]`，输入 1095 维因果观测，输出 32 维潜变量。
- 产出 `results/boxing-sl/epoch_000019.pt`。
- **注意**：官方 Actor 不使用最后两维（时间、比分），并把旧 9 帧和上一帧位姿置零；当前帧 63 维包含角速度 3 + 位姿 45 + 线速度 15，线速度已经在用。后续改进点是时间/比分和 10 帧历史，不是线速度。

### 5.3 Stage 3：热身 + 自对弈

```bash
# 先小后大：烟雾测试可把 --num-envs 128 改为 16、--updates 32 改为 4；
# 确认显存和对手池刷新后再用 128。
# 站立对手热身
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --supervised-checkpoint results/boxing-sl/epoch_000019.pt \
  --config holoboxing/training/configs/sample.json \
  --training-mode standing --output results/boxing-warmup \
  --num-envs 128 --updates 32 --seed 42 --device cuda:0

# 自对弈
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint results/boxing-warmup/model_32.pt \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 256 --output results/boxing-rl \
  --num-envs 128 --updates 256 --seed 42 --device cuda:0
```

- 对手是冻结的历史策略，带自己的观测统计；只有学习方更新。
- 自对弈初始探索标准差 0.2；热身阶段 0.1。
- 产出 `results/boxing-rl/model_256.pt`。
- 第一次启动会编译 GPU 物理 kernel；以 `training-result.json` 的完成 update 数为准，不要只看进程退出。
- 用 `--resume` 续跑同一阶段；用 `--policy-checkpoint` 开新阶段。不要混用。
- 无 SL 的备用入口：`--prior-checkpoint results/boxing-prior/best.pt`（官方教程未文档化；先 4 updates 验证）。

### 5.4 导出与校验

```bash
python -m holoboxing.training.export \
  --checkpoint results/boxing-rl/model_256.pt --output results/boxing.onnx

./holoboxing/run.sh --validate results/boxing.onnx
./holoboxing/run.sh --validate results/boxing.onnx \
  --benchmark --output results/benchmark.json

./holoboxing/run.sh --match results/boxing.onnx results/boxing.onnx \
  --output results/first-match
./holoboxing/run.sh --view results/first-match/replay.hdf5
```

- `--validate` 检查大小、ONNX 结构、算子、张量名/形状、输出合法性、一次真实 CUDA 推理。
- `--benchmark` 测完整推理链：高层策略 + 冻结控制器，**平均** ≤10ms（≥100 FPS）；10 次预热 + 100 次测量；物理仿真和输出修复不计入；官方以 RTX 5090 云端实测为准。注意：本地验证代码以 `mean_ms ≤ 5ms` 作为 `chain_benchmark.passed` 判据（`threshold_ms=5.0`，严于 README 的 10ms），且本地模式 `status=PASS` **不含**时延判定；验收必须看 `passed` 字段，5–10ms 灰区以平台云端为准。
- `--match` 生成 `match.json` 和 `replay.hdf5`，用回放看命中、倒地、KO。

### 5.5 算力预算

| 资源 | 用途 | 判断 |
|---|---|---|
| ModelScope NVIDIA 24G / 36h | 主力训练 + 本地校验 | 先 16 envs 测显存，再扩到 64/128；优先使用 |
| AMD 192G / 40h | 不建议跑 RL | `mujoco-warp` 依赖 CUDA；最多试 ROCm 下的 SL/prior，不作为主力 |
| Colab/Kaggle | 备用 | 可跑 SL/prior 小实验；不能跑 Docker 本地校验 |
| 直接 Python 评测 / 租用 Docker GPU | 校验与对战 | ModelScope 不支持 Docker 时优先直接 Python；装不起来再租 1–2 小时 |

**36 小时额度分批：**

| 批次 | 占比 | 内容 |
|---|---|---|
| B0 | 10% | 环境、依赖、烟雾测试、ONNX 导出/校验 |
| B1 | 40% | 完整基线：prior + SL + warmup + self-play |
| B2 | 30% | 一次改进：优先缩短刷新周期/奖励；P3 观测增强仅在 P1/P2 提前完成且额度富余时考虑，不占用 B3 |
| B3 | 20% | 最终复跑、不同 seed、缓冲 |

原则：

- 每批都保存 checkpoint 和 `training-result.json`。
- 用 `--max-runtime-seconds` 控制墙钟时间，避免额度被一次跑飞。
- 先 16 envs 测显存和速度，再上 128；不要直接 512。
- 如果 B1 超预算，砍 B2，保基线。

---

## 6. 改进方案

### P1：缩短对手刷新周期（首选；扩容与 PFSP 为第二步）

**基线真实行为：**

- `OpponentPool.prepare()` 里 `next_refresh = iteration + interval`，`iteration` 从 0 开始。
- 官方 `--updates 256 --opponent-refresh-updates 256`：只在 iteration 0 生成 1 个初始快照；iteration 1–255 不刷新，容量 8 从未填满。
- 所以基线实际是“对 1 个初始策略自对弈”，不是“对 8 个历史策略”。
- 256 updates 内的大致快照数：interval 256 → 1；64 → 4；32 → 8；16 → 16。

**做什么：**

- 第一轮保持 `capacity=8`、`interval=32`：256 updates 内正好 8 个快照，能把池子填满。
- 第一轮稳定后，再考虑 `capacity=16`、`interval=16`；不要一开始就上 32 容量。
- 按难度/胜率采样（PFSP）；保留不同风格。PFSP 不是配置项：基线 `OpponentPool` 用 `torch.randint` 均匀采样，要做 PFSP 需要改 `opponent_pool.py`；第一轮先只改 interval/capacity。
- 先跑 `--updates 4 --opponent-refresh-updates 2 --num-envs 16` 的烟雾测试（从 warmup checkpoint 用 `--training-mode selfplay --policy-checkpoint` 续跑），打印 `opponent_pool` 快照数，确认刷新语义后再正式训练。

**为什么：**

- 基线只有 1 个初始对手，多样性不足，容易过拟合。
- 缩短 interval 是让 P1 生效的前提；保持/延长 256 会让池子始终只有 1 个对手。
- 这是 RL-only 改进，可直接从基线 actor 续跑，不需要重跑 SL。

**风险：**

- 池子越大，每个 update 的有效学习信号可能变稀；先保持容量 8。
- 刷新太频繁会增加重置和训练波动；先从 interval=32 起，按结果再降到 16。
- 训练池只放本管线冻结快照；原生 RoboStriker 只用于 `evaluate_native` 评测（见 7.2），不要塞进训练池。

### P2：奖励权重对齐裁判（可与 P1 合并）

**基线奖励项：完整表和权重见附录 B。** 学习方 robot_A 不使用 `balance`/`stand_still`（它们只属于站立假人 robot_B）；每项奖励在环境中还会乘以 0.02 步长。

**建议调整（一次只改 1–2 项）：**

- 提高 `offensive_hit_reward_punch`（50.0 → 60–70），或降低 `facing_reward`/`vel_reward` 的 shaping；一次只改一项。
- 上调 `defensive_hit_penalty_punch`（-8.0 → -12.0），而不是新增“被命中惩罚”。
- 倒地：基线已有 `terminate_reward`（0.3，0.4m 高度阈值）做持续高度塑形；如果要加 KO/倒地惩罚，应新增一个明确的终结项，避免和 `terminate_reward` 重复计罚。
- 如果环境支持，加入小权重的“本回合胜负”终结奖励；不支持就只改权重，不要大改环境。
- 所有调整都用同一 seed、同一对手池做 A/B；注意 0.02 缩放。

### P3：使用时间、比分和 10 帧历史（可选）

**现状：**

- 官方 Actor 只取 `obs[:1095]`，丢掉最后两维（剩余时间、比分差）。
- 当前帧 63 维（角速度 3 + 位姿 45 + 线速度 15）+ 对手 15 维；旧 9 帧和上一帧位姿被置零。
- 线速度已经在用，不要再把“启用线速度”列为改进。

**可以做什么：**

- 把 `obs[1095:1097]`（时间、比分）接进 MLP。
- 用 1D CNN 或小 MLP 处理 10 帧历史（含线速度序列），而不是只取当前帧。
- 不引入 GRU/LSTM：ONNX 接口只有 `obs`，没有 recurrent state 输入输出。

**代价：**

- 需要重跑 SL + RL，旧 checkpoint 不兼容；这是第二次完整训练，优先级低于 P1/P2。

### P4：动作空间/网络结构（不建议）

- 更大潜空间（64/128）、残差头、直接 225 输出、Transformer 等，表达力可能更强，但会破坏官方先验，训练不稳且吃预算。
- 初赛阶段不做；只有满足“本地循环赛对基线胜率 ≥60%（至少 20 场）且剩余 GPU 额度 ≥10h”才考虑小改。

---

## 7. 评测与提交

### 7.1 本地校验：ModelScope DSW 不支持 Docker

**机器分工（2026-10-24 确认）：**

- 本地 WSL：RTX 3050 Laptop 4GB，无 Docker，Python 3.14——只用于看资料、改代码，跑不了训练/评测。
- ModelScope DSW（24GB NVIDIA，36h）：训练 + 直接 Python 评测（本节路径）。
- 租用 Docker GPU（RunPod/AutoDL 等，1–2 小时）：仅在需要官方 `run.sh`/Docker 镜像校验或看回放时使用。

已确认：ModelScope Notebook 基于阿里云 PAI-DSW；**标准实例（非灵骏资源组）不支持在容器内再运行 Docker**。`docker version` 能显示客户端，但 `/var/run/docker.sock` 没有 daemon，所以 `docker run` 一定失败。不要尝试手动 `dockerd`，容器没有权限，起不来。

先做最后一个检查：

```bash
ls -l /var/docker/proxy/docker-proxy.sock

# 如果存在，设置代理 socket 后重试
export DOCKER_HOST=unix:///var/docker/proxy/docker-proxy.sock
docker version
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

如果这个 socket 不存在，直接放弃 Docker。

**无 Docker 的官方评测路径（推荐）**

`run.sh` 只是 Docker 包装，真正的入口是 `python -m holoboxing`。可以在 DSW 的 Python 环境里直接跑：

```bash
cd /mnt/workspace/HoloMotion   # DSW 上克隆的仓库；本地只读副本在 /home/czx/HBOT/official/HoloMotion
export HOLOBOXING_ASSETS="$PWD/holoboxing/assets"
export HOLOBOXING_LOWER="$PWD/models/holomotion-v2"

# 1) 创建 Python 3.11 环境（DSW 里用 conda）
conda create -n hbot-eval python=3.11 -y
conda activate hbot-eval

# 2) 安装官方评测运行时依赖
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r holoboxing/evaluation/docker/requirements.in
pip install -e .

# 3) 生成机器人模型与编译观测（第一次需要）
python -m holoboxing.evaluation.robot_models --assets "$HOLOBOXING_ASSETS"
python -m holoboxing.evaluation.runtime.observation_export \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --output holoboxing/evaluation/runtime/compiled

# 4) 直接校验 / 对战
python -m holoboxing validate results/boxing.onnx \
  --output results/validation.json --benchmark
python -m holoboxing match results/boxing.onnx results/opponent.onnx \
  --output-dir results/match
```

说明：

- `requirements.in` 是官方 Docker 镜像的运行时依赖；它没有 torch，但源码回退路径 `ReferenceSchema` 需要 torch 和 `holomotion`，所以单独装 torch。
- 如果 `observation_export` 成功，后续走编译后端，不再依赖 torch 的源码回退。
- 官方 `run.sh` 额外加了 `--network none`、`--read-only`；本地直接跑没有这些限制，不影响校验结果。
- 回放 `view` 需要前端 `viewer/dist`，本地没有；先看 `match.json`，或租 Docker GPU 看回放。
- 如果这套依赖在 DSW 上装不起来，退回租 RunPod/AutoDL 跑官方 Docker 镜像 1–2 小时。
- 只靠平台验证是最后手段：每天 3 次尝试、累计 20 次成功提交，试错成本高。

### 7.2 本地对战与回放

```bash
# 先导出两个 checkpoint（官方 exporter 支持 SL checkpoint 直接导出）
python -m holoboxing.training.export \
  --checkpoint results/boxing-sl/epoch_000019.pt --output results/boxing-sl.onnx
python -m holoboxing.training.export \
  --checkpoint results/boxing-rl/model_256.pt --output results/boxing-rl.onnx

# 同一模型互打
./holoboxing/run.sh --match results/boxing.onnx results/boxing.onnx \
  --output results/match-a

# 两个检查点互打
./holoboxing/run.sh --match results/boxing-sl.onnx results/boxing-rl.onnx \
  --output results/match-b

# 回放
./holoboxing/run.sh --view results/match-b/replay.hdf5
# 浏览器打开 http://localhost:8080
```

以上 `run.sh` 是 Docker 路径。在 ModelScope 无 Docker 时，把 `run.sh --match A B --output DIR` 换成 `python -m holoboxing match A B --output-dir DIR`；`--view` 需要 `viewer/dist`，本地没有，先看 `match.json`。

**看什么：**

- 是否频繁倒地；倒地是脚滑、目标不可达，还是被击中。
- 是否只贴脸不出拳；出拳是否满足 5N·s 冲量。
- 左右手是否交替；是否有 200ms 冷却内的无效重复。
- 领先/落后时行为是否合理；最后 10 秒是否守得住。
- 对手一贴身就倒，还是能护头/后撤。

**本地循环赛：**

- 至少 3 个模型：baseline、P1、P2。
- 每个配对打多场，记录胜率、平均比分、倒地率、KO 率。
- 原生 RoboStriker 只作为 `evaluate_native` 评测对手，**不能加入训练对手池**；模型需自行从 [RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview) 下载 `model_3000.pt` 并保留许可。

```bash
export ROBOSTRIKER_CHECKPOINT=/absolute/path/to/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
python -m holoboxing.training.evaluate_native \
  --candidate results/boxing.onnx \
  --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
  --opponent-sha256 "$ROBOSTRIKER_SHA256" \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --candidate-side 0 --output results/native-side0 --device cuda:0
# 另一侧：--candidate-side 1 --output results/native-side1
```

### 7.3 平台提交节奏

| 时间 | 提交内容 | 目的 |
|---|---|---|
| 10 月底 | 可选冒烟提交 | 只验证上传、验证、对战流程，不追求成绩 |
| 11 月上旬 | 首个基线提交 | 建立基线，收集平台回放 |
| 11 月中下旬 | P1/P2 改进 | 观察胜率变化，保留尝试次数 |
| 12 月 1–2 日 | 最终模型 | 提前验证，避免截止日排队 |
| 12 月 3 日 | 不再提交新模型 | 只确认已有模型验证成功 |

**规则提醒：**

- 上传和验证成功都必须在当天 18:00 前完成；建议上午提交，留出排队和验证时间。
- 验证失败或超时会消耗尝试次数；平台故障按规则退还。
- 最新验证成功的模型才会参赛；失败/验证中的模型不会替换旧模型。
- 每天最多 3 次尝试；累计成功提交上限 20 次。**不要前几天把次数烧光。**

### 7.4 Elo 策略

- Elo 每天重置，**最后一天的排名决定晋级**；前期排名只用于观察。
- 但最后一天能参赛的前提是：模型在截止前已经验证成功。**最晚 12 月 2 日完成最终模型验证。**
- Elo 不看净胜分和 KO 奖励，所以：
  - 训练目标提高胜率，不追求大比分。
  - 领先时保守、落后时进攻；如果做 P3，把时间/比分接进策略。
  - 稳定不倒地比多打一拳更重要。
- 初赛对手未知，**鲁棒的全能策略 > 只克制约一个对手的偏科策略**。
- 中间日期的提交只为收集平台回放和验证流程，取最少次数；所有与排名相关的动作在最后一日前完成验证。不要在最后一天冒险提交未验证的新模型。

---

## 8. 时间表

| 时间 | 任务 | 产出 |
|---|---|---|
| 10/24–10/31 | Day 0 环境、Docker 检查、SMPL/InterHuman 申请、烟雾数据、prior + SL | 能导出 ONNX，能通过 `--validate`；可选冒烟提交 |
| 11/01–11/07 | 完整基线：warmup + self-play 256 updates；本地对战 | `model_256.pt`、首个基线提交 |
| 11/08–11/14 | 回放分析、奖励权重小调、第二次提交 | 基线胜率数据 |
| 11/15–11/21 | P1 刷新周期/容量（PFSP 可选）；A/B | P1 checkpoint、第三次提交 |
| 11/22–11/28 | P2 奖励；有余量再做 P3 观测增强；A/B | 最终候选 |
| 11/29–12/02 | 冻结、最终复跑、验证、提交 | 最终 ONNX + 验证成功 |
| 12/03 | 截止日，只确认 | 不提交新模型 |
| 12/10 | 查看初赛结果 | 前 16 晋级 |

**每周固定动作：**

- 周一：检查 GPU 额度、磁盘、平台提交次数；核对 `#/details` 规则并截图存档。
- 周三：跑本地循环赛，更新胜率表。
- 周五：决定是否提交新模型。
- 周末：回放分析 + 下一轮训练计划。

---

## 9. 风险与预案

| 风险 | 影响 | 预案 |
|---|---|---|
| ModelScope 不支持 Docker | 无法本地校验/对战 | 优先用 `python -m holoboxing` 直接在 DSW 评测；装不起来再租 1–2 小时 Docker GPU |
| NVIDIA 36h 不够 | 训练中断 | 保基线，砍改进；用 Colab/Kaggle 跑 SL/prior |
| AMD 192G 无法跑 Warp | 算力浪费 | 不依赖；最多试 ROCm 下的 SL/prior |
| SMPL/InterHuman 许可延迟 | SL 无法开始 | `--prior-checkpoint` 直接 RL（未文档化；还需 RoboStriker-only `prepare` 生成 `--amp-motions`，先 4 updates 烟雾测试）；优先拿许可 |
| 自对弈训练崩/不收敛 | 没有可用模型 | 回到 baseline checkpoint；降低对手池/探索噪声 |
| 模型倒地或不出拳 | 验证通过但比赛输 | 看回放，检查目标坐标系、目标尺度、奖励权重 |
| 完整链平均 >10ms | 平台验证失败 | 减小网络/潜空间；检查 ONNX 算子；用 `--benchmark` 按官方口径测 |
| 最后一天排队验证失败 | 无法参赛 | 12/2 前完成最终验证；保留旧的成功模型 |
| Elo 波动大 | 排名不稳 | 提交鲁棒模型；中间日只收集回放，最终日前完成验证 |
| 官网规则无版本号 | 规则变更无从察觉 | 每周核对 `#/details` 并截图存档 |

---

## 10. Day 0 清单

按顺序执行，任何一步失败先解决再往下。**本清单在 ModelScope 24GB 实例上执行**（本地 WSL 是 4GB RTX 3050，仅用于看资料/改代码；官方资料离线副本已在 `official/`，控制器已下载并校验）：

1. **确认 GPU 和磁盘**
   ```bash
   nvidia-smi
   python3.11 -V
   df -h .
   ```
2. **确认 Docker（ModelScope 标准实例通常不支持）**
   ```bash
   ls -l /var/docker/proxy/docker-proxy.sock
   # 只有 socket 存在时才设置：
   export DOCKER_HOST=unix:///var/docker/proxy/docker-proxy.sock
   docker version
   docker run --rm --gpus all \
     nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```
   没有 socket 就直接走 7.1 的 `python -m holoboxing` 直接评测。
3. **克隆仓库**
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone \
     https://github.com/HorizonRobotics/HoloMotion.git
   cd HoloMotion
   ```
4. **创建训练环境**
   ```bash
   python3.11 -m venv .venv-holoboxing
   source .venv-holoboxing/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.9.1 \
     --index-url https://download.pytorch.org/whl/cu128
   python -m pip install -r holoboxing/training/requirements.txt
   python -m pip check
   python -c 'import torch; assert torch.cuda.is_available(); \
     print(torch.cuda.get_device_name(0))'
   ```
5. **下载资源和固定控制器**
   ```bash
   git submodule update --init thirdparties/HoloMotion_assets
   git -C thirdparties/HoloMotion_assets lfs install --local
   git -C thirdparties/HoloMotion_assets lfs pull \
     --include="holoboxing/**" --exclude=""
   git -C thirdparties/HoloMotion_assets lfs checkout holoboxing
   export HOLOBOXING_ASSETS="$PWD/holoboxing/assets"
   python -m holoboxing assets --assets "$HOLOBOXING_ASSETS"
   python -m holoboxing download --directory models/holomotion-v2
   export HOLOBOXING_LOWER="$PWD/models/holomotion-v2"
   ```
6. **烟雾数据**
   ```bash
   python -m holoboxing.data download robostriker \
     --output holoboxing/data/raw/RoboStriker \
     --limit-groups 4 --workers 2
   ```
7. **申请 SMPL/InterHuman 许可**（与训练并行，不要等）
8. **跑官方示例的 prior + SL**，确认能导出 ONNX
9. **跑 `--validate`**，记录结果
10. **可选冒烟提交**：只验证上传/验证流程，不追求成绩

---

## 附录 A：观测与输出字段速查

**obs 切片：**

- `[0:1080]`：10 × 108 自身状态。
- `[1080:1095]`：对手 5 点位置。
- `[1095:1096]`：剩余时间。
- `[1096:1097]`：比分差。

**单帧 108 维：**

- `[0:3]`：骨盆角速度（采样时刻 body frame，rad/s）。
- `[3:48]`：5 点当前位姿（当前 heading frame）。
- `[48:63]`：5 点线速度（当前 heading frame，m/s）。
- `[63:108]`：5 点上一帧位姿（当前 heading frame）。

**target 225 维：**

- reshape `[1,5,5,9]`。
- 5 个未来时刻：20/40/60/80/100ms。
- 5 个点：pelvis、left_wrist_yaw_link、right_wrist_yaw_link、left_ankle_roll_link、right_ankle_roll_link。
- 9 维：`px,py,pz,Xx,Xy,Xz,Zx,Zy,Zz`；X/Z 是旋转矩阵第 1/3 列，Y = Z × X。

---

## 附录 B：官方参考奖励项

| 奖励项 | 权重 | 作用 |
|---|---|---|
| `offensive_hit_reward_punch` | 50.0 | 有效出拳命中 |
| `defensive_hit_penalty_punch` | -8.0 | 被有效命中 |
| `complete_punch_reward` | 0.05 | 完整出拳 |
| `active_hit_shaping_reward` | 1.5 | 主动攻击 |
| `strike_reward` | 0.3 | 打击力差 |
| `facing_reward` | 1.2 | 朝向对手 |
| `vel_reward` | 0.5 | 向对手移动 |
| `terminate_reward` | 0.3 | 保持高度（0.4m 阈值） |
| `joint_limit` | -2.0 | 关节限位 |
| `action_rate_l2` | -0.05 | 动作平滑 |
| `foot_slip_penalty` | -0.05 | 脚底打滑 |
| `energy_saving_penalty` | -1e-5 | 近距离节能 |

说明：以上为学习方 robot_A 的奖励项；robot_B（站立假人）另有 `stand_still` 1.0、`balance` 1.0。环境每步实际奖励 = 权重 × 项值 × 0.02。

---

## 附录 C：参考链接

- 赛事详情：https://hbot-competition.horizon.auto/#/details
- 官方拳击教程（HoloMotion 仓库）：https://github.com/HorizonRobotics/HoloMotion/tree/master/holoboxing
- HoloMotion 文档：https://horizonrobotics.github.io/robot_lab/holomotion/
- HoloMotion-1 技术报告：https://arxiv.org/abs/2605.15336
- RoboStriker（官方教程引用版，2026-08）：https://arxiv.org/abs/2608.16195
- RoboStriker（2026-01 早期版，Hierarchical Decision-Making）：https://arxiv.org/abs/2601.22517
- PULSE：https://arxiv.org/abs/2310.04582
- SMPLOlympics：https://arxiv.org/abs/2407.00187
- BeyondMimic：https://arxiv.org/abs/2508.08241
- KungfuBot：https://arxiv.org/abs/2506.12851
- ASAP：https://arxiv.org/abs/2502.01143
- InterHuman：https://github.com/tr3e/InterGen#interhuman-dataset
- SMPL：https://smpl.is.tue.mpg.de/

> arXiv 链接已于 2026-10-24 逐条核验；PULSE、BeyondMimic、KungfuBot、ASAP 编号均正确。

---

## 附录 D：核查记录（2026-10-24）

| 项目 | 结论 | 出处 |
|---|---|---|
| 对手池刷新 | interval=256、updates=256 时只有 1 个初始快照 | `training/opponent_pool.py:56-70`、`rsl_rl/runners/on_policy_runner.py:359-364` |
| 学习方线速度 | 当前 63 维包含线速度 15，未被 mask | `training/spherical_actor.py:52-67` |
| balance 奖励 | 仅 robot_B；学习方无 balance | `training/envs/mujoco_rewards.py:37-56` |
| 奖励缩放 | 每项乘 0.02 | `training/envs/mujoco_env.py:595-597` |
| 原生 RoboStriker | 仅 `evaluate_native` 评测适配器，不在训练池 | `training/README.md:42-70` |
| prior-only RL | `--prior-checkpoint` 入口存在，教程未文档化 | `training/train_rl.py:40-44`、`scripts/train_mujoco.py:139-140` |
| 场地 | 5m×5m（base 2.5×2.5，墙 ±2.5），戴拳套 | `assets/arena/boxing.xml:13-17`、`assets/29dof/*boxing_gloves*.xml` |
| 比赛时长/命中/倒地 | 60s、5N·s/20ms、200ms、0.30m、KO 3s | `holoboxing/README.md` Match rules |
| Elo/提交限制 | 官网详情页有；官方仓库无；页面无版本号 | `#/details`（2026-10-24 抓取） |
| arXiv 链接 | 2608.16195、2601.22517、2310.04582、2508.08241、2506.12851、2502.01143 均有效 | arXiv |
| 直接 Python 评测 | `python -m holoboxing validate/match/view` 可用；`evaluation.robot_models`、`evaluation.runtime.observation_export` 均有 `__main__` 入口 | `evaluation/cli.py`、`evaluation/robot_models.py`、`evaluation/runtime/observation_export.py` |
| 评测运行时依赖 | `evaluation/docker/requirements.in` 无 torch；源码回退 `reference.py` 顶部 `import torch`，需单独装 torch | `requirements.in` 注释、`runtime/reference.py:7` |
| run.sh 限制 | Docker 运行加 `--read-only`、`--network none`，本地直接跑无此限制 | `run.sh:90,110` |
| 回放前端 | 仓库内只有 `viewer/src` 源码，无构建产物 `viewer/dist`，本地 `--view` 不可用 | `evaluation/viewer/` |
| 固定控制器 | `model_10500.onnx` 已下载 1,647,887,681 字节，SHA256 `b0681976…6a3fc86` 复核通过 | `models/holomotion-v2/exported/` |
| amp-motions | `train_rl --amp-motions` 为必填参数（prepared 目录） | `training/train_rl.py:23-27` |
| 本地时延判据 | `chain_benchmark.passed` 阈值为 `mean_ms ≤ 5ms`（严于 README 的 10ms）；validate 本地模式 `status=PASS` 不含时延门槛，仅 official 模式检查 | `evaluation/runtime/chain_benchmark.py`（`threshold_ms=5.0`、`passed=mean<=5.0`）、`evaluation/validation.py` |
| prior 存档 | 训练状态存 `last.pt`（恢复用），最佳验证轮存 `best.pt`（导入用，勿用于恢复） | `training/train_prior.py:198-199,271,281` |
| match 确定性（2026-10-25 实测修订） | 固定 `initial_state` 只保证**初态**一致；同输入重复局的 `match.json` **frames 层不等**（物理浮点非确定性，summary 层本例巧合相同）。A/B 判定不得依赖"两局 JSON 全等"，一律走每侧 3 局胜场制 | `evaluation/match.py:47` + 实测 `ab-p1-red` vs `ab-p1-red-repeat`（PHASE1 §5.3 回填） |

