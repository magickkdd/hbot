# HBOT B5 创新方案：文献驱动的四刀规格

> 版本 v1.0 · 建档 2026-10-25
> **执行规格见 `B5_EXECUTION.md`（v2.0 完整版，2026-10-25 定稿）**：R1/R2/R3 已全部代码核实（R1 解除——冲量数据管道现成，刀③纯 reward 层；R3 确认需改 opponent_pool.py 两处）；权重 20、early-kill 门（update 100/300）、主 KPI = 有效命中数、基线补测 4 局、两场 8h 剧本均已写死。
> 定位：PHASE3（B2 改进）与 B4 破局系列（H1–H5）已执行完毕，本文档是下一轮实验 **B5** 的完整规格。
> 执行窗口：即日起 – 11/24（**新开分支软线**，过了只续不新）；硬线：**12/1 代码冻结（硬）、12/3 初赛截止**。与 B5_EXECUTION.md 口径一致。
> 前置状态：最终候选已切换至 **H2b 谱系**（见 §0）。

---

## 0. 现状基线（B5 的出发点）

### 0.1 最终候选：H2b

| 项 | 值 | 出处 |
|---|---|---|
| 上层模型 | `results/boxing-h2b.onnx`，sha256 `f56fc9d834b40f45aaff1e7f9eba6bb591b27918c2466e991a1a32aadd98cafa` | `records/h2b-validation.json` |
| checkpoint | `/mnt/workspace/HoloMotion/results/h2b-selfplay/model_1280.pt`，sha256 `019b5a37…`，累计 **1280 updates** | `records/b4-h2b-selfplay.json` |
| validation | PASS，`mean_ms = 4.99145231`（threshold 5.0，**贴线通过**） | 同上 json，chain_benchmark 段 |
| native 对战 | 2-0，**但均为对手自倒**（翻案见 winrate.md ★★ 段：side1 倒地瞬间距 1.44m 铁证；side0 我方 HIT=0）；HIT 事件口径我方两局 0 命中、对方 24 | `records/native-h2b-side{0,1}-result.json` + match.json 帧分析 |
| 对手 | RoboStriker `model_3000.pt`，sha256 `9f8bb366…`（evaluate_native 适配器） | 同上 |

**注意**：`winrate.md` ★ 段曾误记 H2b 延迟为 4.75ms，权威口径以 `h2b-validation.json` 的 4.99145231 为准（已回填修正）。贴线意味着**每次采纳新模型前必须重跑 validate**，GPU 负载波动可能把 4.99 推过 5.0。

### 0.2 H 系列教训（B5 硬约束，违反即重蹈覆辙）

| 教训 | 证据 | B5 约束 |
|---|---|---|
| 新阶段重置清空战斗行为 | H4 从 H2b 重开新阶段 → native 0:16/17:0 双败 | **只准 `--resume` 续谱系，禁止新开阶段** |
| 防御惩罚扼杀进攻 | H5（-8→-24）→ native 1:10/7:0 双败，几乎不出拳 | `defensive_hit_penalty` 保持 -8 不动 |
| sigma 网格已穷尽 | {0.5, 1.0, 2.0} × 门 {0.4, 0.8} 全测；2.0=对峙舒适区（白拿 shaping）、0.5=梯度够不着、1.0+重开=谱系断裂 | 保留 H2b 的 sigma 2.0 + vel 重塑组合 |
| 加探索无效 | E1（init_noise_std 0.2→0.5）：strike 峰值反而低于基线 | 不再动探索参数 |
| 镜像 A/B 无效 | 18 局本地对战全 0:0，无法区分「会打」与「不会打」 | **验收一律用 evaluate_native**（§3） |

### 0.3 结构性瓶颈（B5 要打的三件事）

1. **行为迁移缺口**：训练随机策略有偶发命中（offensive 22% updates 非零），确定性对局（均值动作）下距离钉死 2.0m（0/3000 帧 <1.5m）。
2. **镜像均衡**：自对弈双方同步学会同一边缘策略，共谋不接触。
3. **无课程**：直接从 2.0m 对峙距离发现「命中」这个稀疏事件太难（E1 已证明加探索救不了）。

---

## 1. 文献调研结论（2026-10-25）

### 1.1 对我们直接有用的三条

**① RoboStriker（arXiv 2608.16195，官方教程引用版）——入场券标尺。**
潜空间策略博弈：高层在 latent 空间做博弈决策、低层执行——与我们同款架构（actor 输出 32 维向量 + 冻结 decoder）。它的 preview 仓库放出 `model_3000.pt`：**对手的公开入场券就是 3000 updates 量级的单线训练**。我们最强谱系 1280，强队只会更长。含义：长训是入场券不是创新，但必须做。

**② GGSD（arXiv 2609.40137，AMP 作者 Xue Bin Peng 组，2026-10）——自对弈催生技能的条件。**
分层自对弈，高层从离散技能集选择、低层执行，**对过去自己的快照打**，自发涌现技能组合，真机 Unitree G1 可玩。对我们的背书：对手池放**历史快照**（小 interval、多容量）比只打「当前自己」更能催生行为多样性——PHASE3 P1 的 interval=32 方向正确，B5 可更激进。

**③ 联赛训练线（IEEE CoG 2020 格斗游戏；NeurIPS 2023 opponent-aware league）——镜像均衡的标准解药。**
AlphaStar 式三类 agent：主力（main agent）+ 主 exploiter（唯一目标=打爆当前主力）+ 联赛 exploiter（打爆整个对手池）。exploiter 暴露的正是镜像自对弈盲区——我们实测的 2.0m 共谋对峙就是典型盲区。格斗游戏 AI 领域这是成熟标准配置，但**尚未见公开工作把它用于人形拳击**——这是 B5 最有含金量的创新点。

### 1.2 其余工作（一句话结论）

| 工作 | 结论 |
|---|---|
| ZEST（2602.00401，Science Robotics） | 零样本运动技能迁移；参考价值低，比赛接口不允许换架构 |
| PressMimic（2606.26741） | 压力引导捕获→控制；与我们的瓶颈不对口 |
| SuperMimic（2610.28480，CoRL26 oral） | 可微世界模型单卡 3 小时；训练侧理论可用但工程量大，B5 不做 |
| ALMI（2504.14305）/ HAML（MDPI 2026） | 上下半身对抗/多技能绑定；中等相关，备用 |

**诚实边界**：RoboStriker 与 GGSD 目前是摘要级理解 + preview 仓库实测数据；论文全文的 reward 与对手池配置细节未拿到。可选动作：深挖 RoboStriker 全文（等于提前看强队作业），优先级低于直接开训。

---

## 2. B5 四刀（按性价比排序）

> 排序原则：GPU 额度有限（§2.5），先做「独特性高 × 工程量小」的。
> 时速基准：H2b 实测 3.79 s/update（768 updates / 2914s，128 envs），下同。

### 刀③：冲量对齐奖励（最优先——独特性最高，直击评分规则）

- **依据**：训练侧 `offensive_hit_reward_punch` 用 contact+velocity 启发式（`mujoco_rewards.py:144-157`），评测侧用 **5 N·s / 20ms 法向冲量**（`evaluation/runtime/impulse_hit_rules`）。H 系列证明启发式奖励可被「原地挥手」hack（sigma 2.0 下 2m 处 exp(-1)≈0.37 白拿 shaping）。把真判定规则搬进训练奖励 = 奖励不可 hack + 训练信号与比分直接对齐。**公开调研中没有队伍在此赛事语境做此事。**
- **改动**：代码级 patch——将评测的冲量判定移植为训练 reward 项（替换 `offensive_hit_reward_punch`，或以冲量门控它）。
- **第一步（零训练成本）**：核实训练侧 env 能否读到接触冲量（MuJoCo contact impulse 在 `mujoco_env` 管线中的可得性）→ 风险 R1（附录 C）。
- **工程量**：中。**GPU**：与刀①共用。

### 刀①：长谱系续训（入场券，与刀③绑定执行）

- **依据**：model_3000 锚点；H2b 谱系 1280 处 offensive 命中率 22%、native 两胜（虽系对手自倒，抗打不倒是真实的）——谱系仍在上升期，远未见顶。
- **改动**：从 `results/h2b-selfplay/model_1280.pt` 出发，叠加刀③ patch 后 `--resume` 续训。**代码基座必须是 H 谱系分支 `p4-h1-vel@dcbd9b2`（patch 518e0257）**，叠加新 patch 需 `--allow-resume-code-change` 并记录原因。flags 与 H 谱系完全一致（128 envs / capacity 8 / refresh 32 / seed 42 / `selfplay.json`）。
- **节奏**：1280 → **2560**（+1280 ≈ 1.35h）→ native 验收 → 若仍胜且失分减少 → **4096**（+1536 ≈ 1.62h）→ native 验收。
- **硬约束**：只 `--resume`，不新开阶段（H4 教训）。

### 刀②：初始距离课程（第二优先）

- **依据**：瓶颈③（无课程）；E1 证明探索加码无效，唯一未试的破局方向。
- **改动**：代码级 patch `envs/mujoco_env.py` 的 reset——初始相对距离从当前 2.0m 改为 **1.0–1.2m 起步**，随 update 数逐渐推远到 2.0m。近距起手同时天然瓦解镜像对峙（一举打瓶颈②③）。
- **假设需验收**：比赛从官方固定 initial_state（2.0m 对峙）开始，训练近距起手的行为需泛化回 2.0m——用 native 双局确认，不成立即弃。
- **工程量**：中。**GPU**：512 updates 试验 ≈ 0.54h/轮，预留 2–3 轮调参。

### 刀④：exploiter 联赛（最高含金量 × 最高工程量，预算门控）

- **依据**：联赛训练标准做法（§1.1③）；我们实测的 2.0m 镜像共谋。
- **改动**：两步。
  1. 从主谱系 fork 一个 **exploiter 分支**：对手池只放主模型最新快照，唯一目标=打爆主力；训 512 updates（≈0.54h）。
  2. 把训好的 exploiter checkpoint **注入主训练对手池**，主力继续 `--resume`。`opponent_pool.py:56-70` 已有快照池骨架；「外部 checkpoint 注入」是否需改代码待核实 → 风险 R3。
- **硬约束**：exploiter 只进对手池，**永不作为提交候选**（它过拟合单对手，Elo 口径下是废棋）。
- **门控**：仅当刀③① 执行完且额度剩余 **> 5h** 时开启（与 B5_EXECUTION §6 统一）。

### 2.5 预算表与冻结线

| 项 | GPU 估时 | 累计 |
|---|---|---|
| 刀③核实 R1 + patch（不训练） | ~0.2h | 0.2h |
| 刀①至 2560（含刀③ patch） | 1.35h + native 0.2h | 1.75h |
| 刀①至 4096（视验收） | 1.62h + native 0.2h | 3.55h |
| 刀②一轮（512） | 0.54h + native 0.2h | 4.3h |
| 刀②调参预留 1–2 轮 | ≤1.5h | ≤5.8h |
| 刀④（门控）exploiter 512 + 主线 256 | ~0.8h ×2 轮 + native | ≤8h |

- **额度（2026-10-25 晚更新）**：总剩余 **16h，两次开机 × 8h**（用户口述，已记 instance_snapshot.md；31.5h 旧基线作废）。**额度不明确不开训。**
- **冻结线**（沿用 PHASE3 量化规则）：`剩余额度 − 2.5h ≥ 分支成本` 且日期 ≤ 11/24 才开新分支；此后只续不新。

---

## 3. 验收口径（B5 全程统一）

1. **唯一有效验收 = `evaluate_native` vs `model_3000`，每侧 2 局 + 事件级倒地归因**。基准 = H2b 的 HIT 0/0（要改的就是这个数）。镜像 A/B 弃用。（本节为调研版原文，执行口径以 B5_EXECUTION.md §4.5 两级判定为准）
2. **判定**：净 KO 时间更短 / 失分更少 → 采纳并更新 `final-candidate.txt`；劣势 → REJECT，master 回退，谱系 checkpoint 留档。
3. 每轮采纳前 `validate`，`mean_ms < 5.0`（H2b 4.99 贴线，逐轮重测）。
4. 命令模板（主方案 §7.2 原样）：

```bash
export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
python -m holoboxing.training.evaluate_native \
  --candidate results/boxing-h2b.onnx \
  --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
  --opponent-sha256 "$ROBOSTRIKER_SHA256" \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --candidate-side 0 --output results/native-b5-side0 --device cuda:0
# 另一侧：--candidate-side 1 --output results/native-b5-side1
```

---

## 4. 代码修改纪律（承接 PHASE3 附录 B）

- patch 只经 `/tmp` 中转，sha256 留档 `records/`；实验分支命名 `b5-<刀名>`；每轮结束 master 回退。
- `--allow-resume-code-change` 每次使用都在 `winrate.md` 记录原因与 patch sha。
- resume 的 identity 校验含 config 哈希：全程同一 `selfplay.json`；`--updates 0` 会被拒（`on_policy_runner.py:478`），分块续跑带 `if N<目标` 守卫。

## 5. 与提交节奏的衔接

- **H2b 现在就提交，不等 B5**：平台真实对手情报比任何本地实验值钱；窗口 10:00–18:00、3 次/日、20 次累计上限，早提交早拿回放。
- B5 每次采纳新谱系 → 更新 `final-candidate.txt` + `winrate.md` → 下一个提交窗口提交。
- 12/1 冻结后只提交已验收模型，不再改代码。

---

## 附录 A：论文清单

| 工作 | 链接 | 用处 |
|---|---|---|
| RoboStriker（潜空间策略博弈，官方引用版） | arxiv.org/abs/2608.16195 | 入场券标尺；同款架构 |
| RoboStriker-Preview（model_3000 出处） | github.com/yinkangning0124/RoboStriker-Preview | native 验收对手 |
| GGSD：Game-Guided Skill Discovery | arxiv.org/abs/2609.40137 | 历史快照池背书 |
| Mastering Fighting Game Using Deep RL（IEEE CoG 2020） | ieee-cog.org/2020/papers/paper_207.pdf | 联赛三 agent 模板 |
| A Robust and Opponent-Aware League Training Method（NeurIPS 2023） | neurips.cc/virtual/2023/poster/70220 | exploiter 设计细节 |
| ZEST | arxiv.org/abs/2602.00401 | 低相关 |
| PressMimic | arxiv.org/abs/2606.26741 | 低相关 |
| SuperMimic | arxiv.org/abs/2610.28480 | 训练侧理论，不做 |
| ALMI / HAML | arxiv.org/html/2504.14305v2 / mdpi.com/2076-0825/15/4/212 | 备用 |

## 附录 B：机制出处（均已核实）

| 机制 | 出处 |
|---|---|
| 训练 offensive_hit 启发式（threshold 1.0 + vel_threshold 1.0） | `training/envs/mujoco_rewards.py:144-157` |
| 评测 5 N·s / 20ms 冲量判定 | `evaluation/runtime/impulse_hit_rules` |
| 对手池快照机制（capacity/interval） | `training/opponent_pool.py:56-70` |
| resume 代码一致性校验 | `training/rsl_rl/runners/resume.py:314-340` |
| 5ms 本地门槛（mean_ms ≤ 5.0） | `training/chain_benchmark.py` + `h2b-validation.json` |
| 时间预算（save reserve 600s / 优雅停止） | `training/time_budget.py` |

## 附录 C：风险清单

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 训练侧 env 拿不到逐接触冲量（刀③地基） | 开工第一步零成本核实；不可得则降级为「冲量近似」（力 × dt 累积），仍优于现行启发式 |
| R2 | 刀②近距起手行为泛化不回 2.0m 官方开局 | native 双局验收，不成立即弃，谱系不受损（独立分支） |
| R3 | opponent_pool 不支持外部 checkpoint 注入（刀④） | 若需改 `opponent_pool.py` 则工程量上浮，触发门控重估 |
| R4 | mean_ms 贴线（4.99）在验收机上越线 | 每轮重测；越线模型不采纳不提交 |
| R5 | 额度耗尽中断谱系 | 冻结线量化规则 + 每块 `--max-runtime-seconds` + stop_reason 检查（沿用 PHASE2 口径） |
