# RoboStriker 复现可行性评估（HBOT 规则内）

> **执行以 [B8_ROBOSTRIKER_REPRO.md](B8_ROBOSTRIKER_REPRO.md) 主规格为准；本文是可行性分析与偏差记录附录。**
>
> 日期：2026-10-26 · 论文：[papers/robostriker-latent-strategic-games.md](papers/robostriker-latent-strategic-games.md)（全文 PDF/抽取文本同目录）
> 结论先行：**不能 1:1 复现**（Stage I 29-DoF tracker、4096 envs/10^4 iterations、真机与 Xsens 动捕都不在规则/算力范围内）；**可以在规则内复现并改造其高层策略算法**。官方参考实现已经包含论文 Stage II 的 32 维 latent 编解码器和 Stage III(a) 的一部分；真正缺的是 **Stage III(b) LS-NFSP**、论文式长热身/任务奖励课程，以及状态条件先验。
>
> **2026-10-26 用户确认**：先完成 B7 全量数据底座，再做 B8a；平均策略与 best-response **都导出并 native 验收**；原生 `model_3000.pt` 本轮**只做评估对手**，不作为训练对手（B8d 后置）。

---

## 1. 比赛规则硬边界（先划死）

来自 `official/HoloMotion/holoboxing/README.zh-CN.md`：

| 约束 | 要求 | 对复现的影响 |
|---|---|---|
| 提交物 | 单个 `.onnx`，<50,000,000 字节，结构+权重同文件 | 平均策略+解码器必须能塞进 <50MB（实际 <10MB） |
| 输入 | `obs` float32 `[1,1097]` | 不能改观测接口；论文的 192 维 `s_goal` 不能作为新输入维度 |
| 输出 | `target_pose` float32 `[1,225]` | 不能输出 29-DoF 关节目标；论文原生 decoder（→29）不能直接提交 |
| 底层控制器 | 必须使用赛事指定 HoloMotion V2，不得替换/修改 | 论文 Stage I 的 29-DoF tracker 无法复现，也不需要复现（已给定） |
| 时延 | 高层+底层平均总时延 ≤10ms（≥100 FPS，RTX 5090 官方口径） | 我们的 `validate --benchmark` 用更严的 5ms 门槛；加平均策略+解码器仍远低于上限 |
| 比赛 | 60s，50Hz 策略，200Hz 物理；法向冲量 ≥5 N·s 得 1 分，同手 200ms 冷却；骨盆/躯干 <0.30m 倒地；连续 3s KO | 论文的 η_hit/ER 可近似映射，但胜负以赛场规则为准 |
| 高层架构/算法 | **允许选手自行调整数据、高层网络结构和训练算法** | latent policy + 平均策略 + NFSP 合法 |
| 推理 | 单文件、无外部状态 | 平均策略/解码器必须是自包含的前向网络；不能依赖训练期 reservoir |

**关键判断**：规则只冻结「底层控制器 + 输入输出 + 文件/时延」，没有冻结高层网络结构或训练算法。因此论文的 **latent 高层 + NFSP** 属于允许范围；论文的 **29-DoF 低层/原始动作空间**不属于允许范围。

---

## 2. 论文 → 赛场映射表

| 论文组件 | 赛场对应 | 能否复现 | 说明 |
|---|---|---|---|
| Stage I：29-DoF 动作 tracker（Xsens 动捕，RL） | 赛事冻结的 HoloMotion V2 底层控制器 | **否（规则禁止）** | 低层物理执行已由官方给定；我们只能训练高层 1097→225 |
| Stage II：225/29 → 32 维单位球 latent → decoder | 官方 `MotionPrior`：225→32→225；`SphericalActor` 冻结 decoder；planner 输出 32 维 latent | **已实现（核心）** | 我们的 B1/B5/B7 都是 latent-space 策略，不是 225 直接回归 |
| Stage II：状态条件先验 `P_ξ(z|s_prop)` + 残差 `Δz` | 官方先验是 **reference-only**，没有 `s_prop` 条件；planner 直接输出 z | **部分缺失** | 可加状态条件 prior + 残差策略（B8c），但需重训 prior/decoder |
| Stage III(a)：站立对手 warmup | `train_rl --training-mode standing`，官方示例 32 updates | **已有，但太短** | 论文把 warmup 当作独立关键阶段；我们只跑 32 updates，需要 200–500 |
| Stage III(a)：AMP/style reward | `selfplay.json`/`sample.json` 的 style reward + AMP discriminator；`--amp-motions` | **已有** | 论文 `w_task=0.8 / w_style=0.2`；我们已有类似机制 |
| Stage III(a)：`s_goal`（拳-躯干相对几何） | 1097 obs：1080 历史 + 15 task + 2 match；仿真内部已有 192 维 actor measurement（proprio 90 + body-local 90 + attack 6 + defense 6） | **可改造** | 不改 1097 输入；用固定 FK/预处理把拳-躯干几何显式喂给高层（B8c/B8b） |
| Stage III(a)：task reward（facing/approach/gated distance/contact hit） | 已有 facing、vel、active_hit_shaping、impulse_strike、complete_punch | **部分已有** | 缺「0.5–1.2m gated strike distance」和论文的 contact-validated hit 组合；B8b 补齐 |
| Stage III(b)：LS-NFSP（best-response + average policy + reservoir + η） | `opponent_pool.py` 只有冻结历史快照；无 average policy/reservoir/η | **未实现（最大缺口）** | B8a：加平均策略 MLP + reservoir + MSE + η=0.1；这是论文相对我们现有系统的真正增量 |
| 论文 4096 envs / RTX 4090 / 10^4 iterations | A10 23GB / 128 envs / 剩余约 14h | **只能缩比** | 环境数 ×32、迭代数 ×5，总算力差约 2 个数量级；只能看相对趋势/消融，不能复现绝对胜率 |
| 29-DoF raw action-space 基线、BOS/TSτ | 赛场输出是 225 target pose；低层冻结 | **不可复现** | 论文 Table I 的 29-DoF 对比在我们这里没有对应物；我们的「raw」是 225 target pose |
| 真机部署、Xsens 数据 | 无 | **否** | 用仿真 + 官方原生对手验收；数据用 RoboStriker/InterHuman 替代 |

---

## 3. 已经复现了什么（比想象中多）

1. **Stage II 的核心已经在官方参考里**：`holoboxing/training/motion_prior.py` 明确写的是 “Reference-only 225 -> unit-sphere 32 -> 225 motion representation”；`SphericalActor` 输出 32 维 latent，`MotionDecoder` 冻结后解码成 225。我们的 B1/B5/B7 全部走这条路径。
2. **Stage III(a) 的一半已有**：站立热身（32 updates）、AMP 风格奖励、历史对手池都已存在。
3. **原生对手就是论文 Stage III 模型**：`/mnt/workspace/model_3000.pt` 的 state dict 含
   - `taskpolicy_ego`：192 → 512 → 256 → 128 → 32（最佳响应策略）
   - `prior`：900 → 512 → 256 → 128 → 64（状态条件先验，前 32 维作 z_p）
   - `decoder`：122（90 proprio + 32 latent）→ 512 → 256 → 128 → 29
   - `avg_model_state_dict` / `avg_model_state_dict_opponent`：192 → … → 32（NFSP 平均策略）
   - 对应 normalizer：task 192 / prior 900 / decoder 90 / privileged 225
   这直接证明论文的 NFSP + latent prior/decoder 结构；但它输出 29-DoF 关节目标，**不能作为我们的提交物**，只能作为对手/教师/参考实现。
4. **因此论文最著名的「latent vs raw 29-DoF」对比在我们这里不可直接测**：我们的动作空间已经是 225 target pose，且 planner 已经是 latent。我们的 B5 失败（0 命中、2m 对峙）发生在 latent 空间内，说明「加 latent」本身不是我们的瓶颈；论文剩下的真正增量是 **NFSP + 长热身/课程 + 状态条件先验**。

---

## 4. 不能复现的部分（明确说死）

- **Stage I 29-DoF tracker**：规则禁止替换/修改底层控制器；论文的 Xsens 职业拳手动捕也不可得。
- **29-DoF 原始动作空间基线 / BOS / TSτ**：赛场动作是 225 target pose，29-DoF 关节目标不可提交；这些指标只能作为论文背景，不是我们的实验变量。
- **4096 envs / RTX 4090 / 10^4 iterations**：A10 23GB、128 envs、剩余约 14h；算力差约 2 个数量级，不能复现论文的绝对胜率/收敛曲线。
- **原生 decoder（→29）直接复用**：输出维度/语义都不对；必须用我们自己的 32→225 `MotionDecoder`。
- **真机部署**：无硬件；只做仿真 + 官方原生对手 native 验收。

---

## 5. 规则内的复现路线（B8）

### B8a（首选，最高 ROI）：在现有 latent-32 / 225 管线上加 LS-NFSP

- **不改**：1097→225 接口、冻结 V2 控制器、32 维 latent decoder、AMP。
- **新增**：
  - 平均策略 `π̄`：MLP 1097 → 512 → 256 → 128 → 32（ELU），输出单位球 latent；
  - reservoir `D_SL`：容量 K=10^5（论文 10^6 的缩比；1097+32 float ≈ 0.45GB），reservoir sampling；
  - 训练期动作选择：`σ = η·π̄ + (1-η)·π_RL`，η=0.1；
  - 平均策略损失：`L_SL = E ||π̄(o) - z||²`，每 32 updates 更新一次；
  - 评估/导出：分别导出 **平均策略+decoder** 与 **best-response+decoder** 两个 ONNX，native 对比。
- **训练顺序**：B7 SL → 站立 warmup 200 updates → NFSP 2,000 updates（128 envs）。
- **算力**：约 2–3h（PPO 与现有 selfplay 同量级，平均策略更新很便宜）。
- **验收**：validate 双门槛；native 4 局（两侧）；fall_attrib；技术指标（右/左肘 ROM、收拳-出拳循环、HIT 前回撤）；与 H2b/B5a2/B7 SL 同口径。
- **为什么这是第一优先**：这是我们**唯一没试过**的论文核心机制；B5 的失败（奖励/探索/训练量）都已试过，NFSP 直接针对「非传递性策略循环 + 灾难性遗忘 + 2m 对峙均衡」。

**B8a 代码落点（已核对现有代码）**：

| 文件 | 改动 |
|---|---|
| `holoboxing/training/spherical_rl.py` | 增加 `average_policy_{ego,opponent}`（与 best-response 同输入 1095、同 hidden dims、输出 32），单独的 Adam optimizer；`average_latent(role, obs)`；checkpoint 增 `avg_model_state_dict` |
| `holoboxing/training/rsl_rl/algorithms/ppo.py` | 在 `update()` 的 `storage.clear()` 之前，把 `(observations, unit_vector(raw_action))` 采样进 reservoir；新增 `update_average_policy()`（MSE，每 32 个 PPO update 跑一批） |
| `holoboxing/training/scripts/train_mujoco.py` | 创建 reservoir（K=1e5，ring buffer）、average optimizer；训练循环里调用平均策略更新；保存/恢复 average state |
| `holoboxing/training/export.py` | 增加 `--policy {best-response,average}`；average 时用同一个冻结 decoder 包装 average latent net；输入仍是 1097、输出仍是 225 |
| 分支/留档 | 新分支 `b8-nfsp`，patch 记录到 `records/b8-nfsp.patch`；不覆盖 B1/B5/B7 产物 |

**一个必须明确的实现取舍**：PPO 是 on-policy；论文 Eq. 13 把 average policy 的动作混进行为策略，会让 PPO 的 ratio 变成 off-policy。B8a 第一版采用**保守 NFSP 变体**：
- PPO 只用 best-response 自己的 on-policy rollout；
- average policy 只用 reservoir 里的 `(obs, z)` 做监督；
- 评估/导出用 average policy（以及 best-response 对照），不在 PPO 更新中混入 average 动作。
这与论文的「平均策略稳定 + reservoir」核心一致，同时不破坏 PPO 的 on-policy 假设；如果第一版有正信号，再尝试论文的 η 混合（加 behavior importance correction）。

### B8b（第二优先）：论文式长热身 + task reward/课程

- 站立对手 warmup 从 32 → 200–500 updates；
- 补论文的 task 项：facing alignment、approach velocity、**gated striking distance（0.5–1.2m）**、**contact-validated hit**；`w_task=0.8 / w_style=0.2`；
- 再进入 NFSP（B8a）。
- **算力**：warmup 0.5–1h + NFSP 2–3h。
- **风险**：我们已做过大量 reward shaping（H1–H5），单纯加权重可能再次失败；但与「长 warmup + NFSP」组合是论文的完整配方，值得作为单变量实验。

### B8c（可选）：状态条件先验 + 残差 latent

- 按论文 Eq. 8–9：`z_t = Normalize(Δz_t + z_t^p)`，`z_t^p ~ P_ξ(·|s_prop)`；
- 训练 `P_ξ(z|s_prop)`（输入 90 维 proprio，输出 64/32），KL 系数 0.001；
- 需要重训/微调 32→225 decoder 使其条件化，风险较高；建议在 B8a/B8b 有正信号后再做。

### B8d（可选，高成本）：把原生 RoboStriker 作为训练对手

- 原生 checkpoint 输出 29-DoF 关节目标；可用 FK 转成 225 target pose 作为训练对手，或蒸馏其 32 维 latent；
- 这能直接提供论文级对手课程，打破自对弈 2m 均衡；但接口/动作语义转换有误差，属于高成本实验。

### B8e（可选）：latent 维度消融

- 16/32/64；论文 Table II 显示 32 最优；我们已有 32，低优先级。

---

## 6. 规则合规检查表（B8 提交前）

| 检查 | 要求 | B8a 设计 |
|---|---|---|
| 输入 | `[1,1097]` float32 | 不变 |
| 输出 | `[1,225]` float32 | 平均策略/best-response → 32 latent → 冻结 decoder → 225 |
| 文件 | <50,000,000 B | 平均策略 ~0.7M 参数 + decoder ~0.3M，ONNX <10MB |
| 时延 | 平均总时延 ≤10ms（官方）/ 我们 5ms 门槛 | 加一个小 MLP + decoder，预计 <5ms |
| 底层控制器 | 指定 V2，不得替换/修改 | 完全不动 |
| 外部状态 | 单文件、无外部依赖 | 平均策略只吃 1097；reservoir 只在训练期 |
| 训练算法 | 允许自定义 | NFSP/AMP/latent 均属训练算法 |
| 许可证 | 论文代码 BSD-3-Clause；InterHuman 非商业、禁再分发 | 保留 LICENSE；不提交原生 checkpoint；数据只在实例内 |
| 原生 checkpoint | 不能作为提交物 | 只做对手/评估/参考 |

---

## 7. 算力与时间（剩余约 14h）

| 阶段 | 估时 | 说明 |
|---|---:|---|
| B7 拳击 retarget 1,411 | 0.7–0.8h | 单路约 30–45min；已启动 |
| B7 拳击 prepare | 1.5–2.5h | 6–8 workers；本场尽量跑完 |
| B7 通用 retarget 6,399 | 2.5–3.0h | 下一场单路/双路 |
| B7 全量 prior | 1.5–2.0h | 2h cap |
| B7 SL 19 epochs | 1.5–2.0h | 拳击子集 |
| B7 验收 | 0.7–1.0h | validate + native 4 局 + 归因/视频 |
| **B7 小计** | **8.4–11.3h** | 已含当前场 |
| B8a NFSP | 2.0–3.0h | 128 envs / 2k updates |
| **B7+B8a** | **10.4–14.3h** | 与剩余 14h 基本贴合；需要严格砍单 |
| B8b 长热身+奖励 | +2.5–4.0h | 建议下一轮/下一场 |

砍单顺序：B8b → B8c → B8d → B7 native b 局 → B7 SL epochs 19→12。**不可砍**：B7 拳击 prepare/verify、全量 prior、B7 SL、validate、至少 2 局 native、B8a 的 NFSP 对照。

---

## 8. 风险与未决问题

1. **NFSP 在 128 envs / 2k updates 下可能不显著**：论文是 4096 envs / 10^4 iterations；我们只能看相对趋势。若 B8a 无提升，优先检查 warmup/奖励（B8b），而不是继续加迭代。
2. **论文 Eq. 13 与 Algorithm 1 的 η 方向不一致**：Eq. 13 是 `η·π̄ + (1-η)·π_RL`，Algorithm 1 是 Bernoulli(η) 选 π_RL；我们需要固定一种并记录。建议按 Eq. 13（η=0.1 平均策略占 10%），与「anticipatory」语义一致。
3. **平均策略的评估口径**：论文评估用平均策略还是 best-response 未明说；原生 checkpoint 两者都有。B8a 同时导出两个 ONNX，native 分别验收。
4. **状态条件先验/解码器**：官方 `MotionPrior` 是 reference-only，不是论文的状态条件版本；B8c 需要重训，风险高，不作为第一优先。
5. **原生 29-DoF 与 225 target pose 的语义差**：不能把原生 decoder 直接接进我们的提交物；B8d 的 FK 转换有误差，需要单独验证。
6. **side 不对称**：B5a2 已出现 side0 0:11 / side1 对手自倒；B8 必须双侧验收，不能用单侧胜利替代 KPI。
7. **许可**：论文代码 BSD-3-Clause 可复用；InterHuman 非商业、禁再分发；原生 checkpoint 只做评估，不提交。

---

## 9. 建议

**推荐路线：B7 数据底座 → B8a（LS-NFSP）→ 若有效再 B8b（长热身+论文奖励）。**

- B7 不是被论文取代，而是论文 Stage II/III 的前置：全量拳击数据的 SL/prior 是 B8a 的初始化；没有它，NFSP 会在弱策略上放大噪声。
- B8a 是「论文相对我们现有系统」的真正增量，成本 2–3h，且完全合规；先做它，再决定是否投入 B8b/B8c。
- 不建议追求 1:1 复现：Stage I/4096 envs/真机不在规则与算力内，强行对齐只会烧掉剩余额度，且无法提交 29-DoF 模型。

**需要用户拍板的三件事**：
1. 先完成 B7 再 B8a（推荐），还是暂停 B7、直接在 B1/B5 上做 B8a？
2. B8a 的提交候选导出「平均策略」还是「best-response」（或两个都验收后再定）？
3. 是否允许把原生 `model_3000.pt` 作为**训练对手**（B8d），还是只做评估对手？
