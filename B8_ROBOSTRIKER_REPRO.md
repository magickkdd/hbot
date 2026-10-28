# B8 主规格：RoboStriker 论文方法的规则内复现

> **本文件是当前主规格。** B7 是数据/初始化前置，评估文档是可行性附录；除规则强制项外，Stage II/III 按论文结构复现。

> 版本 v1.0 · 2026-10-26
> 论文：[papers/robostriker-latent-strategic-games.md](papers/robostriker-latent-strategic-games.md)（PDF/全文同目录）
> 可行性分析：[ROBOSTRIKER_REPRO_ASSESSMENT.md](ROBOSTRIKER_REPRO_ASSESSMENT.md)
> 数据/初始化前置：[B7_FULL_DATA_SL.md](B7_FULL_DATA_SL.md)（InterHuman 全量 + RoboStriker 全量 + boxing prepared）
> 原则：**除规则强制项外，尽可能按论文方法 1:1 复现。** 规则强制项只做“最小替换”，不改变论文的算法结构。

---

## 0. 规则强制项与不可复现项（先划边界）

| 论文内容 | 规则/资源约束 | 本规格的替代/适配 | 是否改变论文算法 |
|---|---|---|---|
| Stage I：29-DoF motion tracker（Xsens 动捕 + RL） | 底层控制器必须使用赛事指定 HoloMotion V2，不得替换/修改 | 直接使用冻结 V2 控制器作为 Stage I；不训练 | 否（Stage I 被视为“已给定”） |
| 动作空间：29-DoF 关节位置目标 | 提交输出必须是 `target_pose [1,225]`（骨盆/双腕/双踝未来 5 帧） | 论文 decoder 的 29-DoF 输出→规则内的 225 target-pose decoder；原生 29-DoF decoder 只作对手/参考 | 是（规则强制） |
| 观测：论文 192 维 `s_goal` / 900 维 prior history | 提交输入固定 `obs [1,1097]` | 在模型内部用固定 FK/相对几何，把 1097 映射到论文的 192 维 `s_goal`；不增加输入维度 | 否（内部等价适配） |
| 4096 envs / RTX 4090 / 10^4 iterations | **书生 A100-1-80G**（80Gi / 4 vCPU / 16G RAM）/ 128 envs | RL 是 CPU 瓶颈（单核 MuJoCo），4 vCPU 只够 1–2 个训练进程；SL/prior 吃 GPU；结果/venv 全落 `/data` 持久 | 否（规模缩比） |
| Xsens 职业拳手 30min 动捕 | 不可得 | B7 的 RoboStriker 43 组 + InterHuman 7,810 → G1，作为 `D_motion` 替代 | 否（数据替代） |
| 真机 Unitree G1 部署 | 不可得 | 只做仿真 + 官方原生对手 native 验收 | 否（范围外） |
| 提交约束 | <50,000,000 B；高层+底层平均总时延 ≤10ms（我们 validate 用更严的 5ms） | 平均策略+latent decoder 均为小 MLP；ONNX <10MB | 否 |

**结论**：Stage I 被冻结控制器替代；动作输出因规则改为 225 target pose；其余 Stage II/III 的算法结构按论文复现。

---

## 1. 与官方参考实现的差距（必须补齐才算“按论文来”）

官方参考已经覆盖了论文的骨架，但缺三块关键内容：

| 组件 | 官方参考现状 | 论文要求 | B8 改动 |
|---|---|---|---|
| Stage II 先验 | **reference-only**：225→32→225，不条件化于 `s_prop` | 状态条件先验 `P_ξ(z\|s_prop)` + KL 正则，`λ_prior=0.001` | 新增 state-conditioned prior/decoder（v1） |
| Stage II decoder | 32→225，无条件 | `D_ψ(a\|s_prop,z)`：90 proprio + 32 latent → 225 | 新增 90+32→225 decoder |
| Stage III(a) warmup | `train_rl --training-mode standing` 仅 32 updates | 独立的关键 warmup 阶段；残差 `Δz` + 先验 `z_p`；`s_goal`；论文 task reward；AMP | warmup 200–500 updates；残差策略；`s_goal` adapter；reward 补齐 |
| Stage III(a) reward | facing/vel/active_hit/impulse 已有 | facing alignment、approach velocity、gated striking distance（0.5–1.2m）、contact-validated hit；`w_task=0.8 / w_style=0.2` | 增加 gated distance/contact hit 组合；对齐权重 |
| Stage III(b) LS-NFSP | **完全未实现**；只有冻结历史对手池 | 最佳响应 PPO + 平均策略 `π̄` + reservoir + η 混合 | 新增 average policy / reservoir / η / MSE |
| 策略宽度 | `sample.json` hidden 256/128 | 论文 actor/prior/decoder 使用 512/256/128 | B8 新增 `b8.json`（512/256/128），SL/warmup 用同一配置 |
| 评估指标 | HIT/倒地/胜负 | η_hit、ER（0.5–1.2m & cos>0.9）、BOS、TSτ | native 归因 + 技术指标；能量指标按 match.json 可得性 |

---

## 2. Stage I：冻结控制器（规则强制）

| 项 | 内容 |
|---|---|
| 论文原文 | 用 Xsens 职业拳手动捕训练 29-DoF tracker `π_trk(a_t \| s_self, s_ref)`；奖励 root/body 跟踪、速度匹配、控制正则 |
| 规则映射 | 赛事指定 HoloMotion V2 底层控制器，冻结、不可替换/修改 |
| 代码改动 | 无；沿用 `models/holomotion-v2` + `holoboxing/assets` |
| 验收 | `frozen_manifest` 哈希一致；`lower_identity` 与 prepared recipe 一致；不出现任何替换/微调 |

**说明**：论文 Stage I 是“已给定”的物理执行层；B8 的高层策略只输出 225 target pose，由 V2 控制器跟踪。

---

## 3. Stage II：状态条件 latent 蒸馏（论文方法 v1）

### 3.1 论文公式

- 编码器 `E_φ(z \| o)`，对角高斯；
- 解码器 `D_ψ(a \| s_prop, z)`；
- 状态条件先验 `P_ξ(z \| s_prop) = N(z \| μ_p, σ_p)`；
- 单位球约束：`ẑ = z / \|\|z\|\|_2`；
- 损失：`L_distill = L_rec + λ_prior L_prior`，`λ_prior=0.001`；
- `L_prior = D_KL(E_φ(z\|o) \|\| P_ξ(z\|s_prop))`。

### 3.2 规则映射

| 论文 | 本规格 |
|---|---|
| `a`：29-DoF tracker 动作 | `a`：225 target pose（规则输出） |
| `s_prop`：90 维 proprio（omega 3 + joint pos 29 + joint vel 29 + last action 29） | 从 1097 obs 的当前 108 维 self 中提取同样的 90 维布局（实现时按 `observations.py` 的字段顺序核对） |
| `o`：tracker 观测 `(s_self, s_ref)` | prepared 的 `(obs, target_pose)` 对；编码器输入 225 reference target |
| teacher：tracker `π_trk` | 用 prepared 的 225 reference target 作为 teacher（规则内可得的“高层动作 teacher”） |
| decoder 32→29 | decoder 90+32→225（规则强制适配） |

### 3.3 网络与训练

| 模块 | 结构 | 输入/输出 |
|---|---|---|
| Encoder `E_φ` | 225 → 512 → 256 → 128 → 32，ELU | reference target → unit z |
| Decoder `D_ψ` | 122（90 proprio + 32 z）→ 512 → 256 → 128 → 225，ELU | `s_prop`, z → target pose |
| Prior `P_ξ` | 90 → 512 → 256 → 128 → 32/64，ELU | `s_prop` → z_p（前 32 维） |
| Loss | `L_rec = MSE(a, D_ψ(s_prop, E_φ(a)))`；`L_prior = KL(E_φ \|\| P_ξ)`；`L = L_rec + 0.001 L_prior` | 单位球归一化 |

- 数据：B7 `processed/b7-boxing/prepared`（1,235 clip，约 73 万样本；含 `obs` 1097 + `target_pose` 225）。
- 训练：1–2h（A100 更快），validation 重建误差门控；记录 position/rotation/MPJPE（如可算）。
- **v0 fallback**：如果 state-conditioned v1 训练时间不够，先用 B1 reference-only prior 作为 v0，B8 Stage III 仍按论文结构（残差+NFSP）跑；v1 作为下一轮升级。文档与记录中明确标注 v0/v1。

### 3.4 代码落点

- 新模块 `holoboxing/training/state_conditioned_prior.py`（或在 `motion_prior.py` 增加 `StateConditionedMotionPrior`）；
- `prepare_prior`/`train_prior` 增加 `--state-conditioned` 选项：从 prepared `obs` 提取 `s_prop`，训练 encoder/decoder/prior；
- 导出：decoder+prior 冻结；Stage III 的 actor 通过 `z=Normalize(Δz+z_p)` 调 decoder；
- 验收：`validate` 双门槛；decoder 重建误差不差于 v0；prepared asset identity/recipe 一致。

---

## 4. Stage III(a)：Behavioral Warmup with Adversarial Priors

### 4.1 论文公式

- 残差策略：`π_θ(· \| s_goal)` 输出 `Δz_t`；
- 先验：`z_t^p ~ P_ξ^⊥(· \| s_prop)`；
- 组合：`z_t = Normalize(Δz_t + z_t^p)`；
- `s_goal`：ego 双拳→对手躯干、对手双拳→ego 躯干，旋转到 ego root 系；
- AMP 判别器：局部关节旋转/速度、base 角速度；梯度惩罚；风格奖励 `r_style`；
- 奖励：`R = w_task r_task + w_style r_style`，`w_task=0.8, w_style=0.2`；
- `r_task`：facing alignment、approach velocity、gated striking distance、contact-validated hit。

### 4.2 规则映射与实现

| 论文 | 本规格 |
|---|---|
| 192 维 `s_goal` | 固定 adapter：从 1097 obs 的当前 self + 15 task + FK 计算 192 维（proprio 90 + 相对躯干/拳 90 + attack 6 + defense 6）；不改变输入 1097 |
| `s_prop` | 同 Stage II 的 90 维 |
| 残差策略 192→32 | 192 → 512 → 256 → 128 → 32（论文宽度） |
| AMP 参考动作 | `--amp-motions` 指向 B7 prepared（拳击子集） |
| 站立对手 warmup | `--training-mode standing`，200–500 updates（论文关键阶段；官方示例只有 32） |
| task reward | 在现有 facing/vel/active_hit/impulse 基础上，补 0.5–1.2m gated distance + contact-validated hit；权重对齐 0.8/0.2 |

- **策略实现**：残差 `Δz` 叠加 state-conditioned prior 的 `z_p`；decoder 冻结；AMP 判别器训练；只有高层策略/判别器更新。
- **代码落点**：`spherical_rl.py`/`spherical_actor.py` 增加 residual + s_goal adapter；`tasks/combat/mdp/observations.py` 增加 `s_goal` 计算（固定 FK/相对几何）；`tasks/combat/mdp/rewards.py` 增加 gated distance/contact hit；新增 `b8.json`（512/256/128）。
- **验收**：warmup 后 native/standing 对局有有效 HIT；零倒地；validate PASS；ER/η_hit 有爬升。

---

## 5. Stage III(b)：Latent-Space Neural Fictitious Self-Play

### 5.1 论文算法（Algorithm 1 / Eq. 13）

- 每个 agent 同时维护：
  - 最佳响应 `π_RL`：PPO，用 `D_RL`；
  - 平均策略 `π̄`：监督学习 `D_SL`（历史最佳响应 reservoir），`L_SL = E\|\|π̄(o)-z\|\|²`；
- `D_SL` 增量更新 + reservoir sampling（容量 K）；
- 动作选择：`σ = η·π̄ + (1-η)·π_RL`，`η=0.1`；
- 论文超参：K=10^6/agent；latent 32；`w_task=0.8/w_style=0.2`。

### 5.2 规则内实现

| 论文 | 本规格 |
|---|---|
| 192 维 policy 输入 | 192 维 `s_goal` adapter（内部从 1097 计算） |
| 平均策略 192→512→256→128→32 | 同结构；与最佳响应同输入/同宽度 |
| reservoir K=10^6 | 192 维 obs + 32 z → 约 1.5GB/2 agents，58GB RAM 可行；按实际 batch 采样 |
| Eq. 13 混合 | 训练期按 η=0.1 选择平均策略；PPO 对平均策略产生的动作加 behavior importance correction，避免破坏 on-policy |
| 评估/导出 | 平均策略与 best-response **都导出**，native 分别验收（用户已确认） |
| 规模 | 128 envs；2k–4k updates（4 vCPU 受限，RL 可与 SL/prior 并行；论文 4,096 envs/10^4 iterations 不复现绝对数值） |

### 5.3 代码落点

| 文件 | 改动 |
|---|---|
| `spherical_rl.py` | 增加 `average_policy_{ego,opponent}`（192→512→256→128→32）；独立 Adam；`avg_model_state_dict`；`average_latent()` |
| `ppo.py` | 在 `storage.clear()` 前把 `(s_goal, unit(z))` 采样进 reservoir；新增 `update_average_policy()`（MSE，每 32 个 PPO update 一批）；混合动作的 behavior 权重 |
| `train_mujoco.py` | 创建 reservoir/optimizer；训练循环调用平均策略更新；保存/恢复 average state |
| `export.py` | `--policy {best-response,average}`；average 用同一个 state-conditioned decoder；输入 1097，输出 225 |
| 配置 | `b8.json`：hidden 512/256/128；task/style 0.8/0.2；η=0.1；reservoir 1e6 |
| 留档 | 分支 `b8-nfsp`；patch `records/b8-nfsp.patch`；不覆盖 B1/B5/B7 产物 |

### 5.4 验收

- validate 双门槛（`status=PASS`、`chain_benchmark.passed=true`，mean_ms ≤5.0）；
- native 4 局（两侧）对官方原生 RoboStriker；`fall_attrib.py` 归因；
- 技术指标：η_hit（有效命中/出拳尝试）、ER（0.5–1.2m & cos>0.9）、右/左肘 ROM、收拳-出拳循环、HIT 前回撤；
- 平均策略 vs best-response 同口径对比；与 H2b/B5a2 对比；
- 不满足 §6 判定不替换 `final-candidate`。

---

## 6. 采纳判定（与 H2b 基线）

必须同时满足：
1. validate 双门槛；
2. 我方 HIT > 对方 HIT；
3. 败局 = 0；
4. 我方倒地 = 0；
5. 技术门控通过（有收拳/出拳循环，不是固定伸臂）；
6. 不劣于 H2b/B5a2 的 native 结果。

任一不满足：保留 H2b；B8 只归档。论文方法复现的价值以“相对 H2b/B5a2 的增量”评估，不以绝对胜率对标论文。

---

## 7. 算力、时间与砍单

| 阶段 | 估时 | 说明 |
|---|---:|---|
| B7 prepared 合并/补跑 | 0.5–1.0h | 当前 715+补跑中；坏 clip 1630/2829/3272/7323 跳过 |
| Stage II v1 state-conditioned prior | 1.5–2.0h | 73 万样本；v0 用 B1 prior 可省 |
| Stage III(a) warmup 200–500 | 0.5–1.0h | standing + AMP + 论文 task reward |
| Stage III(b) LS-NFSP 2k–4k | 2.0–4.0h | 128 envs；含平均策略更新/reservoir |
| 导出/validate/native/归因/技术指标 | 1.0–1.5h | 平均策略+best-response 各 4 局 |
| 合计 | 5.0–9.5h | 单场 8h 可容纳；超时按砍单 |

砍单顺序（从下往上）：
1. Stage II v1 → v0（B1 prior）；
2. NFSP 4k → 2k；
3. native b 局（保留每侧 a 局）；
4. 全量 prior/generic retarget（B7 后置）；
5. **不可砍**：Stage III(a) warmup、LS-NFSP 至少 1.5k updates、validate、至少 2 局 native、归因/技术指标、平均策略与 best-response 各导出一次。

---

## 8. 规则合规检查表

| 检查 | 要求 | B8 |
|---|---|---|
| 输入 | `[1,1097]` float32 | 不变；`s_goal` 在模型内部计算 |
| 输出 | `[1,225]` float32 | state-conditioned decoder 输出 225 |
| 文件 | <50,000,000 B | 平均策略+decoder <10MB |
| 时延 | 平均总时延 ≤10ms（官方）/5ms（我们的门槛） | 小 MLP+decoder；预计 <5ms |
| 底层控制器 | 指定 V2，不得替换/修改 | 完全冻结 |
| 外部状态 | 单文件、无外部依赖 | reservoir 仅训练期；推理只用 1097 |
| 高层架构/算法 | 允许自定义 | latent/残差/AMP/NFSP 均属训练算法 |
| 许可证 | 论文代码 BSD-3-Clause；InterHuman 非商业/禁再分发 | 保留 LICENSE；原生 checkpoint 只作对手；数据只在实例内 |
| 数据 | RoboStriker/InterHuman 官方许可 | 不拷出、不再分发 |

---

## 9. 风险与偏差记录

| 风险 | 处置 |
|---|---|
| Stage II state-conditioned v1 训练失败/时间不够 | 回退 v0（B1 reference-only prior），Stage III 仍按论文结构；记录 v0 |
| `s_goal` 192 维在 1097 中不可完全恢复 | 固定 FK + 15 task 的近似；在文档标注近似维度；不增加输入 |
| NFSP 在 128 envs/2k updates 无显著提升 | 先看 warmup/技术指标；不够则 4k；仍无提升则记录负结果，不硬跑 10k |
| Eq.13 混合破坏 PPO on-policy | 加 behavior importance correction；若仍不稳，退回“best-response-only PPO + 平均策略监督”的保守 NFSP，并标注偏差 |
| 运行时长到界（书生单次最长 7×24h） | 结果/checkpoint 全在 `/data` 持久；`--max-runtime-seconds` 分块 + `--resume` 续跑 |
| 原生 29-DoF checkpoint 语义不同 | 只作评估对手；不接进提交物 |
| 两/多个 zero-row 坏 clip | B7 已跳过 1630/2829/3272/7323；从 sources/manifest 排除 |

---

## 10. 执行顺序（下一场）

1. 等 B7 补跑收尾；合并 715 + missing 成完整 prepared（坏 clip 排除）；`source_metadata` 同序/同 sha/同 group/split；`COMPLETE.json` 重算。
2. Stage II v1：state-conditioned prior/decoder（90+32→225，KL 0.001）训练；v0 fallback B1 prior。
3. Stage III(a)：standing warmup 200–500，残差 `Δz` + 先验 `z_p` + `s_goal` + 论文 task reward + AMP；`b8.json`（512/256/128）。
4. Stage III(b)：LS-NFSP 2k–4k（平均策略 + reservoir + η=0.1 + behavior correction）。
5. 导出平均策略与 best-response 两个 ONNX；validate 双门槛；native 4 局 + 归因 + 技术指标；与 H2b/B5a2 对比。
6. 满足 §6 才更新 `final-candidate`；否则归档，保留 H2b。

## 11. 产物清单

| 类型 | 路径 |
|---|---|
| Stage II v1 | `results/b8-prior-sc/`（encoder/decoder/prior, manifest） |
| Stage III(a) | `results/b8-warmup/`（policy, avg state, config） |
| Stage III(b) | `results/b8-nfsp/`（model_*.pt, avg_model_state_dict, reservoir 统计） |
| 导出 | `results/boxing-b8-avg.onnx`、`results/boxing-b8-br.onnx` |
| 校验 | `results/b8-avg-validation.json`、`results/b8-br-validation.json` |
| native | `results/native-b8-{avg,br}-side{0,1}-{a,b}/` |
| 归因/技术 | `records/b8-native-attrib.txt`、`records/b8-technique.json` |
| 代码 | 分支 `b8-nfsp` + `records/b8-nfsp.patch` |
| 记账 | `records/winrate.md` B8 段；`deployment/instance_snapshot.md` |
