# RoboStriker 论文提取与要点

| 项 | 内容 |
|---|---|
| 标题 | RoboStriker: Latent-Space Strategic Games for Autonomous Humanoid Boxing |
| 作者 | Kangning Yin, Kaige Liu, Zhe Cao, Wentao Dong, Weishuai Zeng, Tianyi Zhang, Qiang Zhang, Jingbo Wang, Jiangmiao Pang, Yang Li, Ming Zhou, Weinan Zhang |
| 单位 | 上海交通大学、上海人工智能实验室、上海创智学院、华中科技大学、中国科学技术大学 |
| 版本 | arXiv:2608.16195v1 [cs.RO] · 2026-08-17 · 9 页 |
| 原始 PDF | [robostriker-latent-strategic-games.pdf](robostriker-latent-strategic-games.pdf) |
| 全文抽取 | [robostriker-latent-strategic-games-fulltext.txt](robostriker-latent-strategic-games-fulltext.txt)（43,417 字符，按 PAGE 1–9 分段；图表文字为 PDF 文本层原样，含少量排版噪声） |
| 官方早期版 | 比赛 README 链接的 RoboStriker（2026-01）为 arXiv:2601.22517；本 PDF 为 2026-08 更新版 |
| 代码/许可 | 官方预览仓库 `yinkangning0124/RoboStriker-Preview`；`native_robostriker.py` 注明 BSD-3-Clause，并保留 `rsl_rl/LICENSE` |

## 一句话结论

论文把「人形拳击」建模为**由预训练动作解码器诱导的双人 latent-space 零和马尔可夫博弈**：先训练 29-DoF 动作跟踪器（Stage I），把动作蒸馏到 32 维单位球 latent 空间（Stage II），再在 latent 空间做 AMP 站立热身（Stage IIIa）和 **Latent-Space Neural Fictitious Self-Play（LS-NFSP，Stage IIIb）**；论文的核心增量是「在结构化 latent 空间里做 NFSP」，而不是直接用 29-DoF 关节空间做自博弈。

## 方法拆解

### Stage I：Expertise Motion Tracker（29-DoF）

- 数据：职业拳手 Xsens 动捕，约 30 分钟 @50Hz，左右镜像翻倍；用 Generalized Motion Retargeting 重定向到 Unitree G1。
- 跟踪器：`π_trk(a_t | s_self_t, s_ref_t)`，`s_self = (s_prop, s_priv)`；`s_ref` 是未来 K=3 帧的全身参考位姿/速度。
- 动作：29 个关节位置目标，由 PD 控制器跟踪。
- 奖励：root/body 位姿与朝向跟踪、线/角速度匹配、动作率/关节限位/异常接触正则；姿态崩溃、末端高度违规、10s 上限早停。
- 目的：给高层策略一个物理可行、可复用的低层执行器。

### Stage II：Topological Latent Distillation（32 维单位球）

- 教师-学生蒸馏：编码器 `E_φ(z | o)`（对角高斯）、解码器 `D_ψ(a | s_prop, z)`、状态条件先验 `P_ξ(z | s_prop)`。
- 编码器把观测 `o=(s_self, s_ref)` 映射到 latent；解码器在 `(s_prop, z)` 条件下重建教师动作 `a_t ~ π_trk`。
- 关键约束：`ẑ = z / ||z||_2`，latent 位于单位超球面（topologically bounded）。
- 损失：`L_distill = L_rec + λ_prior L_prior`，`λ_prior = 0.001`；`L_prior = D_KL(E_φ(z|o) || P_ξ(z|s_prop))`。
- 解码器训练后冻结，作为高层策略的「物理动作流形」。

### Stage III(a)：Behavioral Warmup with Adversarial Priors

- 先对**保持站立的静止对手**学习出拳，解决竞争冷启动。
- 残差策略：`π_θ(· | s_goal)` 输出 `Δz_t`，叠加固定行为先验 `P_ξ^⊥`：
  - `z_t = Normalize(Δz_t + z_t^p)`，`Δz_t ~ π_θ(·|s_goal)`，`z_t^p ~ P_ξ^⊥(·|s_prop)`。
- `s_goal`：以 ego root 坐标系表示
  - ego 双拳 → 对手躯干的相对向量；
  - 对手双拳 → ego 躯干的相对向量；
  - 使出拳/威胁线索具有朝向不变性。
- AMP：判别器 `C` 区分 `(π_z, D_ψ)` 产生的动作转移与参考动作；输入特征含局部关节旋转/速度、base 角速度；带梯度惩罚；提供 `r_style`。
- 奖励：`R = w_task · r_task + w_style · r_style`；`r_task` 分解为 facing alignment、approach velocity、gated striking distance、contact-validated hit。

### Stage III(b)：Latent-Space Neural Fictitious Self-Play (LS-NFSP)

- 每个 agent 维护两套策略：
  - 最佳响应 `π_RL`：PPO，用 `D_RL` 的最新转移；
  - 平均策略 `π̄`：监督学习 `D_SL`（历史最佳响应 reservoir），`L_SL = E ||π̄(o) - z||²`。
- `D_SL` 增量更新 + reservoir sampling（容量 K），新样本以 `K/n` 概率替换旧样本。
- 动作选择混合：`σ = η · π̄ + (1-η) · π_RL`（Eq. 13；Algorithm 1 用 Bernoulli(η) 在 π_RL/π̄ 间切换，因果方向与 Eq. 13 相反，阅读时需注意这个不一致）。
- 理论上：在 regularity + approximate best-response 假设下，latent 博弈诱导 decoder-reachable 动作流形上的近似 Nash；证明在项目页。
- 关键论点：latent 空间有界，PPO 不会探索到 OOD 物理动作，从而避免 raw 29-DoF 自博弈的平衡崩溃/灾难性遗忘。

## 关键超参数（论文）

| 项 | 值 |
|---|---|
| 机器人 | Unitree G1，29 DoF |
| 仿真 | Isaac Lab，单卡 RTX 4090 |
| 并行环境 | 4,096 |
| 物理/控制频率 | 200 Hz / 50 Hz |
| latent 维度 | 32（单位球） |
| NFSP η | 0.1 |
| reservoir K | 10^6 / agent |
| 任务/风格权重 | `w_task=0.8`，`w_style=0.2` |
| 先验 KL 系数 | `λ_prior=0.001` |
| 训练规模 | 横轴到 10^4 iterations（Fig. 2） |
| Stage II 解码器（论文） | 32 → 128 → 256 → 512 → 29（ELU） |
| Stage III actor（原生 checkpoint） | 192 → 512 → 256 → 128 → 32（ELU） |
| Stage III prior（原生 checkpoint） | 900 → 512 → 256 → 128 → 64（前 32 维作 z_p） |
| Stage III decoder（原生 checkpoint） | 122 (90 proprio + 32 latent) → 512 → 256 → 128 → 29 |

## 主要结果（Table I，节选）

| 方法 | vs Fictitious SP | vs Naive SP | vs PPO-Only | vs Static-Target | vs w/o Warmup | vs 29DoF+AMP | η_hit ↑ | ER ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RoboStriker | 68.52 | 76.24 | 84.47 | 95.38 | 98.50 | 92.50 | **0.685** | **0.824** |
| Fictitious SP | – | 62.35 | 71.84 | 92.11 | 96.18 | 88.14 | 0.420 | 0.650 |
| Naive SP | – | – | 68.45 | 88.61 | 94.52 | 85.25 | 0.350 | 0.580 |
| RoboStriker w/o AMP | – | – | – | 62.12 | 82.14 | 78.40 | 0.490 | 0.720 |
| PPO-Only | – | – | – | – | 45.24 | 68.55 | 0.231 | 0.495 |
| w/o Warmup | – | – | – | – | – | – | 0.050 | 0.120 |
| 29DoF Action-Space SP | – | – | – | – | – | – | 0.142 | 0.315 |
| 29DoF Action-Space + AMP | – | – | – | – | – | – | 0.285 | 0.465 |

要点：
- 去掉 warmup：η_hit 0.05、ER 0.12，几乎不进攻。
- 29-DoF 原始动作空间：η_hit 0.142；加 AMP 提升到 0.285，但仍远低于 latent（0.685）。
- 去掉 AMP：ER 仍有 0.72，但 η_hit 掉到 0.49，且出现不自然姿态。
- 结论：**warmup + latent + AMP 三者互补**；latent 是策略探索空间的结构约束，AMP 是动作风格正则。

## Table II：latent 维度消融

| latent dim | MPKPE (mm) | 对 32 维胜率 | η_hit |
|---:|---:|---:|---:|
| 8 | 128.2 ± 4.4 | 7.1% | 0.328 ± 0.03 |
| 16 | 85.9 ± 1.7 | 23.6% | 0.334 ± 0.03 |
| 32 | 38.5 ± 0.4 | – | **0.685 ± 0.03** |
| 64 | 35.0 ± 0.4 | 39.2% | 0.520 ± 0.08 |

32 维是「动作保真度 vs 策略搜索维度」的折中点；64 维物理误差更低，但策略收敛变差。

## 与 HBOT 项目最相关的 5 点

1. **赛场参考实现已经包含 Stage II**：官方 `MotionPrior` 就是 225 → 32 维单位球 → 225 的参考动作先验/解码器；`SphericalActor` 的 planner 输出 32 维 latent，解码器冻结。我们的 B1/B5 不是「225 直接回归」策略，而是 latent-space 策略。
2. **赛场参考实现已有 Stage III(a) 的一部分**：`train_rl --training-mode standing` 是对站立对手的 warmup；AMP/style reward 已在 `selfplay.json`/`sample.json` 中；历史对手池是冻结策略池。
3. **缺的主要是 Stage III(b) 的 LS-NFSP**：`opponent_pool.py` 只是冻结历史快照，没有平均策略网络、没有 reservoir、没有 η 混合。原生 `model_3000.pt` 里有 `avg_model_state_dict`，证明论文的 NFSP 平均策略确实存在。
4. **底层 29-DoF tracker 不能复现**：比赛规则要求使用指定的 HoloMotion V2 底层控制器，不得替换/修改；论文 Stage I 在赛场是「已给定」的。论文的 29-DoF 动作空间/基线在我们这里不可直接执行。
5. **规模差异巨大**：论文 4,096 envs / RTX 4090 / 10^4 iterations；我们 A10 23GB / 128 envs / 剩余约 14h。只能做**按比例缩小**的复现（相对趋势、消融、native 验收），不能期望复现论文的绝对胜率。

> 复现可行性、规则映射与 B8 实验计划见 [ROBOSTRIKER_REPRO_ASSESSMENT.md](../ROBOSTRIKER_REPRO_ASSESSMENT.md)。
