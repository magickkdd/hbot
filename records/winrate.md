# B2 A/B 与循环赛记录（2026-10-25T05:14:26+08:00）

## 确定性核查
- ab-p1-red vs ab-p1-red-repeat：summary 巧合相同（0:0/TIME/零hit），frames 不等 → 确定性假设被推翻，全部 A/B 改每侧 3 局制（已回填 PHASE3 §0.3/附录A + 主方案附录D）

## P1（interval=32，从 B1 actor 新阶段）A/B vs B1 —— 3+3 六局
- 全部 0:0 (TIME)，零 hit/fall/KO；wins 0:0、净胜分 0、falls 0:0 → **REJECT（保守弃）**
- current-best 维持 results/b1-selfplay

## P2（config 级 task 0.9/style 0.1，selfplay-p2.json；起点=B1 actor）A/B vs B1 —— 3+3 六局
- config 生效确认：p2-cfg-smoke/config.json task_reward_weight=0.9, style_reward_weight=0.1
- validate：PASS / passed=true / mean_ms=4.73ms
- 全部 0:0 (TIME)；逐级 tiebreak 全平 → **REJECT（保守弃）**
- current-best 维持 results/b1-selfplay

## P1b（capacity=16/interval=16）
- **门控关闭**：规格要求仅当 P1 被采纳才开启；P1 REJECT → 不跑

## 循环赛（b1/p1/p2 全有序对，6 局）
- 全部 0:0 (TIME)；胜场全 0 → 无单独冠军 → 保守取基线

## 结论
- **改进未奏效，证据归档；最终候选 = 基线 boxing-b1.onnx（sha256 e42e6f6247c6d07ef2372b72c02ddcd87bb1e70ef851c8744c8a054a01b60ebc）**
- 阶段 7 平台提交：不提交新模型，保留 B1 平台模型；submissions.md 待用户首次提交后建档
- 观察：当前训练规模（256 updates）下所有本地对局均为 0:0——模型尚无主动进攻行为；提升战力需要更长训练/更多数据（超出 B2 预算口径，留给 PHASE4/主方案决策）

## PHASE4 阶段 1 选项 B 实测（2026-10-25，2026-10-25T06:39）
- 扩展 selfplay：B1 actor 新阶段，interval=32/capacity=8，128 envs，**1024 updates 完成**（runtime 3823s，stop=完成）
- validate：PASS / passed=true / mean_ms=4.83ms
- **奖励信号事实（1024 updates 精确统计）**：strike_reward 峰值仅 0.0017（噪声级）、complete_punch 峰值 1.4e-06、offensive_hit 严格 =0——探索从未产生行为级出拳；task_reward min/mean/max = 0.008/0.018/0.021
- A/B vs B1（3+3 六局）：全 0:0/TIME → 逐级 tiebreak 全平 → **REJECT**
- **结论**：训练量不是当前瓶颈——官方奖励结构下 256→1280 累计 updates 仍无进攻行为。要出「会打的模型」需奖励/探索结构改动（P2 代码级加大 offensive 权重、或 curriculum/课程引导），属用户决策；12/1 冻结线前可行
- 最终候选维持 **boxing-b1.onnx**（e42e6f62…）

## P2-b 代码级实测（2026-10-25，patch 41b9efae…，分支 p2b-reward@e36823c）
- 改动：offensive_hit_reward_punch 50→60（唯一变量，config=selfplay.json 原版）
- 256/256 updates 完成（995s）；validate PASS / 4.82ms
- **奖励事实：offensive_hit 仍严格 =0、strike 峰值 0.0011（噪声级）——权重 +20% 对未触发信号无效**
- A/B vs B1（3+3）：全 0:0 → **REJECT**；代码已回退 master，结论与分支留档

## 改进菜单穷尽结论（2026-10-25 06:15）
- 已测：P1(interval=32) REJECT｜P2-a(config 0.9/0.1) REJECT｜P2-b(code 50→60) REJECT｜扩展 1024 updates REJECT｜P1b 门控关闭
- 累计证据：1280+ updates 自对弈、18 局本地对战全部 0:0、进攻类奖励从未达行为级
- **结构性结论：官方奖励/探索设计下策略学不会出拳。破局需：① 进攻项权重数量级提升或稠密化（如距离/朝向 shaping）；② 课程（standing→striking 渐进）；③ 提高探索（init_noise_std / entropy 系数）。均超出当前规格菜单，属用户战略决策，须在 12/1 代码冻结线前动手，否则 B1 即最终模型。**

## E1 探索噪声实验（2026-10-25 07:11–07:44，config 级 init_noise_std 0.2→0.5）
- 256/256 updates 完成（995s）；validate PASS / 4.86ms；零倒地（探索加大未失稳）
- **奖励事实：strike 峰值 0.0004（低于基线运行噪声）、offensive 严格 =0**——探索量翻 2.5 倍仍无法发现出拳行为
- A/B vs B1（3+3）：全 0:0 → **REJECT**；current-best/final-candidate 维持 B1
- **破局三方向已试二（训练量✗、探索✗、奖励稠密化未试）**：剩余唯一未试方向 = 奖励稠密化/课程（代码级，需用户批准，12/1 前执行）

## H1：vel_reward 重塑（2026-10-25 09:11–09:46，分支 p4-h1-vel@23cc632，patch 21039ba6…）
- 改动：tar_speed 1.0→0.5、sigma 0.25→0.5（唯一变量，其余同 P2-b 流程）
- 256/256 完成（1190.7s）；validate PASS / 4.80ms
- **训练侧突破（全部实验首次）**：vel_reward mean 0.0047（基线 10 倍，稳定激活）；strike_reward mean 0.0003 / max 0.0074（首次行为级非零）；active_hit_shaping 后期爬升至 0.0005；offensive_hit 仍 0（最后一环未通）
- A/B vs B1（3+3）：全 0:0 → **REJECT**（行为链点亮但 256 updates 未成熟）
- **处置：H1b = --resume 续训 768 updates（保留 H1 对手池与学习状态，累计 1024），验证行为链能否走通最后一环**

## H1b：H1 续训 768（--resume，2026-10-25 09:47–10:47，累计 1024 updates）
- 1024/1024 完成（本段 2946.6s）；validate PASS / 4.75ms
- 奖励链演化：strike 出现负值（挥空被罚=真实挥拳）；complete_punch 微事件 1e-11→1e-06 增长；offensive 仍 0
- A/B vs B1（3+3）：全 0:0 → **REJECT**；已回退 master
- 处置：**H2 = active_hit_shaping 放宽（sigma 1.0→2.0、vel_threshold 0.8→0.4）叠加 H1 改动，从 H1b checkpoint 续训 512 updates**

## H2：shaping 放宽叠加 H1（2026-10-25 10:52–11:50，分支 p4-h1-vel@dcbd9b2，patch 518e0257…）
- 改动：active_hit_shaping sigma 1.0→2.0、vel_threshold 0.8→0.4（叠加 H1 的 vel 重塑）
- 从 H1b checkpoint 新阶段 512/512 完成（1977s）；validate PASS / 4.63ms
- **里程碑：offensive_hit 首次非零（mean 0.00033）**——训练中真实命中发生；shaping 全程 0.002–0.006（基线 10–20 倍）
- A/B vs B1（3+3）：仍全 0:0 → **REJECT**（命中太稀疏，6 局各 60s 未赶上）
- 处置：**H2b = --resume 续训 768（累计 1536），让命中频率成熟后再 A/B**

## H2b：H2 续训 768（累计 1536）+ 对局帧遥测诊断（2026-10-25 11:39–13:00）
- 1280/1280 完成；validate PASS；A/B 仍全 0:0 → REJECT；已回退 master
- 奖励统计：shaping 100% 激活（mean 0.0044）、complete_punch 90% 更新有微事件、offensive 命中率 27%→22% 不升反降（自对弈镜像均衡）
- **帧遥测关键发现**：H2b vs B1 对局全程距离钉在 2.0m（0/3000 帧进入 1.5m）——训练中的「激活」是原地挥手赚 shaping（sigma=2.0 下 2m 处 exp(-1)=0.37 白拿），接近行为未迁移到确定性对局
- **结论：sigma=2.0 制造了对峙舒适区。修正 = H3：sigma 2.0→0.5（梯度集中到接触距离），保留 vel 重塑与 0.4 出手门**

## H3：sigma 收紧到接触距离（2026-10-25 12:48–13:50，分支 0bdb85a）
- 从 H2b checkpoint 新阶段 512/512 完成；validate PASS / 4.72ms
- 奖励统计：shaping mean 0.0010（原地站立不再得分）、offensive nonzero 14%（低于 H2 的 27%）——**sigma 0.5 太陡，梯度够不着**
- A/B vs B1：全 0:0 → REJECT；已回退 master
- 矩阵收敛：sigma 2.0=对峙舒适区、0.5=够不着 → **H4 = 未测点 sigma 1.0（原始）+ vel_threshold 0.4 + H1 vel 重塑**

## H4：sigma 1.0（原始）+ 0.4 门 + H1 vel 重塑（2026-10-25 13:36–14:35，分支 8446657）
- 从 H2b checkpoint 新阶段 512/512 完成（1979s）；validate PASS / 4.67ms
- 奖励统计：vel 100% 激活（mean 0.0048）、shaping 100%（mean 0.0028）、offensive 11%（mean 0.00005）
- A/B vs B1：全 0:0 → REJECT；已回退 master

## H 系列总结论（2026-10-25 15:10，累计 ~2800 updates 实验量）
- **sigma 网格 {0.5, 1.0, 2.0} × 门 {0.4, 0.8} 全部探测完毕**，训练侧行为链逐环点亮（vel 100% 激活、shaping 全程有梯度、挥拳与出拳循环真实发生、offensive 偶发命中），但：
- **命中频率始终停留在 ~1e-4 奖励量级（每 update 摊薄后），从未达到能改变 60s 对局比分的行为密度**；对局帧遥测证实零接近迁移
- 结构性瓶颈（下一轮主攻方向，均为更大的工程改动）：① 确定性对局与随机训练的行为迁移缺口（训练采样 vs 对局均值动作）；② 自对弈镜像均衡抑制接触（双方同步学会同一边缘策略）；③ 缺少「教师强制/课程」阶段（直接从零发现命中太难）
- **最终模型维持 B1**（boxing-b1.onnx，e42e6f62…）：平台对手未知，B1 的「绝对不倒地」在 Elo 口径下仍是当前最优可提交物

## evaluate_native：B1 vs 原生 RoboStriker（model_3000.pt，2026-10-25 14:54）
- 首个外部对手数据点。model_3000.pt（18.8MB，sha256 9f8bb366…）已就位 /mnt/workspace/
- side0（B1 红）：**0:1 负**，裁判判 native 胜；native 命中 1 次、B1 命中 0
- side1（B1 蓝）：**0:0 平**，双方零命中
- 两局均 TIME 结束、零倒地零 KO；B1 无法对原生控制器得分，native 具备真实攻击能力
- 归档：records/native-side{0,1}-result.json

## ★ evaluate_native 验证：H2b/H4 vs 原生 RoboStriker（2026-10-25 14:54–15:05）
- **H2b（vel 重塑 + shaping 2.0 谱系，1280 updates）：双侧双 KO 胜！**
  - side0：1000:5，H2b 41 秒 KNOCKDOWN→KO native（自身挨 5 拳，零倒地）
  - side1：19:1000，native 19 次命中但 H2b 顶住反杀 KO（93s）
  - 1000 = KO 固定奖励分；H2b 成为**首个能 KO 官方原生控制器的自训模型**
- H4（从 H2b 重开新阶段 + σ1.0）：0:16 / 17:0 双败——**新阶段重置清空战斗行为**，印证行为链依附于连续训练谱系
- **方法论修正：对 B1 的镜像 A/B（永远 0:0）无法区分「会打」与「不会打」，evaluate_native 才是有效验收。**
- **处置：final-candidate / current-best 切换至 H2b 谱系**（boxing-h2b.onnx，validate PASS/mean_ms=4.99145231 贴线通过——h2b-validation.json 权威口径，此前误记 4.75，已修正；<50MB）；待平台提交

## H5：defensive_hit_penalty -8→-24（2026-10-25 15:18–16:15，分支 p4-h5-def@47e8726，patch a0793feb…）
- 动机：H2b 挨 5/19 拳，想用「挨打变贵」逼出防守
- --allow-resume-code-change 从 H2b checkpoint 续训 512/512；validate PASS
- **native 验收双败：side0 [1,10]、side1 [7,0]**——进攻被防御压力扼杀，几乎不出拳
- **结论：进攻性与耐打性此消彼长，-8 恰在激进点；H2b（双 KO native）仍为最优，防御优化此路不通**
- 改动已留档分支，master 已回退

- **处置：final-candidate / current-best 切换至 H2b 谱系**（boxing-h2b.onnx，validate PASS/4.75ms，<50MB）；待平台提交

## ★★ KO 归因翻案（2026-10-25 晚，用户视频观察 + match.json 帧证据）
- **用户看 h2b-ko-side0.mp4 后指出：native（striker）是自己倒的，我方效果明显更差——观察成立，证据链如下**
- 分数分解早已暗示：两局 H2b 得分均为 **1000 = KO 固定奖励分**，有效命中数 = **0**（side0 是我们挨 5 拳，side1 挨 19 拳）
- side1（53.04s 倒地）：倒地瞬间 us-native 根部水平距离 **1.40m**（>臂展，物理上不可能由我方击倒）；前 2s native z=0.683 站立正常 → **纯自倒**
- side0（21.68s 倒地）：倒地瞬间距离 0.59m，但我方零有效命中；距离 2s 内 0.96→0.82→0.59m（native 扑击逼近中）→ **大概率扑击失衡自倒**，我方推动贡献无证据
- **H2b 真实画像修正：抗打不倒（两局 z_min 0.68/0.70，挨 19 拳不倒）+ 对 native 零进攻输出**；「首个能 KO 原生控制器」的说法作废，改为「两局皆因对手自倒而获胜（含运气成分）」
- **候选维持 H2b**（vs B1 的 0:1 负 + 0:0 平仍占优：赢是赢，Elo 口径下对手自倒也是胜利）——但取胜依赖对手失误，强队模型不会白送
- **对 B5 的含义：刀③（冲量对齐奖励）恰好补的就是「零真实命中」这一环，优先级结论不变且更紧迫**；另注意 model_3000 自身有平衡缺陷（60s 内两次自倒），平台对手若同源可能共享此弱点——抗打稳定在 Elo 口径下有真实价值

## B5a2 4096 native 验收（2026-10-25 23:32–23:40，boxing-b5a2-4096.onnx，validate PASS/4.84ms）
- 4 局对原生 RoboStriker：side0 0:11 双败；side1 1000:1 双胜（对手 17.14s 自倒，距离 1.92m，我方零命中）
- 有效命中：**4 局我方全部 0 次**；side0 挨 11 拳，side1 仅挨 1 拳
- 归因：side1 两局均为对手自倒；side0 无倒地、判分负
- **结论：B5a2 4096 的 impulse 奖励在训练侧激活（mean 0.37），但未迁移为确定性对局命中；整体不优于 H2b（H2b side0 挨 5 拳后对手自倒，B5a2 side0 直接 0:11 判负）**
- 处置：归档，不替换 final-candidate；B7 全量数据 SL 继续

## B7 全量数据准备收官（2026-10-26 03:00）
- InterHuman 全量：7,810 motions + 7,897 annots，逐文件 md5 与官方 listing 一致；RoboStriker 全量 86 npz / 43 组。
- 拳击 retarget：1,411/1,411；pair XY 误差 2.5e-7；1124/1843/4033 与旧 converted 逐数组一致。
- prepared：官方 prepare 遇 zero-row clip 整批 abort；补丁 `records/b7-prepare-skip.patch` 逐 clip 跳过；合并 **1,227 clip / 741,842 samples**（train 1,139 / 663,216；val 88 / 78,626）；11 个坏 clip：1630/2829/3546/4218/3272/7323/4938/4526/7053/7153/7302；`Records` loader 校验通过。
- manifest sha256 `8a8d75fe…`；compiled 评价包已恢复；B5a2 4096 native 0 命中（见上）。
- 下一场按 B8 主规格：Stage II state-conditioned prior → Stage III(a) warmup → Stage III(b) LS-NFSP；B7 generic retarget/full prior 后置。

## B8 LS-NFSP 长跑（2026-10-27 → 10-28，results/b8-nfsp* model_8114）
- 4,000 updates 分块续训（256 env，selfplay + LS-NFSP，对手池 8 / 刷新 32）：训练侧 impulse 奖励存在，但 native 四局 **我方 0 击中**。
- 归因结论：**缺拳击参考动作**——冻结解码器下，策略只能从零摸索 225 维目标序列，探索效率过低。
- 处置：停止 B8 长跑，改走「外部拳击动捕示范」路线（见下）。

## B9 拳击动捕示范链（2026-10-28，CMU subject 13，方案见 B9_MIMIC_PIPELINE.md）
- 数据：CMU MoCap **subject 13 = boxing**，42 段 AMC；自写 ASF/AMC 解析 + FK（Y-up→Z-up）+ MuJoCo 阻尼最小二乘 IK 重定向到 G1（`deployment/cmu_retarget.py`）。
- 踩坑：`--freeze-lower`（锁下肢+根）后才稳定采到样本；完整重定向会让冻结控制器 2 秒摔倒、采集 0 样本；根高曾错用 `model.qpos0` 导致 -1.99m，改用 initial-state.json 站立 qpos。
- prepared：**36,229 samples**（dataset `cmu-boxing`，`deployment/prep_b9.sh` + `build_sources.py`）。
- 阶段01 先验重建误差 **6.4mm**；阶段02 BC **0.10m**；阶段03 站立热身 32；阶段04 自对弈短程。
- native 四局：**我方 0 击中**，被击中 8/12/7 拳 → 学会抱架、不出拳。

## B10/B11 补丁式尝试（2026-10-28）
- B10：CMU 单角色 + 合成对手（站桩）；BC 误差略升，native 仍 **0 击中**，但挨打次数下降。
- B11：改用拳击专用先验（阶段01 从零拟合）；native 仍 **0 击中**。
- 归因：单角色示范里对手是合成的、几何不真；BC 闭环协变量偏移（训练对手固定，实战对手压上）。

## B12 LAFAN1 真实双人对打（2026-10-28，当前主链）
- 数据：Ubisoft **LAFAN1**（HF 镜像 `lvhaidong/LAFAN1_Retargeting_Dataset`，已重定向到 G1，30fps，CC BY-NC-ND 4.0）；取 3 段格斗（`fight1_s2_s3` / `fight1_s2_s5` / `fightSport1`），7,347 帧。
- 合成双人：两份 CSV 各自已被归零到原点、yaw≈0（相对位姿丢失）→ 保持 A 不动，B 平移 +1.0m、绕 z 转 π（`deployment/lafan_pair.py`）。
- prepared：**27,844 samples**（dataset `lafan-fight`，`deployment/prep_b12.sh`）。
- 阶段01 先验：早停于第 81 轮，验证位置误差 **2.98cm**；阶段02 BC 60 轮：**0.2022m**（此前 0.181m 平台期，加宽网络无效，确认不是容量问题）。
- 阶段03 站立热身 32 次；阶段04 selfplay + LS-NFSP 2,000 updates（256 env，`deployment/b12_nfsp.sh`）。
- **阶段04 训练侧首次出现真实出拳信号**：`impulse_strike_reward` 分段均值 0.009 → 0.034 → 0.145 → 1.0 → 1.25（第 1,163 轮），`task_reward` 1.31；奖励口径为**腕部→对方躯干 20ms 接触冲量**（`rewards.py: impulse_strike_reward`），不是空挥。
- 事故：2026-10-28 16:5x 训练中**并发**跑 native 评测（mujoco-warp 编译），容器掉线，回退到已归档的 `model_1024`；已立规则：训练期间禁止并发评测。
- 待办：从 `model_1024` 续跑剩余 976 轮 → 训练完全退出后再做四局 native 评测（BR/AVG × side0/1，命中按 `hit_count[candidate_side]` 计）→ 渲染视频。

## B12 阶段04 验收与 04b–04e 改进轮（2026-10-28 晚 → 10-29）

**评测口径修正**：训练日志 `reward/<name>` 是 `weight × raw × 0.02`（50Hz）；`impulse_strike_reward` 权重 200 → 实测 1.25 对应原始冲量仅 **0.31 N·s**，而平台计分要求 20ms 窗口法向冲量 **≥5 N·s**（`evaluation/runtime/impulse_hit_rules.py`），且该项占 `task_reward` 的 95%。

| 轮次 | 配方 | 训练侧命中 | native 评测（我方命中） | 模仿误差 |
|---|---|---|---|---|
| 04a（原） | BC 60 轮起步，2000 updates | impulse 1.25（≈1 次/40s） | `model_1024` BR 1 次、AVG 0 次 | 0.202m → 0.418m |
| 04b | ±1 命中奖励（合力模长，权重 20/50/−50） | ≈0.0003/步 | 8 局 **0**（全 KO 胜=对手自倒） | 2000 轮 0.478m |
| 04c | 同上 + 法向投影（v2） | — | 30 分钟后停止（同权重必然漂移） | — |
| 04d | BC 状态重启 + task0.4/style0.6 + v2 | 0 | 1839/2000 中断（`/data` 配额写满） | 1833 轮 0.456m |
| 04e | BC-long(428 轮, 0.178m) → 热身 32 → 短程 RL 400 | 0 | 4 局 **0** | 400 轮 0.311m |

**关键诊断**
- **训练/评测口径不一致**：训练用腕→躯干合力模长，评测只算法向分量；已改为「腕→躯干中心」方向投影（`deployment/apply_scored_hit.py` v2），并补上对称的「被击中 −1」。
- **RL 会把模仿冲掉**：BC 0.202m → 1024 轮 0.418m → 2000 轮 0.478m，腕部最差（远端帧 0.59m）；配置原为 task 0.8 / style 0.2，改成 0.4/0.6 后漂移仅从 0.478 降到 0.456，AMP 风格奖励压不住。
- **BC 精度已近信息上限**：留一近邻（排除同 clip ±60 帧）误差 **0.414m**，比 BC 的 0.178m 更差 → 给定 obs 时目标本身多义，BC 不是欠训练（600 轮只从 0.202 降到 0.178）。
- **对手能打出计分拳**：robostriker 单局命中我们 2/6/7/11 次，说明平台物理与判定可达 5 N·s；我们 0 次，差距在自身出拳机制。

**结论**：0 命中既不是奖励口径、也不是训练量或算力问题；在「冻结 HoloMotion V2 + 225 维目标序列」接口下，可复现的示范精度（0.18m）与由此得到的拳速，不足以稳定产生 ≥5 N·s 的计分冲量。后续若要突破，优先级为：① 扩充并重定向更多真实对打片段（当前仅 3 段/27,844 样本）；② 提升潜空间/解码器对高速拳段的拟合（阶段01 重建 2.98cm 仍偏粗）；③ 在观测契约内引入对手运动预测（当前 10 帧历史 + 当前对手状态）。
- 产物归档：`/data/checkpoints/latest/{b12-nfsp,b12-04b,b12-04d,b12-04e}`（各保留最新 2 份）。
