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
