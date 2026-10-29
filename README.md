# HBOT — 拳击人形机器人自对弈训练项目

基于 HorizonRobotics/HoloMotion 官方框架（holoboxing 子系统）的 G1 机器人拳击竞赛项目：
动作先验 → 监督学习 → RL 热身 → 自对弈 → ONNX 导出 → 平台提交的完整管线。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `PHASE0_SETUP.md` | 远程环境搭建与数据准备规格（阶段 0：连接/环境/资源） |
| `PHASE1_SMOKE_TRAINING.md` | 烟雾训练与导出验证规格（prior→SL→RL→ONNX 全链最小化跑通） |
| `PHASE2_BASELINE.md` | 完整基线 B1 规格（官方参数复刻 256 updates selfplay） |
| `PHASE3_IMPROVEMENT.md` | 改进与 A/B 规格（P1/P2 分支、3 局制 A/B 协议） |
| `PHASE4_FREEZE_FINAL.md` | 冻结与最终提交规格（提交节奏、12/1 冻结线、12/2 验证硬线） |
| `B5_INNOVATION.md` | B5 创新轮规格（文献调研 + 四刀：冲量对齐奖励/长谱系/距离课程/exploiter 联赛） |
| `B5_EXECUTION.md` | B5 执行规格（机制已核实：冲量奖励 patch/续训命令/native 判定矩阵/课程与联赛门控） |
| `B6_DATA_SCALE.md` | 全量现有数据 × 多种子 prior/SL 规格（B6 v1；RL 部分暂缓） |
| `B8_ROBOSTRIKER_REPRO.md` | **主规格（论文方法复现）**：Stage I→II→III(a)→III(b) 全链，规则强制项最小替换，state-conditioned prior + AMP warmup + LS-NFSP |
| `B7_FULL_DATA_SL.md` | B8 的数据/初始化前置：InterHuman 全量 7,810 + RoboStriker 全量 → boxing prepared |
| `ROBOSTRIKER_REPRO_ASSESSMENT.md` | 可行性分析附录（论文→规则映射、已复现/缺口、偏差记录） |
| `B9_MIMIC_PIPELINE.md` | **B9 方案（对标小红书「单卡 7 分钟学拳击」）**：外部拳击动捕 → G1 重定向 → 潜空间建模 → 模仿初始化 → 站立热身 → 交互优化 |
| `colab/B8_T4_TRAIN.ipynb` | Colab 免费 T4 训练 notebook（tmux/nohup/nice/keepalive/分块 resume；B7 SL 可跑，B8 代码条件加载） |
| `deployment/pack_colab_data.sh` | ModelScope 实例上打包 Colab 所需数据/控制器/资产/checkpoint |
| `deployment/shusheng_setup.sh` / `shusheng_run_sl.sh` / `shusheng_run_b8.sh` | **书生·端砚开发机**初始化 + B7 SL 分块续训 + B8 分块训练（全部落 `/data` 持久目录） |
| `colab/B8_T4_TRAIN.ipynb` | Colab T4 备用 notebook（已弃用） |
| `papers/` | RoboStriker 论文 PDF、全文抽取、结构化要点 |
| `hbot_boxing_prelim_plan.md` | 上游主方案（§4.4 数据流程、§7 提交/Elo 策略、§9 风险预案） |
| `deployment/` | ModelScope 免费实例的一键部署与自检脚本（Tailscale+SSH 隧道、post_boot 恢复、24 项状态自检） |
| `records/` | 全部实验证据链：55 份验收/校验/对战/A-B 记录、补丁、计时、画像 |
| `models/boxing-h2b.onnx` | 当前最终候选模型（H2b，native 双 KO，2.3MB，sha256 f56fc9d8… 见 records/final-candidate.txt；`boxing-b1.onnx` 为回退候选） |

## 项目状态（2026-10-29，B12 链路验收完成）

- **B8/B9/B10/B11/B12 全部 0~1 击中**：B8 4,000 updates、B9 CMU 拳击动捕、B10 合成对手、B11 拳击先验、B12 LAFAN1 真实双人对打，native 对 robostriker 的最好成绩是 `model_1024` 的 **1 次命中**；其余全部 0。多局 KO 胜均来自对手自倒。
- **B12 数据与训练链**：LAFAN1 3 段对打 / 27,844 样本；先验 2.98cm → BC 0.202m（600 轮 0.178m）→ 站立热身 32 → selfplay+LS-NFSP/短程 RL。
- **奖励口径已与平台对齐**：训练原用腕→躯干合力模长（无门槛，占 task_reward 95%），平台只算 20ms 法向冲量 ≥5 N·s；已改为法向投影 + 每次命中 ±1、每次被击中 −1。
- **量化结论（见 `records/winrate.md`）**：RL 会把模仿冲掉（0.202m → 2000 轮 0.478m）；BC 0.178m 已优于排除相邻帧的 1-NN（0.414m），即接近当前观测契约的信息上限；对手单局能打中我们 2–11 次，说明判定可达，差距在自身出拳机制。
- **下一步优先级**：① 扩充并重定向更多真实对打片段；② 提升阶段01 潜空间/解码器对高速拳段的拟合；③ 在观测契约内强化对手运动预测。

## 项目状态（2026-10-25 晚）

- **管线全绿**：PHASE0–PHASE3 验收通过；B1 基线 256 updates 复刻完成（validate PASS、4.79ms）。
- **突破（H 系列，2026-10-25）**：奖励重塑（vel_reward 重塑 + active_hit_shaping 门控放宽）后，**H2b 模型双侧 KO 官方原生 RoboStriker 控制器**（1000:5 与 19:1000）——首个会打、能 KO 原生对手的自训模型。当前最终候选 = `models/boxing-h2b.onnx`。
- **方法论教训**：对不还手的 B1 做镜像 A/B 永远 0:0，无法区分「会打/不会打」；`evaluate_native`（对原生控制器）才是有效验收。
- **未解问题**：命中频率仍低（挨 5–19 拳才 KO 对手）；训练行为→确定性对局的迁移机制待解决（见 `records/winrate.md`）。
- 完整证据链见 `records/winrate.md` 与 `deployment/instance_snapshot.md`。

## 安全红线（摘自 PHASE0 附录 C）

SMPL/InterHuman 数据为非商业研究许可、禁止再分发；`incoming/`、`official/` 不入库；Tailscale AuthKey、SSH 私钥、SMPL 模型文件一律不入 Git。
