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
| `hbot_boxing_prelim_plan.md` | 上游主方案（§4.4 数据流程、§7 提交/Elo 策略、§9 风险预案） |
| `deployment/` | ModelScope 免费实例的一键部署与自检脚本（Tailscale+SSH 隧道、post_boot 恢复、24 项状态自检） |
| `records/` | 全部实验证据链：55 份验收/校验/对战/A-B 记录、补丁、计时、画像 |
| `models/boxing-h2b.onnx` | 当前最终候选模型（H2b，native 双 KO，2.3MB，sha256 f56fc9d8… 见 records/final-candidate.txt；`boxing-b1.onnx` 为回退候选） |

## 项目状态（2026-10-25 晚）

- **管线全绿**：PHASE0–PHASE3 验收通过；B1 基线 256 updates 复刻完成（validate PASS、4.79ms）。
- **突破（H 系列，2026-10-25）**：奖励重塑（vel_reward 重塑 + active_hit_shaping 门控放宽）后，**H2b 模型双侧 KO 官方原生 RoboStriker 控制器**（1000:5 与 19:1000）——首个会打、能 KO 原生对手的自训模型。当前最终候选 = `models/boxing-h2b.onnx`。
- **方法论教训**：对不还手的 B1 做镜像 A/B 永远 0:0，无法区分「会打/不会打」；`evaluate_native`（对原生控制器）才是有效验收。
- **未解问题**：命中频率仍低（挨 5–19 拳才 KO 对手）；训练行为→确定性对局的迁移机制待解决（见 `records/winrate.md`）。
- 完整证据链见 `records/winrate.md` 与 `deployment/instance_snapshot.md`。

## 安全红线（摘自 PHASE0 附录 C）

SMPL/InterHuman 数据为非商业研究许可、禁止再分发；`incoming/`、`official/` 不入库；Tailscale AuthKey、SSH 私钥、SMPL 模型文件一律不入 Git。
