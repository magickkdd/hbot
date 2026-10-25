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
| `hbot_boxing_prelim_plan.md` | 上游主方案（§4.4 数据流程、§7 提交/Elo 策略、§9 风险预案） |
| `deployment/` | ModelScope 免费实例的一键部署与自检脚本（Tailscale+SSH 隧道、post_boot 恢复、24 项状态自检） |
| `records/` | 全部实验证据链：55 份验收/校验/对战/A-B 记录、补丁、计时、画像 |
| `models/boxing-b1.onnx` | 当前最终候选模型（B1 基线，2.3MB，sha256 见 records/b1-checkpoint-hashes.txt） |

## 项目状态（2026-10-25）

- **管线全绿**：PHASE0–PHASE3 全部验收通过；B1 基线 256 updates selfplay 复刻完成；validate `status=PASS` 且 `chain_benchmark=4.79ms`（≤5ms 判据）。
- **改进探索**（H 系列）：6 个完整训练+判定闭环系统探测了奖励几何空间，行为链（接近→贴身→挥拳→命中）在训练侧逐环点亮（vel 激活 10×、命中首次非零），但确定性对局迁移未打通（详见 `records/winrate.md` H 系列总结）。
- **下一方向**：① 训练/对局行为迁移（采样 vs 均值动作）；② 打破自对弈镜像均衡；③ 教师/课程命中阶段。

## 安全红线（摘自 PHASE0 附录 C）

SMPL/InterHuman 数据为非商业研究许可、禁止再分发；`incoming/`、`official/` 不入库；Tailscale AuthKey、SSH 私钥、SMPL 模型文件一律不入 Git。
