# HBOT 阶段 4 规格：冻结与最终提交（PHASE4）

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0 / 2026-10-25（PHASE3 完成当日按主方案 §7.3/§7.4 另立） |
| 前置 | [PHASE3_IMPROVEMENT.md](PHASE3_IMPROVEMENT.md) 验收完成；`results/final-candidate.txt` 存在（当前指向基线 `boxing-b1.onnx`） |
| 目标 | ① 完成首个平台提交并收集回放；② 在 12/2 前完成最终模型验证成功；③ 12/3 只确认不提交 |
| 不做 | 排名动作（12/3 前）、无 A/B 证据的新模型提交、12/1 后的任何代码修改 |
| 预算 | 最终复跑（若开）：≈B1 selfplay 实耗 × 1.5 ≈ 0.5h（A10 实测口径）；提交尝试 ≤3 次/天、累计成功 ≤20 |
| 上游文档 | [hbot_boxing_prelim_plan.md](hbot_boxing_prelim_plan.md) §7.3/§7.4/§8；[PHASE2_BASELINE.md](PHASE2_BASELINE.md)（机制沿用）；[PHASE3_IMPROVEMENT.md](PHASE3_IMPROVEMENT.md)（A/B 3 局制沿用） |

**执行方式**：agent 经 `ssh hbot`；提交动作 **【用户手动】**（agent 备料+核对检查单）；评测用 `/mnt/workspace/miniforge3/envs/hbot-eval/bin/python`（tmux 无 conda PATH，实测教训）。

---

## 阶段 0：冻结资产盘点（agent，一次性）

```bash
cd /mnt/workspace/HoloMotion
cat results/final-candidate.txt          # 最终候选 onnx + sha256
cat results/candidates.txt               # 候选闭环清单
sha256sum "$(cat results/current-best.txt)/model_256.pt"
sha256sum results/boxing-b1.onnx
ls -l results/b1-validation.json         # status==PASS 且 chain_benchmark.passed==true
```

- **验收**：`final-candidate.txt` 指向的 onnx 存在、sha256 与 `records/b1-checkpoint-hashes.txt` 一致、validate 双条件 PASS。
- **代码冻结线**：12/1 起不改任何训练/评测代码；12/1 前的修改必须走 PHASE3 附录 B patch 流程（diff→commit→sha256 记录）。
- 盘点结果追加到 `instance_snapshot.md`。

## 阶段 1：训练量决策点【用户决策，agent 备料】

> 实测事实（2026-10-25）：256-update 规模下所有模型本地对局全 0:0、零主动 hit；A10 实测 ≈2.5h/千 selfplay updates。

| 选项 | 内容 | 成本 | 风险 |
|---|---|---|---|
| A（保守） | 直接以 B1 为最终模型走提交节奏 | 0 | 平台上大概率无战绩，Elo 末日常败 |
| B（推荐） | 扩展 selfplay 至 1024–2048 updates（可换 seed 增稳健），A/B 不弱于 B1 后作最终候选 | 2.5–5h GPU（额度足够：剩 ~30h） | 训练崩坏（metrics 监控 + checkpoint 兜底）；仍可能 0:0 |
| C（激进） | B + P2 奖励加大进攻权重（`offensive_hit_reward_punch` 50→60 或 task 0.95） | B + 0.5h | 行为改变不可预期；12/1 前可回退 |

- 任一扩展产物必须走 PHASE3 的 3 局制 A/B（对手 = `current-best.txt`），采纳才可替换最终候选。
- 决策后回填本表并更新 `final-candidate.txt`。
- **【2026-10-25 已执行：选项 B】** 扩展 selfplay 1024 updates（B1 actor 新阶段，interval=32/capacity=8/seed 42）完成（3823s）→ validate PASS/4.83ms → A/B 六局全 0:0 → **REJECT，最终候选维持 B1**。关键实证：strike 奖励 1024 updates 内峰值仅 0.0017（噪声级）、offensive_hit 严格为 0——**训练量不是瓶颈，是奖励/探索结构问题**。选项 C（代码级加大进攻权重）成为下一候选动作，须在 12/1 冻结线前由用户批准执行；证据见 `records/winrate.md`。
- **【2026-10-25 补充实验】** P2-b（代码级 offensive 50→60）REJECT；E1（探索 init_noise_std 0.2→0.5）REJECT。**破局三方向已试二**：训练量✗（256→1280 updates 无效）、探索✗（噪声 ×2.5 无效）。剩余唯一未试：**奖励稠密化/课程（代码级）**——需用户批准，12/1 前执行；否则 B1 即最终模型。

## 阶段 2：提交节奏执行（§7.3 表落进当前日历）

| 窗口 | 提交内容 | 检查单 | 状态 |
|---|---|---|---|
| 10/25–10/31（今天即符合） | **冒烟提交**：当前最终候选（B1）| PHASE2 §7.1 八项 | 【用户手动】待执行 |
| 11 月上旬 | 正式基线提交（若冒烟已成功且无新模型，可跳过——最新验证成功模型即参赛模型） | 同上 | 视冒烟结果 |
| 11 月中下旬 | 扩展训练后的新候选（若阶段 1 选 B/C 且 A/B 采纳） | PHASE3 阶段 7 检查单 | 视阶段 1 |
| 12/1–12/2 | 最终模型提交并**当天验证成功** | 全量检查单 + 灰区条款 | 【用户手动】 |
| 12/3 | 只确认：平台上「最新验证成功」= 最终模型 | 平台页面核对 | 【用户手动】 |

**每次提交后（agent）**：`records/submissions.md` 追加「日期 / 文件名+sha256 / 验证结果 / 当日尝试序号 / 累计成功数 / 备注」；失败按 PHASE2 §7.3 排查序（sha256→错误码→validate 字段→次日重试 ≤1 次）。

## 阶段 3：回放分析与每周循环（agent，随提交驱动）

1. 平台对战公布后：回放风格记录（贴身/游走/重拳）与我方失分模式 → `records/platform_replays.md`。
2. 每周一：查 GPU 额度、磁盘、平台剩余次数；`#/details` 截图存档（`records/rules_snapshot_$(date +%m%d).png`，规则无版本号，靠截图对比察觉变更）。
3. 每周三：本地循环赛刷新胜率表（沿用 `p3_rr.sh`，candidates.txt 自动闭环）。
4. 每周五：决定是否提交新模型（默认不提交，除非 A/B 有采纳级证据）。

## 阶段 4：最终验证（12/1–12/2，agent）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
# 最终候选重导出（保证与 checkpoint 同源）+ 官方口径校验
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$(cat results/current-best.txt)/model_256.pt" --output results/boxing-final.onnx
sha256sum results/boxing-final.onnx
/mnt/workspace/miniforge3/envs/hbot-eval/bin/python -m holoboxing validate results/boxing-final.onnx \
  --output results/final-validation.json --benchmark
python3 -c "import json; r=json.load(open('results/final-validation.json')); cb=r.get('chain_benchmark') or {}; print(r.get('status'), cb.get('passed'), cb.get('mean_ms'))"
```

- **验收**：`status==PASS` 且 `chain_benchmark.passed==true`（≤5ms；5–10 灰区=不通过，例外须用户显式确认并注明）；sha256 记入 `records/final-hashes.txt`。
- 本地卡非 RTX 5090，`mean_ms` 仅参考；平台验证才是最终判据——**12/2 当天必须完成一次平台验证成功**。

## 阶段 5：截止日确认（12/3，用户手动）

1. 平台「我的模型」页确认当前生效模型 = 最终模型（最新验证成功者才参赛）。
2. 不上传任何新文件；截图存档。
3. agent 在 `instance_snapshot.md` 记「PHASE4 收官」。

## 阶段 6：验收表

| 类别 | 检查项 | 通过标准 |
|---|---|---|
| 冻结 | 资产盘点 | final-candidate 指针/哈希/validate 三者一致 |
| 决策 | 阶段 1 训练量 | A/B/C 三选一已记录并执行（选 B/C 则 A/B 采纳证据在 winrate.md） |
| 提交 | 节奏 | 冒烟/基线提交完成；12/2 前最终验证成功；12/3 只确认 |
| 记录 | submissions.md | 每次尝试一行；累计成功 ≤20 |
| 记录 | platform_replays.md | 首次平台对战回放分析归档 |
| 记录 | rules 截图 | 每周一一张，`records/rules_snapshot_*.png` |
| 预算 | 记账 | 全部消耗记入 `instance_snapshot.md` |

## 附录 A：高频故障（承接 PHASE2 附录 C 平台部分）

| 现象 | 处理 |
|---|---|
| 平台验证超时 | 等待不计尝试次数（平台故障退还）；上午提交留排队时间 |
| `MODEL_TOO_LARGE` | 本地 `ls -l` 复核 <50,000,000 字节；超了回训练侧减网络 |
| `CONTRACT_CHECK_FAILED` | 对照 `final-validation.json` 字段找差异；检查 ONNX 算子集 |
| 验证成功但 Elo 全败 | 正常（基线可能偏防守）；按 §7.4 调整训练目标（提高胜率不追求大比分、稳定不倒地优先），回阶段 1 选 B/C |
| 12/2 验证失败 | 次日 12/3 是最后确认日：回退到**已验证成功的旧模型**（B1），不冒险提交新文件 |
