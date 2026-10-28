# HBOT B5 执行规格 v2.1：H2b 谱系 × 冲量对齐奖励

> 版本 v2.1 · 2026-10-25 深夜（v2.0 全步骤脚本化 → v2.1 外部评审 18 条修订：完成判定字段/0.02 缩放权重/相对门控/新目录续块/事件级归因与判定/原子 patch）
> 上游：`B5_INNOVATION.md`（文献调研与立项，结论已与本版同步）；本文档 = 唯一执行依据。
> 硬线：**12/1 代码冻结（硬）、12/3 初赛截止**；11/24 = 新开分支软线（过了只续不新）。提交窗口 10:00–18:00、3 次/日、20 次累计成功。
> 额度：**总 16h，两次开机 × 8h**（2026-10-25 用户口述，已记 `deployment/instance_snapshot.md`）。

---

## 0. 定案（全部写死，执行期不再讨论）

| 决策 | 结论 | 依据 |
|---|---|---|
| 谱系 | **基于 H2b（model_1280.pt）`--resume` 续训，禁止新开阶段** | H4 新阶段重置 → native 双败；H5 证明代码级 resume 可行 |
| 算法 | **PPO+AMP 不换** | striker 同家族同接口，方法天花板 ≥ striker；16h 装不下新栈 |
| 主 KPI | **我方有效命中数（HIT 事件口径）**；胜负/失分作辅助条件 | 翻案教训：native 会自倒，胜负含运气（winrate.md ★★ 段） |
| 目标 | 正常战胜 striker：命中数压过它、KO 由我方打出（causal KO 为追踪指标，命中压制为采纳门槛，见 §4.5） | 用户 2026-10-25 定 |
| 防御/探索/σ | defensive -8 不动；init_noise_std 不动；σ 网格不再扫 | H5 / E1 / H2-H4 教训 |
| 验收 | evaluate_native 双侧×2 局 + **事件级倒地归因**（fall_attrib.py v2.1） | 18 局镜像 0:0 无信息量 |
| 重新来（fresh） | 仅当：刀③两轮双负 **且** 奖励链熄火 **且** 剩额 >10h 三条同时成立 | 届时另写规格 |

**执行方式**：agent 经 `ssh hbot` 操作；训练全部在 tmux 会话 `hbot` 内；**本文档所有命令块默认已执行 `cd /mnt/workspace/HoloMotion && source env.sh`，新开 shell/tmux 窗口必须先补这一行**（各独立块已尽量自带前缀）。训练用 `.venv-holoboxing/bin/python`，评测用 `conda run -n hbot-eval`。命名：`b5-impulse-smoke`、`b5a-selfplay[-rN]`、`b5a2-selfplay[-rN]`、`b5-near-selfplay`、`b5-exp-selfplay`、`boxing-b5a.onnx` 等。

---

## 1. 已核实机制基础（全部本地代码核实，2026-10-25；v2.1 补 G 系列）

| # | 结论 | 出处 |
|---|---|---|
| F1 | 训练物理每步算逐接触力 + 6 步历史：`net_history (W,6,2,30,3)`、`paired_history (W,6,2,2,3)` | `training/envs/mujoco_physics.py:246-248,405-408` |
| F2 | 成对腕→躯干力矩阵：**头/手套经 weld 刚体并入，无需改 schema** | `mujoco_physics.py:228-241` + 内核 :87-115 |
| F3 | 20ms 冲量窗 = 恰好一个控制步（4×5ms）；history `[:, :4]` = 本步完整因果窗 | `mujoco_physics.py:405-408`；评测 `impulse_hit_rules.py:8-11` |
| F4 | reward 层访问范例：读**对手**的 `contact_sensor_{对方}_torso`（`[W,6,1,2,3]`） | `tasks/combat/mdp/rewards.py:25-44`、`mujoco_env.py:305-309` |
| F6 | 初始站位 = 构造期 `reset_state` 快照；自由关节 qpos 地址存于 **`self.root_qi`** | `mujoco_physics.py:200,253-262,429-438` |
| F7 | 对手池无外部注入 API；快照随 checkpoint 持久化 | `training/opponent_pool.py:22-73` |
| F8 | 已知偏差（接受）：paired 力取模含摩擦、多接触可相消——对训练信号足够 | `impulse_hit_rules.py:52-66` vs `rewards.py:41-43` |
| F9 | metrics：每 update 一行 JSON（`iteration` 为**累计值**，resume 后从 1281 起） | `on_policy_runner.py:527-544`；b4-h2b 存档 `total_updates=1280` 佐证 |
| G1 | **complete.json = `{updates_completed: runner.iteration(累计), training_result: result, …}`，顶层无 stop_reason**；stop_reason 在 `training_result.budget.stop_reason` | `training/scripts/train_mujoco.py:148-156`（v2.1 核实） |
| G2 | **输出目录必须不存在或为空**：`prepare()` 遇非空目录 raise "Use a new empty output directory" → 续块一律新目录 `-rN` | `training/scripts/common.py:13-16`（v2.1 核实） |
| G3 | **奖励实际执行 `weight × fn × 0.02`**（0.02 = 步长缩放）；所有权重推导必须按有效值算 | `training/envs/mujoco_env.py:595`（v2.1 核实） |
| G4 | match.json 事件：**`HIT{actor,target,hand,peak_force,…}`**（命中直接数事件）；`KO{actor}` 的 actor 是**受益方非因果方**（对手自倒时也标我方）；`FALL{target}` 无 actor | D 盘两局实测（v2.1 核实） |

---

## 2. 阶段 0：开工门与基线补测（T+0:00 – T+1:00）

### 2.1 连接（用户开机后）

用户操作：ModelScope 控制台启动实例 → WebIDE 终端粘贴：

```bash
nohup bash /mnt/workspace/.remote-access/post_boot.sh > /mnt/workspace/.remote-access/post_boot.log 2>&1 &
```

然后对 agent 说「按 PHASE0 连接」。恢复后确认 `tmux ls`（无则 `tmux new -s hbot -d`）。

### 2.2 资产盘点（五项全过才继续）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
sha256sum results/h2b-selfplay/model_1280.pt   # = 019b5a375d9c588c48aec0b78f22506629d5df171683745dbdb7288adc6033d1
sha256sum /mnt/workspace/model_3000.pt          # = 9f8bb366cdb3fd9d2752d7f7aed67a7391d9ca58aadf361f0f569b35ed54f8bc
git log --oneline -3 p4-h1-vel                  # 含 dcbd9b2（H2 提交 = H2b 代码基座）
git status --short                              # master 必须干净（H5 已回退）
```

**第五项（v2.1 新增）——checkpoint 内部校验**（resume 要求 config/协议完全一致，提前到盘点暴露）：

```bash
.venv-holoboxing/bin/python - <<'PYEOF'
import torch
ck = torch.load('results/h2b-selfplay/model_1280.pt', map_location='cpu', weights_only=False)
print('top keys:', list(ck)[:12])
it = ck.get('iteration', None)
if it is None and isinstance(ck.get('runner'), dict):
    it = ck['runner'].get('iteration')
print('iteration =', it)
assert it == 1280, '起点 iteration 不是 1280，停止并人工检查'
cfg = ck.get('config') or {}
print({k: cfg.get(k) for k in ('num_envs', 'seed') if k in cfg})
# 字段名以实际打印为准；若 config 含 opponent 池相关字段（capacity/interval），核对 8/32
PYEOF
```

### 2.3 基线补测：H2b vs native ×4 局（side0/1 各 2，~25 分钟）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
for SIDE in 0 1; do for R in a b; do
  conda run -n hbot-eval python -m holoboxing.training.evaluate_native \
    --candidate results/boxing-h2b.onnx \
    --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
    --opponent-sha256 "$ROBOSTRIKER_SHA256" \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --candidate-side $SIDE --output results/native-b5base-side$SIDE-$R --device cuda:0
done; done
```

逐局跑 `fall_attrib.py`（§4.5）归因，回填基线表。**候选验收同为每侧 2 局（§4.5），口径一致**；winrate 已记录同侧重复局 frames 不同（非确定性），重复局提供独立样本。

| 基线（4 局合计） | 现值（2 局） | 补测后 |
|---|---|---|
| 我方有效命中（HIT 事件） | 0 | ___ |
| native 自倒局数 / 对方命中 | 2 / 24 | ___ |
| 我方倒地 / 失分 | 0 / 24 | ___ |

---

## 3. 阶段 1：刀③ patch 与烟雾（T+1:00 – T+1:30）

### 3.1 改动内容

`rewards.py` 新增（插在 `def reset_punch_state` 之前）：

```python
PHYSICS_DT = 0.005
IMPULSE_WINDOW_STEPS = 4  # 4 x 5ms = 20ms，对齐评测 impulse_hit_rules


def impulse_strike_reward(env, sensor_cfg, asset_cfg=None):
    """本控制步 20ms 内 腕(含手套)->躯干(含头) 实测接触力冲量，N·s，双手取 max。

    有效奖励 = weight * 本值 * 0.02（mujoco_env.py:595）。
    双手取 max 而非求和：双轻触不得翻倍，语义对齐"单次有效出拳"。
    """
    force = env.scene.sensors[sensor_cfg.name].data.force_matrix_w_history  # [W,6,1,2,3]
    per_hand = (force[:, :IMPULSE_WINDOW_STEPS, 0].norm(dim=-1) * PHYSICS_DT).sum(dim=1)  # [W,2]
    return per_hand.max(dim=-1).values
```

`mujoco_rewards.py` terms() 中**整个 `offensive_hit_reward_punch` 元组块**替换为：

```python
        (
            "impulse_strike_reward",
            200.0,
            rewards.impulse_strike_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + other + "_torso"),
            ),
        ),
```

**权重依据（v2.1 按 G3 有效奖励重算，含 0.02 缩放）**：

| 事件 | 冲量 I | 有效奖励 = 200×I×0.02 | 对照 |
|---|---|---|---|
| 旧 offensive 二值触发一次 | — | 50×1×0.02 = **1.0** | 基准 |
| 轻触 | 0.25 N·s | 4×0.25 = **1.0** | 与旧事件等值（保底不降） |
| 半力命中 | 1 N·s | **4.0** | 旧值 4 倍 |
| 达标命中（评测计分线） | 5 N·s | **20.0** | 旧最大值的 20 倍 |

w=20（v2.0）的有效奖励只有 0.4×I，轻触 0.1——比旧值弱 10 倍，有熄火风险，作废。升降档（C-F2）同步按有效值：熄火→w=400（0.125 N·s 即达旧值）；失稳→w=100（5 N·s 得 10）。

### 3.2 远程应用脚本（先全量校验后统一写盘，失败零写入）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
git checkout -b b5-impulse p4-h1-vel@dcbd9b2
.venv-holoboxing/bin/python - <<'PYEOF'
import re

rp = 'holoboxing/training/tasks/combat/mdp/rewards.py'
mr = 'holoboxing/training/envs/mujoco_rewards.py'
s1, s2 = open(rp).read(), open(mr).read()

# --- 全部校验在内存完成，任一失败不写任何文件 ---
assert 'impulse_strike_reward' not in s1, 'already patched?'
assert 'def reset_punch_state' in s1, 'anchor missing in rewards.py'
pat = re.compile(r'        \(\n            "offensive_hit_reward_punch",.*?\n        \),\n', re.S)
assert len(pat.findall(s2)) == 1, 'offensive block count != 1'

fn = '''

PHYSICS_DT = 0.005
IMPULSE_WINDOW_STEPS = 4  # 4 x 5ms = 20ms, aligned with evaluation impulse_hit_rules


def impulse_strike_reward(env, sensor_cfg, asset_cfg=None):
    """20ms wrist->torso(+head via weld) measured impulse, N*s, max over hands."""
    force = env.scene.sensors[sensor_cfg.name].data.force_matrix_w_history  # [W,6,1,2,3]
    per_hand = (force[:, :IMPULSE_WINDOW_STEPS, 0].norm(dim=-1) * PHYSICS_DT).sum(dim=1)  # [W,2]
    return per_hand.max(dim=-1).values

'''
new_s1 = s1.replace('def reset_punch_state', fn + '\ndef reset_punch_state', 1)
new_block = '''        (
            "impulse_strike_reward",
            200.0,
            rewards.impulse_strike_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + other + "_torso"),
            ),
        ),
'''
new_s2 = pat.sub(new_block, s2, count=1)

# --- 校验全过，统一落盘 ---
open(rp, 'w').write(new_s1)
open(mr, 'w').write(new_s2)
print('PATCH_OK')
PYEOF
git diff --stat        # 应恰好 2 个文件；人工过目 git diff 后：
git add -A && git commit -m "B5 knife-3: impulse_strike_reward (20ms, N.s, max-hands) replaces offensive_hit"
git diff p4-h1-vel@dcbd9b2 HEAD > /tmp/b5-impulse.patch
sha256sum /tmp/b5-impulse.patch | tee results/b5-impulse.patch.sha256
```

### 3.3 烟雾（8 updates）与验收

```bash
cd /mnt/workspace/HoloMotion && source env.sh
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --resume results/h2b-selfplay/model_1280.pt --allow-resume-code-change \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 \
  --output results/b5-impulse-smoke \
  --num-envs 128 --updates 8 --seed 42 --device cuda:0 \
  --max-runtime-seconds 1800
```

| # | 验收 | 判法 |
|---|---|---|
| 1 | 无 crash/NaN，完成 | **`complete.json.training_result.progress.completed_updates == 8`**（本次增量；`updates_completed` 是累计 iteration=1288，不作 8 判据——G1） |
| 2 | config 覆盖生效 | 只比 selfplay.json 里**出现的覆盖字段**（完整 TrainConfig 不可能与部分覆盖文件全等）；见下方比对脚本 |
| 3 | 新项出现且有限 | metrics.jsonl 有含 `impulse` 的字段，值全有限 |
| 4 | （软观察）非零情况 | 8 updates 样本小，全零**不算失败**；非零统计移到门 1（Δ100）判 |

```bash
.venv-holoboxing/bin/python - <<'PYEOF'
import json
over = json.load(open('holoboxing/training/configs/selfplay.json'))
out = json.load(open('results/b5-impulse-smoke/config.json'))
def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a
exp = merge({}, over)
bad = [k for k, v in exp.items() if k in out and out[k] != v]
missing = [k for k in exp if k not in out]
print('mismatch:', bad, '| missing:', missing)
assert not bad, 'config override mismatch'
PYEOF
```

---

## 4. 阶段 2：B5a 主线 1280→2560（T+1:30 – T+3:30）

### 4.1 启动（tmux 会话 hbot 内）

```bash
cd /mnt/workspace/HoloMotion && source env.sh
tmux new -s hbot 2>/dev/null || tmux attach -t hbot
# tmux 内（首块）：
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --resume results/h2b-selfplay/model_1280.pt --allow-resume-code-change \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 \
  --output results/b5a-selfplay \
  --num-envs 128 --updates 1280 --seed 42 --device cuda:0 \
  --max-runtime-seconds 7200
```

预算：1280 × 3.79s ≈ **81 分钟**。断点续块模板（**G2：输出目录必须新**，绝不复用旧目录）：

```bash
# 下一块目录序号 = 现有最大 rN + 1；首块无后缀，续块 -r2、-r3…
N=$(ls results/b5a-selfplay*/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
LATEST=$(ls -t results/b5a-selfplay*/model_*.pt | head -1)
NEXT=r$(( $(ls -d results/b5a-selfplay* | sed 's/.*-r//' | sort -n | tail -1) + 1 ))
.venv-holoboxing/bin/python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --resume "$LATEST" --allow-resume-code-change \
  --config holoboxing/training/configs/selfplay.json \
  --training-mode selfplay --opponent-pool-capacity 8 \
  --opponent-refresh-updates 32 \
  --output results/b5a-selfplay-$NEXT \
  --num-envs 128 --updates $((2560-N)) --seed 42 --device cuda:0 \
  --max-runtime-seconds 7200
```

### 4.2 监控 + 两道 early-kill 门（**相对本次启动的增量 Δ**，F9：iteration 为累计值从 1281 起）

```bash
watch -n 60 'D=$(ls -dt /mnt/workspace/HoloMotion/results/b5a-selfplay* | head -1); \
  echo "DIR=$D"; ls -t $D/model_*.pt 2>/dev/null | head -3; tail -1 $D/metrics.jsonl'
```

```bash
cd /mnt/workspace/HoloMotion && .venv-holoboxing/bin/python - <<'PYEOF'
import json, glob
d = sorted(glob.glob('results/b5a-selfplay*/metrics.jsonl'), key=lambda x: open(x).readline() and x)[-1]
rows = [json.loads(l) for l in open(d)]
start = rows[0]['iteration'] - 1          # 本次启动前的累计 iteration
delta = rows[-1]['iteration'] - start     # 本次已跑增量 Δ
last = rows[-1]
keys = [k for k in last if 'impulse' in k.lower()]
print(f'DIR={d} Δ={delta} (iteration {rows[-1]["iteration"]})')
if keys:
    k = keys[0]
    vals = [r.get(k, 0) for r in rows]
    nz = sum(v != 0 for v in vals)
    print(f'{k}: nonzero {nz}/{len(vals)} ({100*nz//max(1,len(vals))}%)')
    # 反推平均冲量（N·s）：term_mean / (w*0.02)，w=200
    print(f'平均冲量≈{sum(abs(v) for v in vals)/max(1,len(vals))/4.0:.4f} N·s')
PYEOF
```

| 门 | 时机（**Δ = 本次增量**） | 判据 | 不过怎么办 |
|---|---|---|---|
| 门 1 | Δ ≈100（约 7 分钟） | impulse 字段存在且历史有非零 | 立即 `kill -TERM`，按 C-F1 查；止损 ≤10 分钟 |
| 门 2 | Δ ≈300（约 19 分钟） | nonzero% 有爬升迹象；shaping/vel 不崩（附录 A） | 停止，按 C-F2 升降权重重试；止损 ≤25 分钟 |
| 全过 | — | 允许跑满（本段 1280） | H 系列教训：方向错 256 updates 才知道；两道门压到 Δ100/Δ300 |

**健康画像（口径修正）**：impulse nonzero% 是**接触频率诊断**（含轻触/摩擦，无 5 N·s 门槛，不等于有效命中率）；平均冲量由 term/(200×0.02) 反推。**合法命中只能由 native 的 HIT 事件判定**（G4）。对照附录 A：nonzero% 目标 ≥30% 且升（H2b offensive 22% 且降）；shaping 持平 0.0044；vel 持平 0.0048；零倒地。

### 4.3 中断恢复（SSH 掉线 / tmux 丢失）

```bash
tmux ls; ps -ef | grep train_rl | grep -v grep; nvidia-smi
```

进程在 → `tmux attach -t hbot` 等待；不在 → §4.1 续块模板（新目录）从最近 checkpoint 恢复；恢复后必查上一块 `training_result.budget.stop_reason`。

### 4.4 完成、导出与验证

```bash
cd /mnt/workspace/HoloMotion && source env.sh
# 完成判定（G1 口径）：
#   最新块 complete.json.updates_completed == 2560（累计 iteration）
#   且 complete.json.training_result.budget.stop_reason == "requested_progress_complete"
FINAL_DIR=$(ls -dt results/b5a-selfplay* | head -1)
.venv-holoboxing/bin/python -m holoboxing.training.export \
  --checkpoint "$FINAL_DIR/model_2560.pt" --output results/boxing-b5a.onnx
conda run -n hbot-eval python -m holoboxing validate results/boxing-b5a.onnx \
  --output results/b5a-validation.json --benchmark
# 双门槛：status == PASS 且 chain_benchmark.passed == true（mean_ms <= 5.0；H2b 4.99 贴线，逐轮必复测）
```

### 4.5 native 验收（每侧 2 局，与基线同口径）+ 事件级归因 + 两级判定

```bash
cd /mnt/workspace/HoloMotion && source env.sh
export ROBOSTRIKER_CHECKPOINT=/mnt/workspace/model_3000.pt
export ROBOSTRIKER_SHA256="$(sha256sum "$ROBOSTRIKER_CHECKPOINT" | cut -d' ' -f1)"
for SIDE in 0 1; do for R in a b; do
  conda run -n hbot-eval python -m holoboxing.training.evaluate_native \
    --candidate results/boxing-b5a.onnx \
    --opponent-checkpoint "$ROBOSTRIKER_CHECKPOINT" \
    --opponent-sha256 "$ROBOSTRIKER_SHA256" \
    --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
    --candidate-side $SIDE --output results/native-b5a-side$SIDE-$R --device cuda:0
done; done
# 逐局归因（match.json 在各输出目录内）：
python3 deployment/fall_attrib.py results/native-b5a-side0-a/match.json 0
# …side0-b/side1-a/side1-b 同理（本地 scp 上去的 fall_attrib.py v2.1）
```

`fall_attrib.py` v2.1 输出（已用历史两局回归验证）：**HIT 事件级命中数（我方/对方/峰值力）**、双方倒地次数、我方 z_min、对手每次倒地的因果判定（距离 >1.0m=自倒；<0.7m 且倒地前 2s 我方有 HIT=我方打出；否则存疑）。注意 **KO 事件的 actor 是受益方非因果方**（G4），不作归因依据。

**两级判定（输入 = 4 局合计，事件口径）**：

| 级 | 条件 | 动作 |
|---|---|---|
| **生效性**（是否继续堆量） | 我方 HIT 合计 ≥3（无论胜负） | 续 §4.6 4096 |
| | 我方 HIT 1–2 且不劣于基线 | 续 4096 再判 |
| | 我方 HIT 0 且战绩不劣于基线 | 运气局 → C-F2 调权重重试一轮 |
| | 我方倒地 >0 或失分较基线大增 | **REJECT** → 谱系留档回 H2b 出口 |
| **采纳**（替换候选，需先过 validate 双门槛） | 我方 HIT > 对方 HIT **且** 败局 = 0 **且** 我方倒地 = 0 | 更新 `final-candidate.txt` |

**causal_ko_by_us（我方打出的 KO）为终局目标追踪指标**：fall_attrib 逐局记录，不设门槛——命中压制是采纳条件，KO 会随命中密度自然到来（目标表述见 §0）。

### 4.6 续 4096（条件：生效性达标，且剩余额度 ≥4.5h）

同 §4.1 续块模板：从 2560 checkpoint 起，`--updates 1536`（≈97 分钟），输出 `results/b5a2-selfplay`；导出 `boxing-b5a2.onnx` + validate + native 4 局 + 归因，同 §4.4–4.5 两级判定。

---

## 5. 阶段 3：刀② 近距课程（第一场剩 ≥1.8h 做 pilot（0.8h+1.0h 缓冲）；完整轮放第二场）

### 5.1 改动（分支 `b5-near`，自 b5-impulse 采纳提交切出）

`mujoco_physics.py` 构造函数 `reset_state` 建好后追加（**用现有属性 `self.root_qi`**，F6）：

```python
# B5 knife-2: constant 1.2m spawn (was ~2.0m); generalization judged by native match
qx = self.reset_state["qpos"].clone()
for root in self.root_qi:      # mujoco_physics.py:200，两个 free joint 的 qpos 地址
    qx[0, root + 0] *= 0.6
self.reset_state["qpos"] = qx
```

**核实步（必做，C-F4）**：先跑 1-update 烟雾打印 `self.root_qi` 实际地址与 reset 后双方 root 水平距离（未 patch 应 ≈2.0m；patch 后 ≈1.2m）。距离不对 = 索引错，禁止开训。

### 5.2 训练与验收

从 B5a 系列最优 checkpoint `--resume` + `--allow-resume-code-change`，512 updates（≈32 分钟），输出 `results/b5-near-selfplay`；导出/validate/native 4 局/归因同 §4.4–4.5。**泛化判定**：比赛开局 2.0m，native 劣化即弃；独立分支，谱系不受损。

---

## 6. 阶段 4：刀④ exploiter 联赛（门控：第二场且剩 >5h；否则整刀跳过）

### 6.1 代码改动（`opponent_pool.py` 两处，分支 `b5-league`）

1. `inject_external(actor_sd, norm_sd, iteration)`：按 `_snapshot()` 格式 append 外部权重（算 sha256）入池；
2. 构造参数 `fixed_external=None`：`prepare()` 直接 install 指定外部快照、跳过自家快照与随机选择。

### 6.2 执行序列

1. **exploiter 训练**：从主线最优 checkpoint `--resume` + fixed_external（对手恒为主模型）+ `--allow-resume-code-change`，512 updates，输出 `results/b5-exp-selfplay`（新目录，G2）；
2. **注入**：`inject_external(exploiter 最优)` → 主线 resume 256 updates（新目录）；
3. **红线：exploiter 永不导出为提交候选**（过拟合单对手，Elo 口径废棋）；
4. 验收：主线新模型 native 4 局 + 归因（§4.5 两级判定）；另打一场 主线 vs exploiter 本地局作诊断（不入判定）。

---

## 7. 会话规划：两场 8h 剧本与砍单顺序

**总原则：两次开机 = 两个决策点；材料全部离线备好（本文档 + `fall_attrib.py`）；训练不盲跑，两道 early-kill 门（Δ100/Δ300）随时止损；步骤背靠背执行，不留空闲烧额度。**

### 第一场（8h）——主线日

| 时段 | 内容 | 估时 |
|---|---|---|
| T+0:00 | §2 连接 + 五项盘点 + 基线补测 ×4 | 1.0h |
| T+1:00 | §3 patch + 烟雾三条硬验收 | 0.5h |
| T+1:30 | §4 B5a 1280→2560（守门 Δ100/Δ300） | 1.4h |
| T+2:55 | §4.4–4.5 导出/validate/native 4 局/归因/两级判定 | 0.7h |
| T+3:30 | 生效 → §4.6 续 4096（1.7h + 0.7h 验收）；未生效 → C-F2 调权重重试（1.4h + 0.7h） | 2.4h |
| 尾声 | **若剩 ≥1.8h**（多见于"调权重重试"路径；4096 主路径通常已到 T+5:45+，剩 <2h）→ §5 刀② pilot；否则收工关机，pilot/4096 滚入第二场 | ≤0.8h |

最坏 ≈6.5h，典型 ≈5h，预留 ≥1.5h 机动。**刀② pilot 在 4096 主路径上大概率顺延到第二场——这是砍单顺序的预期行为，不是事故。**

### 第二场（8h）——机动日（隔 1–2 天，按第一场结论定内容）

优先级：① 4096 续跑（若未完成）→ ② 刀② 完整轮（若 pilot 有苗头或未做）→ ③ 谱系加深 4096→6144（2.2h/2048，仅当命中数仍爬升）→ ④ 刀④（仅当主线已采纳且剩 >5h）。
**固定保留 1.5h（不可侵占）**：最终候选重导出 + validate 复测 + 记录归档 + `models/` 镜像。

### 砍单顺序（额度吃紧时从上往下砍）

1. 刀④；2. 刀② 第二轮回调；3. 4096（仅当 2560 已生效；未生效的 4096 没有意义）；
4. **不可砍**：刀③ + 2560 主线 + validate + native 验收。

---

## 8. 阶段 5：记账、归档与提交

### 8.1 每阶段归档清单（scp 回 `records/`）

| 阶段 | 必归档 |
|---|---|
| §3 | `b5-impulse.patch` + sha、smoke `complete.json` + `metrics.jsonl` |
| §4 | 每块 `complete.json`（含 training_result）、`metrics.jsonl` 尾部、`boxing-b5a.onnx` sha、`b5a-validation.json`、native 4 局 `result.json`+`match.json`、归因输出 |
| §5/§6 | 同 §4 模式 + 分支/patch sha |

### 8.2 记账模板（`winrate.md` 追加，B5a 示例）

```markdown
## B5a：冲量对齐奖励（2026-10-XX，分支 b5-impulse@<commit>，patch <sha256>）
- 改动：offensive_hit_reward_punch 50.0(有效1.0/次) -> impulse_strike_reward 200.0(有效4×I N·s, 双手max)（唯一变量，基座 p4-h1-vel@dcbd9b2）
- 奖励统计：门1@Δ100 nonzero __%；门2@Δ300 __%；出口 impulse nonzero __% / 平均冲量 __ N·s / shaping 0.00__ / vel 0.00__
- native 4 局：HIT 我方__:对方__；我方倒地__；胜负 W-L-D __；causal_ko_by_us __
- 两级判定：生效性 __ / 采纳 __；处置：__
```

### 8.3 采纳流程与提交纪律

- 采纳 = 更新 `records/final-candidate.txt`（路径 + sha + 一行依据）+ `candidates.txt` 追加 + winrate.md 记账 + `models/` 镜像；
- **H2b 在下一窗口（10:00–18:00）先行提交，不等 B5**；此后每个被采纳模型在下一窗口提交；
- 每日 ≤3 次、累计 ≤20 次成功；validate 双门槛不过的模型坚决不上传；上午提交留排队余量。

---

## 附录 A：H2b 出口基线表（监控与判定对照）

| 指标 | 值 | 出处 |
|---|---|---|
| offensive_hit nonzero% | 27%→22%（降，镜像均衡征兆） | winrate.md H2b 段 |
| active_hit_shaping mean | 0.0044（100% 激活） | 同上 |
| complete_punch 微事件 | 90% updates 有 | 同上 |
| vel_reward mean | 0.0048（100% 激活） | winrate.md H4 段 |
| 对局距离 | 钉 2.0m（0/3000 帧 <1.5m） | H2b 帧遥测 |
| native（HIT 事件口径） | 我方命中 0/0 局、对方 5+19；对方倒地均为自倒/存疑（1.44m / 0.61m 无命中佐证） | winrate.md ★★ 段 + fall_attrib.py v2.1 回归 |
| validate | PASS，mean_ms 4.99145231（贴线） | h2b-validation.json |

## 附录 B：机制出处汇总（全部本地核实）

| 机制 | 位置 |
|---|---|
| 评测冲量规则（20ms/5N·s/200ms 冷却） | `evaluation/runtime/impulse_hit_rules.py:8-11,52-66` |
| 评测合法手/目标 | `evaluation/runtime/hand_hit_rules.py:56-75` |
| 训练成对力管道（含头/手套 weld 合并） | `training/envs/mujoco_physics.py:87-115,228-241,246-248,405-408` |
| 传感器暴露 | `training/envs/mujoco_env.py:305-309` |
| **奖励 0.02 步长缩放** | `training/envs/mujoco_env.py:595` |
| **complete.json 结构（updates_completed=累计 iteration；stop_reason 在 training_result.budget）** | `training/scripts/train_mujoco.py:148-156` |
| **输出目录必须为空** | `training/scripts/common.py:13-16` |
| **root_qi（自由关节 qpos 地址）** | `training/envs/mujoco_physics.py:200` |
| 初始站位构造/回放 | `mujoco_physics.py:253-262,429-438` |
| 对手池（无注入 API） | `training/opponent_pool.py:22-73` |
| resume 代码一致性 | `training/rsl_rl/runners/resume.py:314-340` |
| 5ms 本地门槛 | `evaluation/runtime/chain_benchmark.py` |
| metrics 每 update 一行（iteration 累计） | `on_policy_runner.py:527-544` |
| 优雅停止 | `training/time_budget.py:68-87` |

## 附录 C：风险与故障

| # | 症状 | 处置 |
|---|---|---|
| C-F1 | 门1（Δ100）impulse 全零 | 依序查：①读的是**对手** `contact_sensor_{对方}_torso` ②history index 0=最新、取 `[:, :4]` ③norm/max 维度 ④smoke `config.json` 确认 patch 生效 |
| C-F2 | 奖励切换 extinguish 进攻（nonzero% 归零、shaping 崩） | 按**有效奖励**升降档（§3.1 表）：熄火→w=400（0.125 N·s 达旧值 1.0）；失稳（loss 尖峰/倒地）→w=100（5 N·s 得 10）；两轮皆崩 → REJECT 刀③，H2b 维持候选。每次重试记录 config.json 与 impulse mean |
| C-F3 | validate mean_ms 越线 | 不采纳不提交；空闲 GPU 复测一次；持续越线弃该 checkpoint |
| C-F4 | 刀② qpos 索引错 | §5.1 核实步打印 `root_qi` 与开局距离后再训 |
| C-F5 | 额度中断 | 每块 `--max-runtime-seconds`；查 `training_result.budget.stop_reason`；续块**新目录**（G2） |
| C-F6 | 刀④ fixed 模式 vs resume 校验 | 一律 `--allow-resume-code-change` + winrate.md 记录原因 |
| C-F7 | SSH 掉线/tmux 丢失 | §4.3 三连查；**不假设进程状态** |
| C-F8 | 评测重复局同 summary 异 frames | winrate 已记录此非确定性——重复局是独立样本，4 局口径保留 |

## 附录 D：与上游/前版的差异

| 来源 | 本版结论 |
|---|---|
| 调研版 R1 冲量可得性 | 已解除：数据管道现成，刀③纯 reward 层 |
| 调研版刀②随 update 推远 | 常量 1.2m |
| 调研版 R3 对手池注入 | 需改代码两处；门控 >5h（与 B5_INNOVATION §2.5 已同步） |
| v2.0 w=20 | **作废**：漏算 0.02 缩放（G3），v2.1 = 200（有效值见 §3.1 表） |
| v2.0 门控绝对 iteration 100/300 | **作废**：F9 累计口径，改 Δ100/Δ300 |
| v2.0 同目录续写 | **作废**：G2 禁止，一律 -rN 新目录 |
| v2.0 分数分解推命中 | **作废**：G4 HIT 事件直数 + KO actor=受益方警示 |
| B5_INNOVATION 旧结论（双 KO 胜/31.5h/>6h 门槛） | 已同步翻案口径/16h/>5h（2026-10-25 深夜修订） |

## 附录 E：执行日命令速查（按序）

```bash
# 0 连接后盘点：§2.2 五项（sha×2 + git×2 + checkpoint 内部校验）
# 1 基线补测：§2.3 双重循环 → fall_attrib.py ×4 → 回填基线表
# 2 patch：§3.2 应用器（先校验后写盘）→ 烟雾 §3.3 → 三条硬验收
# 3 主线：§4.1 tmux 首块 → 续块一律 -rN 新目录；§4.2 监控 Δ100/Δ300 两道门
# 4 验收：§4.4 导出+validate（双门槛）；§4.5 native 4 局 + fall_attrib + 两级判定
# 5 记账：§8.2 模板；归档 §8.1；采纳 §8.3
```
