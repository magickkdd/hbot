# 复现「7 分钟拳击」四阶段训练链（B9）

## Context

当前问题：B8 的 selfplay RL 跑到 7,657/20,000，评测 hit_count 只有 0/0/5/0，机器人基本不出拳。根因不是算力，而是**没有拳击参考动作**——策略只能靠奖励从零摸索出拳。

小红书视频（王煜城-地平线）拆解了真实做法，与我们架构完全同构（32 维 latent + 冻结解码器 + 冻结 HoloMotion V2）：

| 阶段 | 耗时 | 内容 |
|---|---|---|
| 01 潜空间建模 | 47s | 重定向后的拳击动作（225 维目标序列）→ 编码器离线拟合 → 32 维潜变量 → 解码器，重建误差驱动更新 |
| 02 模仿初始化 | 7s | 双机器人示范（自身+对手状态）→ 拳击策略预测后续动作目标，误差反传（BC），解码器冻结 |
| 03 站立热身 | 85s | 32 次站立更新，迁移权重与观测统计 |
| 04 交互优化 | 275s | 交互奖励（击中 +1 / 被击中 −1），解码器与底层全程冻结 |

注：SuperMimic 论文本身训练的是全身跟踪器（我们被冻结的那层），视频是**复用已训好的控制器**做高层任务策略，所以可直接借鉴。

**目标**：补齐缺失的“拳击示范数据”，按 01→04 顺序重训高层，把 hit_count 从 0~5 提到能稳定出拳。

## 前置（第 0 步）

停止当前训练链：kill `run_all` 与 `train_rl`，保留 `watchdog`（不动 checkpoint 归档）。

## 方案

### 1. 获取拳击动捕数据（外部数据集）

- 源：**CMU Graphics Lab Motion Capture Database, subject 13 = boxing**（免费、无需登录）。
  - 站点：`http://mocap.cs.cmu.edu/search.php?subjectnumber=13`
  - 文件形如 `/subjects/13/13.asf`（骨架定义）+ `/subjects/13/13_XX.amc`（逐帧关节旋转，度）
  - 无 BVH，需自己解析 ASF/AMC（文本格式，FK 简单）
- 本地下载 → 打包 → 经云盘/scp 推到开发机 `/data/boxing-mocap/`（开发机出网白名单不含 cmu.edu）

### 2. 动捕 → G1 关节轨迹（新模块）

新建 `holoboxing/data/cmu_mocap.py`：

1. 解析 `13.asf`：骨骼层级、每段方向/长度、各关节 DOF 轴
2. 解析 `13_XX.amc`：逐帧欧拉角 → 局部旋转矩阵 → FK 得世界坐标关节位置与朝向
3. 重定向到 G1（29 DoF）：
   - 首选：把 FK 关节位置喂给仓库已有 `thirdparties/joints2smpl` 拟合 SMPL → 复用 `holosmpl.converters.smpl_to_body_poses.SmplToBodyPosesConverter`（与 InterHuman 完全同一条路）
   - 备选（若拟合太慢/质量差）：直接按关节名映射局部旋转 + 骨骼长度缩放，写出 G1 qpos
4. 输出 source npz（`records.load_trajectory` 已支持的 legacy 单角色 schema）：
   `joint_pos (T,29)`、`body_pos_w (T,1,3)`、`body_quat_w (T,1,4)`、`joint_vel`、`body_ang_vel_w`、`fps=50`
   - 单角色（roles=1）时 `motion_kinematics` 会自动合成一个站立对手，正好对上阶段 02 的“双机器人示范”

### 3. 生成 prepared 拳击数据集

复用现有 CLI（不改代码）：

```
python -m holoboxing.data.records --assets ... --lower models/holomotion-v2 \
  --motion /data/boxing-mocap/g1 --output holoboxing/data/processed/b9-boxing/prepared
```

`MeasuredCollector` 会用**冻结的 HoloMotion V2** 跑轨迹采集，产出 `obs(1097)` + `target_pose(225)`。

必须在 `sources.json` 里给这批 clip 打一个新 source 标签（例如 `cmu-boxing`），并让 `verify_dataset` 通过。

### 4. 阶段 01：潜空间重建误差体检（可选，先测不重训）

用现有 prior（225→32→225）测 boxing target 的重建误差。
- 误差与 InterHuman 同量级 → 直接复用，省掉 47s 阶段
- 明显偏大 → 在 boxing target 上微调先验解码器（`train_prior.py`），再继续

### 5. 阶段 02：模仿初始化（BC）

改 `holoboxing/training/supervised/data.py` 一行：`select_sources(..., dataset="interhuman")` 增加参数化 `--dataset`，让 `train_sl.py` 能指向 `cmu-boxing`。

```
python -m holoboxing.training.train_sl --data holoboxing/data/processed/b9-boxing/prepared \
  --prior-checkpoint <现有 prior> --output results/b9-sl --epochs 8 --batch-size 256
```

### 6. 阶段 03：站立热身

沿用现有 warmup，改为 32 次更新（对齐视频口径），从 `results/b9-sl` 最新 checkpoint 起。

### 7. 阶段 04：交互优化

沿用现有 selfplay + NFSP 链（奖励已含 `strike_reward` / `defensive_hit_penalty` / 击中判定），但**从 BC 初始化后的权重起跑**，先跑短程（2,000~4,000 updates）验证是否保持出拳，再决定是否延长。

### 8. 验收

- `holoboxing validate` 双导出（BR / AVG）
- `evaluate_native` 四局（BR/AVG × side0/1），重点看 `hit_count`
- 渲染 mp4 回 D 盘肉眼确认
- 与 model_7657 基线对比（当前 0 / 5 / 0 / 0 击中）

## 需要改动的文件

| 文件 | 动作 |
|---|---|
| `holoboxing/data/cmu_mocap.py` | 新建：ASF/AMC 解析 + FK + G1 重定向 + source npz 写出 |
| `holoboxing/training/supervised/data.py` | 改：`dataset` 参数化（默认仍 interhuman） |
| `holoboxing/training/train_sl.py` | 改：暴露 `--dataset` |
| `/data/shusheng_run_all.sh` | 改：链路改为 SL(b9) → warmup(32) → RL(短程) |

复用的现成件：`records.motion_kinematics` / `MeasuredCollector` / `train_sl` / `standing warmup` / `nfsp` / `eval_ckpt.sh` / `render_replay.py`。

## 验证方式

1. 数据：跑完 prepared 后检查 npz 的 `obs (N,1097)`、`target_pose (N,225)`，并 `verify_dataset`
2. 视觉：用脚本策略（直接输出重定向后的 225 维序列）跑一局，渲染 mp4 确认**动作确实是拳击**（这一步不过就不用往下走）
3. BC：`results/b9-sl` 的验证误差 + 一次 native 对局
4. 全链：四局 native 的 `hit_count` 与基线对比 + 视频

## 风险与预案

- **CMU 骨架与 G1 差异大**（身高/关节轴向）→ 若 joints2smpl 拟合质量差，退化为直接角度映射，并用渲染验证物理可行性
- **BC 过拟合到单一片段** → 保留 InterHuman 数据混合训练（boxing 采样加权），避免丢失基础平衡能力
- **冻结控制器跟不上拳击目标** → 降低动作幅度/速度（50Hz 下每拳 12~20 帧），以控制器实际跟踪误差为准迭代
- 数据许可：CMU mocap 免费用于研究；不上传、不对外分发原始数据
