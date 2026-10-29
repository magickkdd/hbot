# HBOT 迁移清单（2026-10-29）

目标机：与当前**同配置**的新书生·端砚开发机（A100 80G / py3.12 / cuda12.6 / torch 2.9.1+cu128）。

## 需要搬运的文件（均在 `/data`）

| 路径 | 大小 | 用途 |
|---|---|---|
| `venv-py312.tgz` | 4.4G | Python 环境；解到 `/root` → `/root/hbvenv` |
| `hbot-core.tgz` | 3.8G | 冻结底层控制器 `models/holomotion-v2` + b7-boxing prepared + b1-prior |
| `hbot-code-full.tgz` | 1.5G | **当前已打补丁的代码**（`HoloMotion/`，不含 results / processed） |
| `migrate/hbot-b12-data.tgz` | 189M | LAFAN1 对打数据链（b12-boxing prepared / lafan-pair / lafan-csv / b12-prior-data / b12-measured-prior） |
| `migrate/hbot-b12-results.tgz` | 1.7G | 先验/BC/04g 结果与 ckpt + 评测现场（含 mp4） |
| `migrate/hbot-scripts.tgz` | 33K | `/data/*.sh *.py`（recover / watchdog / eval / 补丁 / 诊断） |
| `migrate/hbot-access.tgz` | 39M | tailscale 二进制+state、sshd 配置、authorized_key |
| `migrate/SHA256SUMS` | 535B | 校验和 |

合计约 **11.6G**。

## 新机恢复顺序

```bash
# 0. 上传后保持路径：前 3 个在 /data 根，其余在 /data/migrate/
cd /data && sha256sum -c migrate/SHA256SUMS

# 1. 解包
tar xzf /data/venv-py312.tgz -C /root
tar xzf /data/hbot-core.tgz -C /root/hbot
tar xzf /data/hbot-code-full.tgz -C /root/hbot
tar xzf /data/migrate/hbot-b12-data.tgz -C /root/hbot/HoloMotion holoboxing/data/processed/b12-boxing
tar xzf /data/migrate/hbot-b12-data.tgz -C /data lafan-pair lafan-csv b12-prior-data b12-measured-prior
tar xzf /data/migrate/hbot-b12-results.tgz -C /root/hbot/HoloMotion results
tar xzf /data/migrate/hbot-b12-results.tgz -C /data eval-b12-04g
tar xzf /data/migrate/hbot-scripts.tgz -C /data
tar xzf /data/migrate/hbot-access.tgz -C /data

# 2. 接入 + 系统依赖（sshd/tmux/rsync/mesa + tailscale）
bash /data/recover.sh

# 3. 自检
ls /root/hbot/HoloMotion/holoboxing/data/processed/b12-boxing/prepared/manifest.json
/root/hbvenv/bin/python -c "import torch;print(torch.__version__)"
ls /root/hbot/HoloMotion/results/{b12-sl2/epoch_000200.pt,b12-warm3/model_32.pt,b12-04g/model_200.pt}

# 4. 续跑 / 评测（训练退出后再评测，严禁并发）
bash /data/b12_04d.sh results/b12-04g/model_200.pt 200 results/b12-04h
bash /data/eval_quick.sh results/b12-04g/model_200.pt /data/eval-new
```

## 迁移时点状态

- 先验（阶段01）已在**实测目标分布**上重训：重建 **3.7cm**（原为参考分布 2.4cm / 实测 13–51cm）。
- BC（阶段02）：`epoch_000200` 位置误差 **0.0996m**。
- 阶段04g：BR 0 计分命中但把对手打倒 1 次（KO 胜）；AVG 0 命中被 KO；400 轮 RL 后模仿漂移 **0.626m**。
- 评测口径：20ms 窗口**法向**冲量 ≥ **5 N·s**、每手 0.2s 冷却；训练侧 `scored_hit` ±1 已对齐该口径。
- `/data` 配额 30G；watchdog v5 只归档当前训练目录的**最新 1 份** ckpt。

## 不在迁移范围

- 旧 ckpt（b8 / b12-04b–04f / sl-long 等，均已无续训价值）
- `official/` 官方克隆（1.7G）与 `incoming/` SMPL 源数据（授权限制，禁止再分发）
