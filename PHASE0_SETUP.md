# HBOT 阶段 0 规格：远程连接、环境搭建与资源准备

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0 / 2026-10-24 |
| 适用实例 | ModelScope 免费 GPU 实例（24GB 显存，镜像 `ubuntu22.04-cuda13.0.3-py312-torch2.13.0`） |
| 目标 | 实例开机后：① agent 按本文档连上远程 notebook；② 训练/评测环境搭建完成；③ 官方资源与烟雾数据准备完成 |
| 完成判据 | 见「阶段 4：总验收」，四类检查全绿 |
| 上游文档 | [hbot_boxing_prelim_plan.md](hbot_boxing_prelim_plan.md)（§4.4 数据流程、§7.1 评测环境、§10 Day 0） |
| 参考来源 | RLPD 项目 `deployment/ModelScope_Notebook_用户操作指引.md`（Tailscale+OpenSSH 完整 SOP，本文故障排查引用它） |

**角色约定**：标注 **【用户手动】** 的步骤需要人在浏览器/官网操作（agent 无法代替）；其余步骤 agent 均可在本地 WSL 或 SSH 会话中代跑。**每次开机后，用户只需说一句「按 PHASE0 连接」，agent 即执行 §1.4–§1.6。**

---

## 阶段总览

| 阶段 | 内容 | 执行位置 | 执行者 | 前置 |
|---|---|---|---|---|
| 0 | 本地预检（tailscale/密钥/脚本/AuthKey） | 本地 WSL | agent | 无 |
| 1 | 实例启动 + SSH 打通（仅首次 10–20 分钟，日常约 2 分钟） | 浏览器 + WebIDE + 本地 | 用户手动（1.1–1.3）/ agent（1.4–1.6） | 阶段 0 |
| 2 | 远程环境搭建（系统依赖、仓库、训练 venv、资源、评测环境） | `ssh hbot` | agent | 阶段 1 |
| 3 | 数据资源准备（RoboStriker、InterHuman、SMPL→retarget→prepare） | `ssh hbot` | agent（3.3 用户手动） | 阶段 2.1–2.4 |
| 4 | 总验收与交接 | 本地 + 远程 | agent | 全部 |

远程统一约定：仓库根 `/mnt/workspace/HoloMotion`；长任务一律放 `tmux`（会话名 `hbot`）；环境变量三件套写入 `~/.bashrc` 持久化。

---

## 阶段 0：本地预检（agent，本地 WSL）

### 0.1 本地 tailscaled 存活与登录态

```bash
TS=~/.local/ts/tailscale; SOCK=$HOME/.local/share/ts/ts.sock
$TS --socket=$SOCK status | head -5
```

- **验收**：列表中 `czx-dev` 存在且**非** `offline`；无 `Logged out` / `NeedsLogin`。
- 若 daemon 未运行：执行 `deployment/connect_hbot.sh` 内置的拉起逻辑，或参照 RLPD SOP 步骤 4 启动后重新检查。

### 0.2 SSH 密钥与 bootstrap 公钥一致

```bash
cat ~/.ssh/id_ed25519.pub
grep ssh-ed25519 /home/czx/HBOT/deployment/bootstrap_hbot.sh
```

- **验收**：两行完全一致（当前为 `czx@DESKTOP-E93227A`）。不一致则以本地公钥为准修改脚本内 `EOF_PUB` 段。

### 0.3 项目脚手架就位

```bash
ls -l /home/czx/HBOT/deployment/bootstrap_hbot.sh /home/czx/HBOT/deployment/connect_hbot.sh
bash -n /home/czx/HBOT/deployment/bootstrap_hbot.sh && bash -n /home/czx/HBOT/deployment/connect_hbot.sh && echo SYNTAX_OK
chmod +x /home/czx/HBOT/deployment/*.sh
```

- **验收**：两个脚本存在、`SYNTAX_OK`、可执行位已加。

### 0.4 Tailscale Auth Key 有效性（仅实例首次入网需要）

```bash
grep -c '^TS_AUTHKEY=' /home/czx/RLPD/deployment/.secrets.env 2>/dev/null
```

- **验收**：输出 `1` 表示有可尝试复用的 key（RLPD 项目 2026-10-17 创建，是否过期待实例首连时验证）。
- key 失效时的处理：**【用户手动】** 到 `https://login.tailscale.com/admin/settings/keys` 新建 **reusable + preauthorized** key，在运行 bootstrap 时通过 `TS_AUTHKEY=` 环境变量注入（见 1.3）。key 只通过环境变量传递，不写入任何脚本/文档/Git。

---

## 阶段 1：实例启动与 SSH 连接

### 1.1 【用户手动】启动实例并打开 WebIDE 终端

1. 登录 `https://www.modelscope.cn` →「我的 Notebook」→「创建/启动实例」。
2. 规格：**GPU 类型**（免费额度，24GB 显存）；镜像选 `ubuntu22.04-cuda13.0.3-py312-torch2.13.0`（驱动栈最新，保证 venv 里自装 `torch==2.9.1+cu128` 可用；预装 py312/torch2.13 不使用）。
3. 状态变为「运行中」后打开 Notebook，进入 WebIDE，打开底部「终端」。
4. 在 WebIDE 终端里自查：

```bash
pwd          # 验收：/mnt/workspace
whoami       # 验收：root
nvidia-smi   # 验收：看到 24GB GPU（如 NVIDIA A10/或其他）；报错说明开成 CPU 实例 → 停止重开
df -h /mnt/workspace   # 记录可用空间，应 > 30GB
```

### 1.2 【用户手动，仅首次】放置 bootstrap 脚本

方式 A（推荐，文件上传）：WebIDE 文件管理器把本地 `/home/czx/HBOT/deployment/bootstrap_hbot.sh` 上传为 `/mnt/workspace/.remote-access/bootstrap_hbot.sh`。

方式 B（终端粘贴）：

```bash
mkdir -p /mnt/workspace/.remote-access && chmod 700 /mnt/workspace/.remote-access
# 用 WebIDE 编辑器在 /mnt/workspace/.remote-access/ 新建 bootstrap_hbot.sh，
# 粘贴本地脚本全文保存
chmod +x /mnt/workspace/.remote-access/bootstrap_hbot.sh
bash -n /mnt/workspace/.remote-access/bootstrap_hbot.sh && echo SYNTAX_OK
grep ssh-ed25519 /mnt/workspace/.remote-access/bootstrap_hbot.sh
```

- **验收**：`SYNTAX_OK`；grep 到的公钥与本地 `~/.ssh/id_ed25519.pub` 一致。

### 1.3 【用户手动】运行 bootstrap

```bash
# TS_AUTHKEY 仅首次需要；可先不带 key 直接跑（凭持久 state 复用），失败再带 key 重跑
TS_AUTHKEY=<可选：RLPD .secrets.env 中的 key 或新建 key> \
nohup bash /mnt/workspace/.remote-access/bootstrap_hbot.sh \
  > /mnt/workspace/.remote-access/bootstrap_run.log 2>&1 &
sleep 12
cat /mnt/workspace/.remote-access/receipt.txt
```

- **验收（四项全过）**：
  - `receipt.txt` 含 `RECEIPT: IP=100.x.x.x UP_RC=0 HOST=hbot-dsw`；
  - `ss -ltn | grep 2223` → `127.0.0.1:2223 LISTEN`；
  - `tailscale --socket=/mnt/workspace/.remote-access/ts.sock serve status` → `2222 -> tcp://127.0.0.1:2223`；
  - 同上 socket 的 `status` 中本机非 `offline`。
- 失败对照 RLPD SOP《常见问题排查》§3.1–3.6（`/home/czx/RLPD/deployment/ModelScope_Notebook_用户操作指引.md`）。高频三条：`UP_RC≠0` 且 `up_err.log` 提示 invalid key → 换新 key；`SSHD_UP` 未打印 → `apt-get install -y openssh-server` 后重跑；实例此前跑过其他项目残留旧 daemon → 脚本已 `pkill`，重跑一次即可。

### 1.4 【agent】本地确认节点并记录 IP

```bash
TS=~/.local/ts/tailscale; SOCK=$HOME/.local/share/ts/ts.sock
$TS --socket=$SOCK status | grep hbot-dsw      # 验收：出现且非 offline
$TS --socket=$SOCK ping hbot-dsw -c 3 2>/dev/null || $TS --socket=$SOCK ping hbot-dsw | head -3
```

- **验收**：`hbot-dsw` 在列表中非 `offline`，ping 有 `pong`（DERP 高延迟也算通）。
- 把 IP 写入 `/home/czx/HBOT/deployment/.ts_ip`（`connect_hbot.sh` 首次带 IP 调用会自动写）。

### 1.5 【agent】固化 SSH 配置并首连

向 `~/.ssh/config` 追加（IP 用 1.4 拿到的值）：

```sshconfig
Host hbot
    HostName 100.x.x.x
    User root
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
    ProxyCommand /home/czx/.local/ts/tailscale --socket=/home/czx/.local/share/ts/ts.sock nc %h %p
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 4
```

首连验收（逐条执行并记录输出）：

```bash
ssh hbot 'hostname && pwd && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv && df -h /mnt/workspace && free -h | head -2'
ssh hbot 'touch /mnt/workspace/.write_test && ls -l /mnt/workspace/.write_test && rm /mnt/workspace/.write_test'
```

- **验收**：`hostname` 为实例名；`nvidia-smi` 显示 24GB GPU；持久盘写入测试通过。

### 1.6 【agent】实例基线快照

把 1.5 的输出追加保存到本地 `/home/czx/HBOT/deployment/instance_snapshot.md`（含日期、IP、GPU 型号、磁盘、内核 `uname -a`），并请用户口述当前免费额度剩余，一并记录——后续所有额度核算以此为基线。

### 1.7 日常开机复用（每次实例重启后）

1. **【用户手动】** 浏览器启动实例 → WebIDE 终端执行（同一行，幂等）：

```bash
nohup bash /mnt/workspace/.remote-access/bootstrap_hbot.sh \
  > /mnt/workspace/.remote-access/bootstrap_run.log 2>&1 & sleep 12; \
cat /mnt/workspace/.remote-access/receipt.txt
```

2. **【agent】** `bash /home/czx/HBOT/deployment/connect_hbot.sh`（或直接 `ssh hbot`）。
3. IP 变化（持久盘身份丢失/重建）时：以新 `receipt.txt` 为准，更新 `.ts_ip` 与 `~/.ssh/config` 的 `HostName`；若报 host key 变化，先 `ssh-keygen -R <旧IP>`。

---

## 阶段 2：远程环境搭建（agent，经 `ssh hbot`）

> 以下命令均在 SSH 会话内、`/mnt/workspace` 下执行。第 2.3/2.6 步耗时较长（各 10–20 分钟，下载 torch 与依赖），放 `tmux new -s hbot` 里跑，防止 SSH 断开中断。

### 2.1 系统依赖与 Python 3.11

```bash
apt-get update -qq && apt-get install -y -qq git git-lfs tmux
git lfs version                      # 验收：有版本输出
python3.11 -V 2>/dev/null || true    # 分支判定
```

- 有 `python3.11`（≥3.11）→ 直接用；
- 没有 → `apt-get install -y python3.11 python3.11-venv`；仍没有（apt 源缺包）→ 用 DSW 自带 conda：`conda create -y -n py311 python=3.11`，后续以 `~/.conda/envs/py311/bin/python3.11` 代替 `python3.11`。
- **验收**：`python3.11 -V` 输出 `Python 3.11.x`。

### 2.2 克隆 HoloMotion 与 LFS 资源

```bash
cd /mnt/workspace
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/HorizonRobotics/HoloMotion.git
cd HoloMotion
git submodule update --init thirdparties/HoloMotion_assets
git -C thirdparties/HoloMotion_assets lfs install --local
git -C thirdparties/HoloMotion_assets lfs pull --include="holoboxing/**" --exclude=""
git -C thirdparties/HoloMotion_assets lfs checkout holoboxing
cat >> ~/.bashrc <<'EOF'
export HOLOBOXING_ASSETS=/mnt/workspace/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/mnt/workspace/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/mnt/workspace/HoloMotion/holoboxing/data/processed/combined/prepared
EOF
# 注意：~/.bashrc 在 ssh 非交互命令中可能不加载；再落一份可 source 的 env.sh（后续阶段统一用它）
cat > /mnt/workspace/HoloMotion/env.sh <<'EOF'
export HOLOBOXING_ASSETS=/mnt/workspace/HoloMotion/holoboxing/assets
export HOLOBOXING_LOWER=/mnt/workspace/HoloMotion/models/holomotion-v2
export HOLOBOXING_DATA=/mnt/workspace/HoloMotion/holoboxing/data/processed/combined/prepared
EOF
source ~/.bashrc
```

- **验收**：`thirdparties/HoloMotion_assets/holoboxing/resources.json` 存在；`ls thirdparties/HoloMotion_assets/holoboxing/` 能看到 `arena/`、`29dof/`、`mode15_boxing_meshes/`。

### 2.3 训练环境（venv + torch 2.9.1 cu128）

```bash
cd /mnt/workspace/HoloMotion
python3.11 -m venv .venv-holoboxing
source .venv-holoboxing/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r holoboxing/training/requirements.txt
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

- **验收**：`pip check` 无冲突；最后一条打印 GPU 名称。失败处理：CUDA 断言失败 → `nvidia-smi` 看驱动，驱动过旧则说明镜像选错，回到 1.1；依赖冲突 → 照 `pip check` 输出逐个固定版本。

### 2.4 官方资源下载与校验（assets + 固定控制器 1.65GB）

```bash
# 仍在 .venv-holoboxing 内
python -m holoboxing assets --assets "$HOLOBOXING_ASSETS"
# 验收：输出 49/49 文件大小与 SHA256 全部通过（本地 WSL 已验证过同一份，远程应一致）
python -m holoboxing download --directory models/holomotion-v2
# 下载中断会留 .partial，重跑同一命令续传；完成后自动校验
ls -l models/holomotion-v2/exported/model_10500.onnx
# 验收：1,647,887,681 字节（本地基线值）
sha256sum models/holomotion-v2/exported/model_10500.onnx
# 验收：b0681976135eed572d1d20aefc52ec65c149bdc636f2594a6436fa7136a3fc86
```

### 2.5 Docker 探测（预期不可用，仅记录）

```bash
ls -l /var/docker/proxy/docker-proxy.sock 2>/dev/null || echo NO_DOCKER_PROXY
ls -l /var/run/docker.sock 2>/dev/null || echo NO_DOCKER_SOCK
```

- **验收**：两个都不存在 → 在 `instance_snapshot.md` 记录「无 Docker；本地校验/对战走 2.6 直接 Python 评测」。若 socket 存在则补 `export DOCKER_HOST=...` 后按主方案 §7.1 Docker 路径走（概率低）。

### 2.6 评测环境（直接 Python 评测，无 Docker 路径）

```bash
# 独立于训练 venv，用 conda 隔离，避免依赖互相污染
conda create -y -n hbot-eval python=3.11
conda run -n hbot-eval pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n hbot-eval pip install -r holoboxing/evaluation/docker/requirements.in
conda run -n hbot-eval pip install -e .
# 生成机器人模型与编译观测（第一次需要；requirements.in 无 torch，但源码回退需要，故先装）
conda run -n hbot-eval python -m holoboxing.evaluation.robot_models --assets "$HOLOBOXING_ASSETS"
conda run -n hbot-eval python -m holoboxing.evaluation.runtime.observation_export \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --output holoboxing/evaluation/runtime/compiled
conda run -n hbot-eval python -m holoboxing validate --help
```

- **验收**：`robot_models` 输出 JSON 校验通过；`observation_export` 生成 `runtime/compiled` 产物；`validate --help` 正常打印。失败处理：依赖装不上（torch/onnxruntime 版本冲突）→ 退化为「只靠平台验证 + 租 1–2 小时 Docker GPU 跑官方镜像」（主方案 §7.1 末段），并在阶段 4 登记。

### 2.7 磁盘盘点

```bash
df -h /mnt/workspace
du -sh /mnt/workspace/HoloMotion/{.venv-holoboxing,models,thirdparties} 2>/dev/null
```

- **验收**：剩余空间 > 25GB（torch 双环境约 10–12GB + 控制器 1.65GB + 预留数据 ~10GB）。不足则清理 `~/.cache/pip` 并考虑删一个评测环境。

---

## 阶段 3：数据资源准备（agent，经 `ssh hbot`，训练 venv 内）

### 3.1 RoboStriker 烟雾数据（4 组）

```bash
cd /mnt/workspace/HoloMotion && source .venv-holoboxing/bin/activate
python -m holoboxing.data download robostriker \
  --output holoboxing/data/raw/RoboStriker --limit-groups 4 --workers 2
```

- **验收**：`holoboxing/data/raw/RoboStriker` 下出现 4 个动作组目录，无 `FAILED.json`；下载命令本身会校验摘要。

### 3.2 InterHuman 片段（3 段，许可允许范围内的官方示例）

```bash
python -m holoboxing.data download interhuman \
  --output holoboxing/data/raw/InterHuman \
  --clip-id 1124 --clip-id 1843 --clip-id 4033 --workers 2
```

- **验收**：`holoboxing/data/raw/InterHuman` 下 3 个 clip 数据就位。
- 注意：InterHuman 原始许可为非商业研究用途、禁止再分发；仅在实例内使用，不拷出共享。

### 3.3 【用户手动，外部依赖】SMPL 申请（不阻塞 3.1/3.2 与阶段 2）

1. 在 `https://smpl.is.tue.mpg.de/` 注册账号并发起 neutral SMPL 模型申请（学术非商业）。
2. 批准邮件到达后下载 `neutral_smpl.pkl`（实际文件名以官网为准），通过 WebIDE 上传到实例 `/mnt/workspace/licenses/neutral_smpl.pkl`（目录自建，chmod 700）。
3. 告知 agent「SMPL 已就位」。
- 状态登记：申请提交与否、就位与否写入 `instance_snapshot.md`。

### 3.4 【SMPL 就位后】重定向 + 合并预处理

```bash
cd /mnt/workspace/HoloMotion
python3.11 -m venv .venv-holoboxing-retarget
.venv-holoboxing-retarget/bin/python -m pip install torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-holoboxing-retarget/bin/python -m pip install -r holoboxing/data/retarget-requirements.txt
.venv-holoboxing-retarget/bin/python -m pip check

.venv-holoboxing-retarget/bin/python -m holoboxing.data retarget \
  --source holoboxing/data/raw/InterHuman \
  --output holoboxing/data/converted/InterHuman \
  --smpl-model /mnt/workspace/licenses/neutral_smpl.pkl \
  --assets "$HOLOBOXING_ASSETS" --device cuda:0 --iterations 4

# 回到训练 venv 做合并与校验
source .venv-holoboxing/bin/activate
python -m holoboxing.data prepare \
  --source holoboxing/data/raw/RoboStriker \
  --source holoboxing/data/converted/InterHuman \
  --output holoboxing/data/processed/combined \
  --assets "$HOLOBOXING_ASSETS" --test-fraction 0.1 --workers 2
python -m holoboxing.data verify \
  --output holoboxing/data/processed/combined --assets "$HOLOBOXING_ASSETS"
```

- **验收**：`holoboxing/data/processed/combined/` 下出现 `COMPLETE.json` 且**没有** `FAILED.json`；`prepared/` 目录非空；`HOLOBOXING_DATA` 路径有效。
- SMPL 未就位时本步挂起（登记为阻塞项），其余阶段照常。

### 3.5 清单归档回本地

```bash
# 本地 WSL 执行
scp hbot:/mnt/workspace/HoloMotion/holoboxing/data/processed/combined/COMPLETE.json \
    /home/czx/HBOT/records/COMPLETE_$(date +%m%d).json 2>/dev/null || true
```

（`records/` 目录自建。）所有关键校验输出（assets 49/49、控制器 SHA256、verify 结果）追加进 `instance_snapshot.md`。

---

## 阶段 4：总验收与交接

| 类别 | 检查项 | 通过标准 |
|---|---|---|
| 连接 | `ssh hbot` | 直接进入 root shell；`~/.ssh/config` 与 `.ts_ip` 一致 |
| 连接 | 日常复用 | 重启实例后按 §1.7 两步恢复，全程 ≤ 2 分钟 |
| 环境 | 训练 venv | `pip check` 干净；`torch.cuda.is_available()` 为真 |
| 环境 | 评测环境 | `python -m holoboxing validate --help` 可用；或已登记降级方案 |
| 资源 | assets / 控制器 | 49/49 校验通过；`model_10500.onnx` SHA256 与本地基线一致 |
| 数据 | RoboStriker / InterHuman | 4 组 + 3 片段就位，无 FAILED |
| 数据 | prepared（可选） | `COMPLETE.json` 存在（依赖 SMPL，允许阻塞） |
| 记录 | `instance_snapshot.md` | IP、GPU、磁盘、额度基线、Docker 结论、SMPL 状态齐全 |

**交付物**：

- 远程：`/mnt/workspace/HoloMotion`（仓库+资源+控制器+数据+两个环境）；
- 本地：`deployment/{bootstrap_hbot.sh, connect_hbot.sh, .ts_ip, instance_snapshot.md}`、`records/`。

**阻塞与降级**：

- SMPL/InterHuman 许可延迟 → 3.4 挂起；训练侧可走主方案 §4.2 的 RoboStriker-only prior + `--prior-checkpoint` 冒烟路径（属下一阶段文档，不在本阶段展开）。
- 评测环境装不起来 → 降级为「平台验证 + 租用 Docker GPU」，登记后不阻塞训练。

**下一步**：本表全绿（prepared 一项除外）后，进入「阶段 1：烟雾训练与导出验证」——见 [PHASE1_SMOKE_TRAINING.md](PHASE1_SMOKE_TRAINING.md)（prior + SL + RL 冒烟、ONNX 导出与 `--validate`、对手池语义实测）。

---

## 附录 A：高频故障速查（详表见 RLPD SOP §3）

| 现象 | 一句话处理 |
|---|---|
| `UP_RC≠0` 且 `up_err.log` 有 invalid key | AuthKey 过期 → 新建 reusable+preauthorized key 重跑 1.3 |
| `Connection timed out`（本地） | 本地 tailscaled 没起或实例离线 → 0.1 / RLPD SOP §3.1 |
| `Permission denied (publickey)` | bootstrap 内公钥与本地私钥不配对 → 0.2 修正后重跑 1.3 |
| `Host key verification failed` | 实例重建后 host key 变化 → `ssh-keygen -R <IP>` 重连 |
| WebIDE 白屏/终端冻结 | 刷新页面；不影响实例内进程，SSH 不受牵连 |
| 实例空闲 1h/单次 4h 自动停 | 长任务必须 `tmux`/`nohup`；停了按 §1.7 重来 |
| 写 `/root`、`/tmp` 的文件丢了 | 非持久目录；一律写 `/mnt/workspace` |
| `nvidia-smi` 不存在 | 开成 CPU 实例 → 停止，按 1.1 重开 GPU 规格 |

## 附录 B：本项目 deployment/ 文件清单

| 文件 | 用途 |
|---|---|
| `bootstrap_hbot.sh` | 实例侧引导（Tailscale userspace + sshd 2223 + serve 2222），上传到 `/mnt/workspace/.remote-access/` |
| `connect_hbot.sh` | 本地一键连接；首次带 IP 参数，之后读取 `.ts_ip` |
| `.ts_ip` | 实例 tailnet IP 记录（连接成功后自动写入） |
| `instance_snapshot.md` | 实例基线与各项校验记录（IP/GPU/磁盘/额度/Docker/SMPL 状态） |

## 附录 C：安全红线

- `TS_AUTHKEY`、SSH 私钥、`tailscaled.state`、`ssh-host-keys/`、`neutral_smpl.pkl` 一律不入 Git、不入文档正文、不上传公开位置。
- `.remote-access` 目录 700、凭证文件 600；sshd 仅监听 `127.0.0.1:2223`，不映射公网。
- InterHuman/SMPL 数据仅限本队实例内非商业研究使用，禁止再分发。
