# HBOT 实例基线快照（PHASE0）

> 建档：2026-10-24 · 最后更新：2026-10-25 01:00（Asia/Shanghai）
> 执行：agent，按 PHASE0_SETUP.md v1.0
> 免费额度剩余：单次会话约 7 小时、总额度 31.5 小时（用户口述，2026-10-25 01:14 时点）——后续额度核算基线

## 1. 连接

| 项 | 值 |
|---|---|
| tailnet 节点 | `rlpd-mi300`（沿用旧名，实际为 CUDA 实例） |
| tailnet IP | `100.78.127.38`（`.ts_ip` 与 `~/.ssh/config` 一致） |
| 连接路径 | `ssh hbot` → sshd `127.0.0.1:2223` → tailscale serve `2222`；本地 userspace tailscaled + DERP(hkg) 中继 |
| 当前 hostname | `dsw-2294682-8b8695474-pp5dc`（2026-10-25 00:18 重启后；旧快照为 `dsw-2293909-...`） |
| 镜像 | Ubuntu 22.04.5，内核 `5.10.134-17.3.al8` |
| 重连时间 | 2026-10-25 00:18:40 CST（旧实例空闲关机，节点离线 46 分钟后恢复） |

## 2. 硬件基线

| 项 | 值 |
|---|---|
| GPU | NVIDIA A10, 23028 MiB（当前占用 0 MiB） |
| 内存 | 28 GiB |
| `/mnt/workspace` | NAS 1.0P，已用 29T，可用 996T |

## 3. 环境搭建结果（PHASE0 §2）

| 步骤 | 状态 | 关键输出 |
|---|---|---|
| 2.1 系统依赖 | ✅ | git-lfs 3.2.0（镜像自带）；tmux 3.2a、rsync 3.2.7 已装；deb 已缓存到 `.bootstrap/apt/` |
| 2.1 Python 3.11 | ✅ | conda `py311` = 3.11.17（`/mnt/workspace/miniforge3/envs/py311`，持久化） |
| 2.2 仓库 | ✅ | `/mnt/workspace/HoloMotion`，HEAD `adaacd5`（GitHub 直连） |
| 2.2 资产子模块 | ✅ 文件级 | 49 文件 / 32,197,369 字节；manifest sha256 `2d3d9459…`；HF 不可达，资产由本地官方克隆同步 |
| 2.3 训练 venv | ✅ | `.venv-holoboxing`（7.8G）+ torch 2.9.1+cu128；CUDA=A10；`pip check` 干净 |
| 2.4 控制器 ONNX | ✅ 已修复 | 1,647,887,681 B；sha256 `b0681976135eed572d1d20aefc52ec65c149bdc636f2594a6436fa7136a3fc86`；frozen bundle 校验 `FROZEN_BUNDLE_OK` |
| 2.4 资产校验 | ✅ | 训练 venv `python -m holoboxing assets` 通过 |
| 2.5 Docker | ✅ 已确认 | 无 `/var/docker/proxy/docker-proxy.sock`、无 `/var/run/docker.sock` → 直接 Python 评测 |
| 2.6 评测环境 | ✅ | conda `hbot-eval` + torch 2.9.1+cu128；`robot_models` JSON 通过；`validate --help` 通过 |
| 2.6 观测编译 | ✅ | `observation_export` 生成 `runtime/compiled/`（6 文件 381K），`export-parity.json` = `PASS`；`COMPILED_MANIFEST_OK schema=1` |
| 2.7 磁盘 | ✅ | repo venv 7.8G / models 1.6G / thirdparties 31M / InterHuman 2.2M / compiled 381K；NAS 余量 996T |

### 2.4 控制器修复记录（重要）

- 2026-10-24 22:06 快照曾把控制器标为完成，但实例关机前 v2 分块下载实际只完成 14/16 块（缺 `chunk_06`、`chunk_14`），且旧快照的 sha 记录不成立。
- 2026-10-25 00:21–00:23 用 hf-mirror 重新生成 **每块独立签名 URL**，仅补两块；远端已有 14 块逐块 MD5 与本地官方 ONNX 完全一致。
- 拼接后 sha256 与基线一致，`chunks2/` 已清理；回执：`/mnt/workspace/logs/onnx_fix_receipt.txt`（本地 `records/onnx_fix_receipt_20261025.txt`）。
- 模型仓库：`HorizonRobotics/HoloMotion_models` @ `e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a`，子目录 `HoloMotion_motion_tracking_model_v2.0.0`。

### 2.6 观测编译注意事项

- `observation_export` 会用「`holoboxing/evaluation/runtime/compiled` 是否存在」判断已有编译包；**输出目录必须事先不存在**，脚本会自行创建。预先 `mkdir` 空目录会报 `Export from source without an existing compiled bundle`。
- 日志：`/mnt/workspace/logs/observation_export.log`；本地副本 `records/observation_export_20261025.log`。

## 4. 持久化与每次开机恢复（新增）

### 4.1 持久盘 vs 系统盘

| 位置 | 生命周期 | 内容 |
|---|---|---|
| `/mnt/workspace/HoloMotion` | 持久 | 仓库、venv、models、数据、compiled、env.sh |
| `/mnt/workspace/miniforge3` | 持久 | conda、`py311`、`hbot-eval` |
| `/mnt/workspace/.remote-access` | 持久 | bootstrap/post_boot/verify 脚本、ssh host keys、authorized key、tailscaled.state、receipt |
| `/mnt/workspace/.bootstrap` | 持久 | tailscale 二进制缓存、apt deb 缓存、校验清单、环境快照、工具副本 |
| `/mnt/workspace/logs` | 持久 | 所有长任务日志与回执 |
| `/root`、`/tmp`、`/usr`、`/etc` | **易失** | 每次开机重置；不要把关键文件放这里 |

### 4.2 每次开机一条命令（在 WebIDE 终端执行）

```bash
nohup bash /mnt/workspace/.remote-access/post_boot.sh \
  > /mnt/workspace/.remote-access/post_boot.log 2>&1 &
```

等约 60–90 秒后查看：

```bash
cat /mnt/workspace/.remote-access/post_boot_receipt.txt
```

`post_boot.sh` 会依次：运行 `bootstrap_hbot.sh`（恢复 sshd/tailscale、注入 env、恢复 tmux/rsync）→ 运行 `verify_state.sh` → 输出 19 项核心检查结果。**必须在 WebIDE 运行**，因为 bootstrap 会重启 tailscaled，SSH 会话会短暂断开。

**已实测（2026-10-25 00:34）**：通过 SSH 启动 `post_boot.sh` 后 tailscale/sshd 重启，**5 秒内 SSH 自动恢复**；`bootstrap_rc=0 verify_rc=0`，`VERIFY_PASS=19 VERIFY_FAIL=0`；serve `2222 -> 127.0.0.1:2223` 与 sshd 监听均正常。回执：`/mnt/workspace/.remote-access/post_boot_receipt.txt`（本地 `records/post_boot_receipt_20261025.txt`）。

**00:15 重启实测补充（2026-10-25 01:00）**：本次重启 `/root/.bashrc` 被重置（三条 HOLOBOXING 导出丢失，已重新写入）；`/usr` 这次**未**被重置（apt 装的 tmux/rsync/git-lfs 幸存），但不可依赖，持久化策略不变。`post_boot.sh` 已增强：每次开机自动补回 `~/.bashrc` 环境变量（幂等 grep 判断），本地副本 `deployment/post_boot.sh` 已同步更新。

### 4.3 持久化缓存清单

- `.bootstrap/bin/tailscale`、`tailscaled`（33M/42M，避免下次联网安装）
- `.bootstrap/apt/`：openssh-server、openssh-sftp-server、tmux、rsync 的 deb
- `.bootstrap/checksums/`：ONNX sha256、frozen-model.json、16 块 MD5
- `.bootstrap/env/`：训练/评测 venv 的 `pip freeze`、两个 conda 环境导出
- `.bootstrap/tools/`：bootstrap/post_boot/verify_state/cache_persistent 四个脚本副本
- `env.sh` 已增加 `PIP_CACHE_DIR=/mnt/workspace/.cache/pip`、`CONDA_PKGS_DIRS=/mnt/workspace/.cache/conda-pkgs`

### 4.4 实例约束与作业规范

- 单次开机最长 **8 小时**；无操作约 **1 小时**自动关机。
- 长任务必须放 `tmux`（会话 `hbot`）或 `nohup`，日志写 `/mnt/workspace/logs/`。
- 大文件下载必须分块、每块独立签名 URL、逐块大小/MD5 校验，拼接后整体 sha256；不要依赖同 URL 的不同 Range（CloudFront 缓存会串块）。
- 关键产物一律写 `/mnt/workspace`；不要写 `/root`、`/tmp`。

## 5. 数据资源（PHASE0 §3）

| 数据集 | 状态 | 说明 |
|---|---|---|
| RoboStriker 4 组（3.1） | ✅ **完成** | 用户浏览器下载 GDrive 全文件夹 zip（145MB，86 npz，本地留档 `incoming/robostriker/`）；前 4 组 8 文件 sha256 逐个 MATCH 后推上实例；官方 `download robostriker --limit-groups 4` 零网络通过，receipt 落盘（`complete_source=false` 为预期，仅取 4/43 组） |
| InterHuman 3 片段（3.2） | ✅ **完成** | 10/10 文件（3 motions + 3 annots + 3 split + LICENSE.md）经官方命令 sha256 全部校验通过，receipt `raw/InterHuman/download.json` |
| 3.2 download.json 回执 | ✅ 已生成 | `holoboxing/data/raw/InterHuman/download.json`；无 FAILED、无 `.partial` 残留 |
| SMPL neutral（3.3） | ✅ **已就位** | `/mnt/workspace/licenses/neutral_smpl.pkl`（官方 `SMPL_python_v.1.1.0.zip` 内 `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`，247MB；结构校验 6890 顶点/24 关节/shapedirs 300 列）。用户另下载的 `SMPL_NEUTRAL.npz` 不适用（代码只吃 pkl），已弃用 |
| 3.4 retarget+prepare | ✅ **完成** | retarget venv `.venv-holoboxing-retarget`（torch cu128 + mujoco/warp/newton/chumpy，`pip check` 干净）；retarget 3 clips RC=0；prepare RC=0；verify RC=0。日志 `/mnt/workspace/logs/{retarget_env,retarget_prepare,prepare_verify}.log` |
| 3.5 COMPLETE.json 回传 | ✅ **完成** | `records/COMPLETE_1025.json`：status PASS、11460 supervised samples、10 robot_h5 clips、9 prepared files、无 FAILED.json、`HOLOBOXING_DATA` 有效 |

**3.4 执行要点（复跑必读）**：`prepare` 的实测采集与 `runtime/compiled/`（2.6 观测编译产物）**互斥**——`MeasuredCollector` 检测到 compiled 目录存在即报 `Measured preparation requires the source runtime`。复跑 prepare 用现成脚本 `/mnt/workspace/.bootstrap/tools/run_prepare_verify.sh`（自动：挪开 compiled → 清残留 → prepare+verify → 恢复 compiled）。chumpy 0.70 需 `pip install --no-build-isolation --no-deps` 安装（其 setup.py 在隔离构建环境缺 pip）。

### 用户待办清单（2026-10-25 01:14 已全部完成 ✅）

1. ~~RoboStriker~~：用户下载 GDrive 全文件夹 zip，agent 抽取前 4 组校验后推送。
2. ~~InterHuman 4 小文件~~：split×3 + LICENSE.md 下载并推送，md5/sha256 全对。
3. ~~SMPL neutral~~：官方 zip 已上传实例 `licenses/neutral_smpl.pkl`。
4. ~~免费额度~~：单次 ~7h / 总 31.5h（已记录在文档头）。

## 6. 网络可达性结论（实例出网白名单式）

| 目标 | 实例 | 本地 WSL | 结论 |
|---|---|---|---|
| github.com | ✅（偶发 HTTP/2 抖动，`-c http.version=HTTP/1.1` 可救） | ✅ | 可用 |
| download.pytorch.org | ✅ 34MB/s | ✅ | 可用 |
| pypi.org / mirrors.aliyun.com | ✅ | ✅ | 可用 |
| huggingface.co | ❌ timeout | ❌ | 双端不通 |
| hf-mirror.com | ❌ timeout | ✅ | 仅本地可用 |
| cas-bridge.xethub.hf.co（HF LFS CDN） | ✅ | ✅ | 大文件走 CDN + Range；每请求必须独立签名 URL |
| drive.google.com / 全 Google 域 | ❌ | ❌ | 双端不通，含 WebFetch |
| modelscope.cn | ✅ | ✅ | 本数据集未镜像 |

## 7. 验收对照（PHASE0 §4）

| 类别 | 检查项 | 状态 |
|---|---|---|
| 连接 | `ssh hbot` 直进 root shell；config 与 `.ts_ip` 一致 | ✅ |
| 连接 | 重启后按 §4.2 恢复 | ✅（2026-10-25 已实测） |
| 环境 | 训练 venv `pip check` 干净 + CUDA available | ✅ |
| 环境 | 评测 `validate --help` 可用 + compiled 产物 | ✅ |
| 资源 | assets 49/49 + ONNX SHA256 一致 | ✅ |
| 数据 | RoboStriker 4 组 | ✅ 官方 receipt 8/8（complete_source=false 为预期，仅取 4/43 组） |
| 数据 | InterHuman 3 片段 | ✅ 官方 receipt 10/10（含 split/LICENSE） |
| 数据 | prepared `COMPLETE.json` | ✅ PASS（11460 samples；`records/COMPLETE_1025.json`） |
| 记录 | 本文件 + `records/` | ✅（额度已记录：单次 ~7h / 总 31.5h @ 2026-10-25 01:14） |

**核心状态校验（2026-10-25 00:30）**：`VERIFY_PASS=19 VERIFY_FAIL=0`；`FROZEN_BUNDLE_OK`；`COMPILED_MANIFEST_OK schema=1`。

**恢复演练（2026-10-25 00:34）**：`post_boot.sh` 重启 tailscale/sshd 后 5 秒恢复 SSH；`bootstrap_rc=0 verify_rc=0`；`VERIFY_PASS=19 VERIFY_FAIL=0`。

**最终验收清扫（2026-10-25 01:00，重启后）**：① bashrc 环境变量 ×3 ✅；② 训练 venv `pip check` 干净 ✅；③ CUDA available（A10）✅；④ `validate --help` OK ✅；⑤ assets manifest sha256 `2d3d9459…` 与基线一致 ✅；⑥ ONNX sha256 `b0681976…` 与基线一致 ✅；⑦ compiled 产物 7 文件（manifest 校验 OK）✅；⑧ InterHuman raw 3 motions + 3 annots ✅。记录：`records/PHASE0_acceptance_20261025.txt`。磁盘（2.7）：venv 7.8G / models 1.6G / thirdparties 31M / data 6.1M，NAS 余量 996T（>25GB 达标）。

**PHASE0 全表转绿（2026-10-25 02:45）**：连接/环境/资源/数据四类检查全部通过，无阻塞项、无降级项。数据流水线端到端打通：用户浏览器下载（GDrive/SMPL 官网）→ 本地逐文件校验（profile md5/sha256）→ 分块并行推送（DERP 中继单流不稳，64MB 分块 ×4 并行可靠）→ 官方命令零网络校验落 receipt → retarget(GPU) → prepare → verify → `COMPLETE.json` PASS 回传。**下一步：进入 PHASE1 冒烟训练（PHASE1_SMOKE_TRAINING.md），其前置条件已全部满足。**

## 9. PHASE1 烟雾训练（A 线）——2026-10-25 全绿 ✅

| 项 | 结果 |
|---|---|
| prior | 100/100 epochs，runtime 48.0s，stop=完成；`best.pt`+`last.pt` 在位 |
| SL | 2/2 epochs，runtime 25.3s；`epoch_000002.pt` 在位 |
| warmup(standing) | 4 updates，runtime 141.3s（墙钟 144s），每 update 3.98s，编译+开销 ≈125.4s |
| selfplay | 4 updates，runtime 79.0s（墙钟 81s），每 update 3.16s，编译+开销 ≈66.4s |
| 对手池实测 | **snapshots=2, iterations=[0,2]**（interval=2/updates=4）→ 主方案 §2.3 推断成立，无需回填 |
| 导出 | `boxing-smoke.onnx` 生成，导出比对通过（EXPORT_RC=0） |
| validate | **status=PASS 且 chain_benchmark.passed=true，mean_ms=4.92ms**（阈值 5.0，A10） |
| 自对战 | 0:0、end_reason=TIME、hit/fall/KO 全 0、本地时延 0.59ms、~51FPS（4-update 模型符合预期） |
| 归档 | `records/{boxing-smoke-validation.json, boxing-smoke-match.json, smoke-measurements.md, boxing-smoke-{prior,sl,warmup,selfplay}.json}` |
| 坑 | ① tmux 无 conda → 用 `/mnt/workspace/miniforge3/envs/hbot-eval/bin/python` 绝对路径；② 实测编译远快于规格预估（两次 RL 调用开销 125s/66s），PHASE2 排程按保守口径不变 |
| 额度 | PHASE1 消耗 ≈0.5h GPU 时间（02:40–03:00 会话时段内） |

## 10. PHASE2 完整基线 B1——2026-10-25 凌晨全绿 ✅

| 段 | 结果 | 实耗 |
|---|---|---|
| envs 预检 | 128 envs 两模式通过（standing 4.2s/update、selfplay 4.1s/update）→ 全程 128，无偏差 | ~5min |
| prior 1000 | **290 epochs 早停（validation_plateau）**，best/last 在位，指针 `b1-prior-final.txt` | 83.6s |
| SL 19 | `epoch_000019.pt`，stop=完成 | 36.4s |
| warmup 32 | `model_32.pt`，32/32，stop=完成 | 171.6s |
| selfplay 256 | **256/256 单块完成**（官方参数 capacity=8/interval=256），`b1-final-dir.txt`=results/b1-selfplay | 1007.1s |
| pool 复核 | model_256.pt snapshots=**1**, iterations=[0] → 与代码推断一致，无需回填 | 0 |
| 导出+校验 | `boxing-b1.onnx`（<50MB）**status=PASS、mean_ms=4.79ms ≤5**（A10） | ~1min |
| 对战画像 | 4 局全 0:0/TIME/零 hit-fall-KO；自对战 3 局完全一致（确定性正常）；SL vs RL 无交手（基线偏防守，忠实记录）；时延 0.58–0.59ms | ~25min |
| 哈希 | model_256.pt `ec925eff…`；boxing-b1.onnx `e42e6f62…` | — |
| 归档 | `records/{b1-validation.json, baseline_profile.md, b1-timings.md, b1-checkpoint-hashes.txt, 4×指针, 4×b1-*.json}` | — |

- **B1 总消耗 ≈0.6h**（规格预算 14.4h 的 4%；A10 实测速度远超保守估计，selfplay 未触发分块）
- 坑：目录指针登记的 `ls | tee` 自引用竞态（tee 先建文件被 ls glob 扫到）→ 已改 `grep -v final.txt` 模式，登记逻辑写入各 p2 脚本
- 阶段 7【用户手动】平台提交：检查单 1/2/3/8 项已满足（PASS+5ms 内+<50MB+fall 0）；第 4 项窗口（北京时间 10:00–18:00）与第 6 项截图需用户醒后操作；`records/submissions.md` 待建档

## 11. PHASE3 改进与 A/B（B2）——2026-10-25 凌晨完成 ✅

| 段 | 结果 |
|---|---|
| P1（interval=32，从 B1 actor 新阶段） | 256/256 完成（1012s）；快照复核 **snapshots=8, iterations=[0,32,…,224]** 与推断一致；validate PASS / 4.96ms |
| P1 A/B vs B1 | 核查局推翻确定性假设（frames 层浮点非确定，summary 巧合相同）→ **全部 A/B 改每侧 3 局制**；6 局全 0:0/TIME → 逐级 tiebreak 全平 → **REJECT** |
| P2（config 级 task 0.9/style 0.1） | config 生效已验证（输出目录 config.json）；256/256 完成；validate PASS / 4.73ms；A/B 6 局全 0:0 → **REJECT** |
| P1b | **门控关闭**（规格：仅当 P1 被采纳才开启） |
| 循环赛（b1/p1/p2 六局） | 全 0:0/TIME，无单独冠军 → 保守取基线 |
| **最终候选** | **boxing-b1.onnx**（sha256 `e42e6f62…`）；结论「改进未奏效，证据归档」；阶段 7 不提交新模型 |
| 归档 | `records/{winrate.md, p1-validation.json, p2-validation.json, rr-summary.txt, final-candidate.txt, candidates.txt, b2-p1-selfplay.json, b2-p2-selfplay.json, selfplay-p2.json}`；远程 `results/{p1-*, p2-*, ab3b-*, rr-*, winrate.md}` |
| 回填 | ① match 确定性：PHASE3 §0.3/附录A + 主方案附录D（固定 initial_state 只保初态，frames 层非确定）；② 实测编译开销远低于预估（已在 §9 登记） |
| 预算 | B2 实耗 ≈1.2h GPU（P1 17min + P2 17min + 18 局对战 ≈36min + 导出校验），远低于 10.8h |

**关键观察（供 PHASE4/主方案决策）**：256-update 规模下所有本地对局（含 SL vs RL、全部改进分支）均为 0:0、零主动 hit——模型尚无进攻行为。这不是管线问题（管线全绿），是训练量/奖励塑造问题；要出「会打的模型」需加大训练量（如 selfplay 2048+ updates，A10 实测约 2.5h/千 updates）或调整奖励（P2 单轮权重微调 0.8→0.9 不足以改变行为）。**建议下会话优先与用户讨论加大 selfplay 规模的额度分配。**

## 12. 会话收尾状态（2026-10-25 06:10）

- 本会话完成：PHASE0 收尾 → PHASE1 全绿 → PHASE2 B1 全绿 → PHASE3 B2 完成（结论：保留基线）。
- 实例将于 ~08:15 到达 8h 单次上限被平台停止（或 GPU 空闲 1h 后自动关机）——无需干预。
- **下次开机**：用户 WebIDE 跑 `nohup bash /mnt/workspace/.remote-access/post_boot.sh > /mnt/workspace/.remote-access/post_boot.log 2>&1 &` → 说「按 PHASE0 连接」即可恢复全部环境。
- 待办（需用户）：① 平台提交 B1 模型（检查单已备好，窗口 10:00–18:00）；② 决策是否加大 selfplay 训练规模；③ PHASE4 冻结与最终提交文档尚未编写（PHASE3 结尾注明「届时另立文档」）。
- 全部关键状态已持久化于 /mnt/workspace（NAS）；本地 records/ 33 个文件为完整证据链。

## 13. PHASE4 规格落盘 + 冻结资产盘点（2026-10-25 05:20）

- **新增文档**：`PHASE4_FREEZE_FINAL.md`（v1.0）——按 PHASE3 交接指针与主方案 §7.3/§7.4 编写：阶段 0 冻结盘点 / 阶段 1 训练量决策点（A 保守 / **B 推荐**：扩 selfplay 1024–2048 updates，≈2.5–5h / C 激进）/ 阶段 2 提交节奏（10 月底冒烟提交窗口**今天即符合**）/ 阶段 3 周循环 / 阶段 4 最终验证（12/2 前）/ 阶段 5 截止日确认（12/3）/ 附录 A 故障表。
- **阶段 0 已执行 ✅**：final-candidate 指针（boxing-b1.onnx `e42e6f62…`）、current-best ckpt（`ec925eff…`）、validate（PASS/True/4.79ms）三者闭环，且与本地 `records/b1-checkpoint-hashes.txt` 完全一致；winrate/timings/profile/verdict 等 7 个关键产物全部在位。
- **代码冻结线**：12/1 起不改训练/评测代码（此前修改须走 PHASE3 附录 B patch 流程）。
- **待用户**：① 冒烟提交 B1（今天 10:00–18:00 窗口即可，检查单 8 项中 agent 可测项全部 ✅）；② 阶段 1 训练量 A/B/C 决策。

## 14. PHASE4 阶段 1 选项 B + P2-b 执行结果（2026-10-25 05:30–06:15，按用户「按推荐执行」授权）

| 实验 | 结果 |
|---|---|
| 选项 B：扩展 selfplay 1024 updates（B1 actor 新阶段，interval=32/capacity=8） | 1024/1024 完成（3823s）；validate PASS/4.83ms；A/B 3+3 全 0:0 → **REJECT** |
| P2-b：offensive 权重 50→60（代码级，分支 p2b-reward@e36823c，patch sha `41b9efae…`） | 256/256 完成（995s）；validate PASS/4.82ms；A/B 全 0:0 → **REJECT**；代码已回退 master |
| 奖励实证 | 1280+ updates 累计：strike 峰值 0.0017（噪声级）、complete_punch 峰值 1.4e-06、**offensive_hit 严格 =0**（权重 +20% 依旧） |
| 最终候选 | **维持 boxing-b1.onnx**（e42e6f62…）；改进菜单（P1/P2-a/P2-b/扩展训练/P1b 门控）**全部穷尽** |
| 记录 | winrate.md 含逐局明细与穷尽结论；records/ 38 文件 |

**给用户的战略结论**：管线与奖励管线本身全部健康（训练/导出/校验/对战零故障），但官方奖励设计下策略学不会出拳——这是「验证通过但比赛会输」的典型形态（主方案 §9 预案第 6 行）。破局三选一（均需在 12/1 代码冻结前）：① 进攻奖励稠密化/数量级提升；② standing→striking 课程；③ 提高探索噪声。**否则 B1 就是最终模型**——按 §7.4 Elo 口径它「稳定不倒地」（0 fall/KO），至少不会送分。

本会话 GPU 总消耗 ≈2.2h（B1 0.6h + B2 1.2h + 扩展/P2b 0.4h），总额度剩 ~29h。实例 ~08:15 到 8h 上限自动停止，无需干预。

## 15. 破局实验 E1：探索噪声（2026-10-25 07:11–07:44，会话末窗口）

- **E1**：`init_noise_std 0.2→0.5`（config 级 `selfplay-e1.json`，单变量），B1 actor 新阶段 256 updates → 完成（995s）、零倒地、validate PASS/4.86ms → A/B 3+3 全 0:0 → **REJECT**；strike 峰值 0.0004（低于基线噪声）、offensive 严格 =0
- **破局矩阵现状**：训练量 ✗（256→1280）｜探索 ✗（×2.5）｜奖励稠密化 **未试**（唯一剩余方向，代码级，需用户批准）
- 归档：winrate.md E1 节、`records/{b3-e1-selfplay.json, e1-validation.json, selfplay-e1.json}`；PHASE4 文档已同步回填
- 最终候选维持 **boxing-b1.onnx**；会话 ~08:14 到期自动停止，下次开机「按 PHASE0 连接」

## 16. 破局实验系列 H1–H4（2026-10-25 09:11–15:10，新 pod dsw-2295180）

用户授权后执行的奖励稠密化定向搜索（全部单变量、patch 分支流程、每轮 ~50 分钟完整闭环）：

| 实验 | 改动 | 训练侧信号 | A/B |
|---|---|---|---|
| H1 (23cc632) | vel_reward: tar_speed 1.0→0.5, σ 0.25→0.5 | **vel 激活 10×**、strike 首次非零 | REJECT |
| H1b | H1 --resume 768（累计 1024） | complete_punch 微事件出现 | REJECT |
| H2 (dcbd9b2) | + shaping σ 1.0→2.0、门 0.8→0.4 | **offensive 首次非零**（27% 更新） | REJECT |
| H2b | H2 --resume 768（累计 1536） | shaping 100% 激活；命中率 22% 平台期 | REJECT |
| H3 (0bdb85a) | shaping σ 2.0→0.5（梯度集中接触距离） | shaping 跌至 0.001、offensive 14% | REJECT |
| H4 (8446657) | σ 回 1.0（未测网格点）+ 门 0.4 | vel/shaping 100%；offensive 11% | REJECT |

**关键诊断（帧遥测）**：H2b vs B1 对局全程双机距离钉在 2.0m、0 帧进入 1.5m——训练中的行为链信号没有迁移到确定性对局（对局跑均值动作无探索噪声；且 σ=2.0 让 2m 对峙本身就能拿 shaping 分）。

**结构性结论**：奖励几何空间（sigma × 门控）已系统探测完毕，命中奖励始终停留在 1e-4 量级、无法改变 60s 对局比分。下一轮需要更大的工程改动（三选一）：① 确定性对局迁移（对局用采样动作或训练降噪声）；② 打破自对弈镜像均衡（异构对手课程）；③ 教师/脚本引导的命中课程。**最终模型维持 B1**（e42e6f62…）。

- 所有实验：validate 全 PASS（4.6–4.9ms）、零倒地、代码改动全部走 patch 分支留档（`records/p4-h*.patch`）、A/B 全部 3+3 制
- 本会话 GPU 消耗 ≈4h（6 次完整训练+判定循环）；额度剩 ~26h
- records/ 55 个文件；`results/h{1,1b,2,2b,3,4}-selfplay/` 全部 checkpoint 在 NAS 可续训

## 8. 交付物位置

- 远程：`/mnt/workspace/HoloMotion`、`/mnt/workspace/.remote-access/`、`/mnt/workspace/.bootstrap/`、`/mnt/workspace/logs/`
- 本地：`/home/czx/HBOT/deployment/{bootstrap_hbot.sh,post_boot.sh,verify_state.sh,cache_persistent.sh,connect_hbot.sh,.ts_ip,instance_snapshot.md}`、`/home/czx/HBOT/records/`

## 17. ★ native 评测验证：H2b 双 KO 官方原生控制器（2026-10-25 15:05）

- `model_3000.pt`（RoboStriker-Preview 官方 ckpt，18.8MB，sha256 `9f8bb366…`）经 api.github.com raw 端点就位 `/mnt/workspace/`（raw.githubusercontent 不通）
- **H2b（boxing-h2b.onnx，f56fc9d8…）：side0 1000:5 KO（41s）、side1 19:1000 KO（93s）——双侧双 KO 胜**；挨 5/19 拳但零倒地
- H4（H2b 重开新阶段 σ1.0）：0:16 / 17:0 双败——新阶段重置清空战斗行为
- **final-candidate / current-best 切换至 H2b 谱系**；镜像 A/B 方法论作废（对 B1 永远 0:0），evaluate_native 为有效验收
- 归档：records/native-{h2b,h4}-side{0,1}-result.json ×6

## 18. H5 防御强化实验：负结果（2026-10-25 15:18–16:15）

- 改动：defensive_hit_penalty -8→-24（跨码 --allow-resume-code-change 从 H2b 续训 512，分支 p4-h5-def@47e8726，patch `a0793feb…`）
- native 验收**双败**：side0 [1,10]、side1 [7,0]——进攻性被防御压力扼杀
- **结论：当前奖励结构下进攻性与耐打性此消彼长；H2b（-8，双 KO native）保持在激进最优点上，仍为 final-candidate**
- 本会话实验全部收官（H1–H4 + native 验证 + H5）；额度剩 ~24h；pod ~16:50 到期自动停止

