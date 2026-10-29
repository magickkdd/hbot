"""按 (future frame, link) 拆解策略对示范目标的误差，定位 0.2m 来自哪里。

用法: bc_breakdown.py <checkpoint> <prepared_dir> [batch]
link 顺序: 0=pelvis 1=左腕 2=右腕 3=左脚踝 4=右脚踝
"""

import glob
import sys

import numpy as np
import torch

from holoboxing.training.export import load_policy

ckpt, prepared = sys.argv[1], sys.argv[2]
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

policy = load_policy(ckpt).eval()
total = np.zeros((5, 5), dtype=np.float64)
worst = np.zeros((5, 5), dtype=np.float64)
count = 0

with torch.no_grad():
    for path in sorted(glob.glob(prepared + "/*.npz")):
        data = np.load(path)
        obs = torch.from_numpy(data["obs"].astype(np.float32))
        target = torch.from_numpy(data["target_pose"].astype(np.float32))
        for start in range(0, len(obs), batch):
            o = obs[start : start + batch]
            t = target[start : start + batch].reshape(-1, 5, 5, 9)[..., :3]
            p = policy(o).reshape(-1, 5, 5, 9)[..., :3]
            err = (p - t).norm(dim=-1).numpy()  # [B,5,5]
            total += err.sum(axis=0)
            worst = np.maximum(worst, err.max(axis=0))
            count += len(o)

print("samples", count)
print("mean error_m by future_frame row / link col")
print("        pelvis  Lwrist  Rwrist  Lankle  Rankle")
for i in range(5):
    row = " ".join(f"{total[i, j] / count:7.4f}" for j in range(5))
    print(f"frame{i} {row}")
print("overall mean %.4f  worst %.4f" % (total.sum() / count / 25, worst.max()))
