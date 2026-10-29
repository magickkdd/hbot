"""按拳速分箱统计先验(解码器)的重建位置误差。"""
import glob, sys
import numpy as np
import torch
from holoboxing.training.motion_prior import load_prior

ckpt = sys.argv[1]
prepared = sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
model, saved = load_prior(ckpt, device=device)
model.eval()
edges = [0, 1, 2, 3, 5, 1e9]
sums = np.zeros(len(edges) - 1)
counts = np.zeros(len(edges) - 1)
with torch.no_grad():
    for path in sorted(glob.glob(prepared + "/*.npz")):
        tp = torch.from_numpy(np.load(path)["target_pose"].astype(np.float32)).to(device)
        latent = model.encode(tp)
        pred = model.decoder(latent)
        a = pred.reshape(-1, 5, 5, 9)[..., :3]
        b = tp.reshape(-1, 5, 5, 9)[..., :3]
        err = (a - b).norm(dim=-1).mean(-1)  # [N,5] 每帧五连杆平均
        wrist = b[:, 0, 1:3]  # [N,2,3] 当前帧双腕
        step = torch.zeros(len(tp) - 1, device=device)
        step = (wrist[1:] - wrist[:-1]).norm(dim=-1).mean(-1) * 50.0
        step = torch.cat((torch.zeros(1, device=device), step))
        per = err.mean(-1).cpu().numpy()
        sp = step.cpu().numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sp >= lo) & (sp < hi)
            sums[len([e for e in edges if e <= lo]) - 1] += per[m].sum()
            counts[len([e for e in edges if e <= lo]) - 1] += m.sum()
for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    if counts[i]:
        print(f"speed {lo:>4.0f}-{hi if hi < 1e9 else 'inf':>4} m/s  n={int(counts[i]):6d}  err_m={sums[i]/counts[i]:.4f}")
