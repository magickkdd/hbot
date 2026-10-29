"""留一 + 时间邻域排除的近邻测试：排除同 clip 内 ±GAP 帧，避免相邻帧泄漏。"""
import glob, sys
import numpy as np

prepared = sys.argv[1]
queries = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
gap = int(sys.argv[3]) if len(sys.argv) > 3 else 60
obs_all, pos_all, owners = [], [], []
for clip, path in enumerate(sorted(glob.glob(prepared + "/*.npz"))):
    d = np.load(path)
    o = d["obs"].astype(np.float32)
    obs_all.append(o)
    pos_all.append(d["target_pose"].astype(np.float32).reshape(-1, 5, 5, 9)[..., :3])
    owners.append(np.full(len(o), clip, dtype=np.int32))
O = np.concatenate(obs_all); P = np.concatenate(pos_all); W = np.concatenate(owners)
rng = np.random.default_rng(0)
idx = rng.choice(len(O), queries, replace=False)
pad = (O ** 2).sum(1)
err = np.zeros(queries)
for start in range(0, queries, 256):
    qi = idx[start:start + 256]
    q = O[qi]
    dist = (q ** 2).sum(1)[:, None] + pad[None] - 2 * q @ O.T
    mask = (W[None, :] == W[qi][:, None]) & (np.abs(np.arange(len(O))[None, :] - qi[:, None]) <= gap)
    dist[mask] = np.inf
    nn = dist.argmin(1)
    err[start:start + 256] = np.linalg.norm(P[nn] - P[qi], axis=-1).mean(axis=(-1, -2))
print("pool", O.shape, "gap", gap)
print("NN(gap) err_m mean %.4f median %.4f p10 %.4f p90 %.4f" % (err.mean(), np.median(err), np.percentile(err, 10), np.percentile(err, 90)))
