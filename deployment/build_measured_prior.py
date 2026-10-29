"""把 prepared 数据集的实测 target_pose 转成 prior-data（reference_only_225 格式）。

原因：阶段01 先验原先训练在「参考目标」上（自身重建 2.4cm），而阶段02 BC 的
目标是「实测 rollout」（先验重建 13–51cm），解码器成了 BC 的天花板。
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

prepared = Path(sys.argv[1])
out = Path(sys.argv[2])
dataset = sys.argv[3] if len(sys.argv) > 3 else "b12-measured"
src_manifest = json.loads((prepared / "manifest.json").read_text())
clips = sorted(prepared.glob("*.npz"))
out.mkdir(parents=True, exist_ok=True)

records, sources = [], []
for index, path in enumerate(clips):
    data = np.load(path)
    target = data["target_pose"].astype(np.float32)
    dest = out / f"reference-{index:05d}.npz"
    np.savez_compressed(dest, target_pose=target)
    blob = dest.read_bytes()
    samples = int(len(target))
    roles = 2 if samples % 2 == 0 else 1
    frames = samples // roles + 5
    records.append(
        {
            "path": dest.name,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "source_path": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dataset": dataset,
            "group": path.stem,
            "split": "validation" if index == len(clips) - 1 else "train",
            "samples": samples,
            "frames": frames,
            "roles": roles,
        }
    )
    sources.append({"source": path.name, "samples": int(len(target))})

manifest = {
    "format": "holoboxing_source_reference_windows_v2",
    "asset_identity": src_manifest["asset_identity"],
    "control_hz": src_manifest.get("control_hz", 50),
    "future_offsets": [1, 2, 3, 4, 5],
    "coordinate_frame": "source_current_pelvis_full_3d",
    "input": "reference_only_225",
    "synthetic_opponent_samples": 0,
    "split_seed": 42,
    "validation_fraction": 0.0,
    "test_fraction": 0.0,
    "excluded_groups": [],
    "held_out_test_groups": [],
    "records": records,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"clips": len(records), "samples": sum(r["samples"] for r in records), "out": str(out)}))
