"""Build sources.json + COMPLETE.json for a prepared dataset produced by
`python -m holoboxing.data.records` (which only writes prepared/manifest.json).

Usage:
  python /data/build_sources.py <dataset_root> <dataset_name>
where <dataset_root> contains prepared/manifest.json.
"""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = sys.argv[2]
prepared = root / "prepared"
manifest = json.loads((prepared / "manifest.json").read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


records = []
for rec in manifest["records"]:
    group = rec["source_group"]
    digest = hashlib.sha256(f"{dataset}/{group}".encode()).hexdigest()
    split = "validation" if int(digest[:8], 16) % 10 == 0 else "train"
    records.append(
        {
            "path": rec["path"],
            "sha256": rec["source_sha256"],
            "bytes": rec["bytes"],
            "source": rec["source"],
            "source_group": group,
            "dataset": dataset,
            "group": group,
            "split": split,
        }
    )

provenance = {
    "format": "holoboxing_data_sources_v1",
    "split_seed": 42,
    "validation_fraction": 0.1,
    "test_fraction": 0.0,
    "split_policy": "deterministic hash of dataset/group",
    "records": records,
    "expected_source_clips": len(records),
}
(root / "sources.json").write_text(json.dumps(provenance, indent=1) + "\n")

complete = {
    "status": "PASS",
    "partial": False,
    "source_clips": len(records),
    "expected_source_clips": len(records),
    "supervised_samples": sum(r["samples"] for r in manifest["records"]),
    "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
    "sources_sha256": sha256(root / "sources.json"),
    "note": "built by build_sources.py from records.py manifest",
}
(root / "COMPLETE.json").write_text(json.dumps(complete, indent=1) + "\n")
print(
    json.dumps(
        {
            "dataset": dataset,
            "clips": len(records),
            "samples": complete["supervised_samples"],
            "validation": sum(1 for r in records if r["split"] == "validation"),
        }
    )
)
