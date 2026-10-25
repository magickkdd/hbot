#!/bin/bash
# One-time (re-runnable) cache of everything needed after a system-disk reset.
set -u
B=/mnt/workspace/.bootstrap
R=/mnt/workspace/.remote-access
REPO=/mnt/workspace/HoloMotion
mkdir -p "$B"/{bin,apt,tools,checksums,env,logs}
mkdir -p /mnt/workspace/.cache/pip /mnt/workspace/.cache/conda-pkgs

# Tailscale static binaries (avoid network install on next boot)
for x in /usr/bin/tailscale /usr/sbin/tailscaled; do
  [ -x "$x" ] && cp -fL "$x" "$B/bin/$(basename "$x")"
done

# apt fallback packages (main tools; base image supplies common libs).
# git-lfs is preinstalled in this image and its 3.2.0 build is not in the
# configured Ubuntu source, so cache only the packages that can be fetched.
cd "$B/apt" || exit 1
for p in openssh-server openssh-sftp-server tmux rsync; do
  apt-get -o APT::Sandbox::User=root download "$p" >/dev/null 2>&1 || true
done

# Checksums for the frozen controller and assets
sha256sum "$REPO/models/holomotion-v2/exported/model_10500.onnx" > "$B/checksums/model_10500.onnx.sha256"
cp -f "$REPO/holoboxing/evaluation/frozen-model.json" "$B/checksums/frozen-model.json"
cat > "$B/checksums/onnx_chunks.md5" <<'EOF_MD5'
cbc156e3a3a84ec26b6d66423d57fc74  chunk_00
c569cffbb59ae07b74fb16471c0c2484  chunk_01
001ad8a71e6dd78edb59dee5afb70af5  chunk_02
32aee57d8625a2df451c878d23062cc9  chunk_03
4cf43ebeb2db1fb87a6788b3ea26891f  chunk_04
b4b7ac42c2cec47e6da05db58960f625  chunk_05
16413cd8cca99fa5bbfd48011ef191b0  chunk_06
11fe8ffd01f9cfa8fb9987a129849d3b  chunk_07
5813c5a1e7be6249f2f7f1b3a3caf1c6  chunk_08
9beaaac5bc97dade489e3452c9f3d140  chunk_09
a05424cab921433558d53a52d03505bf  chunk_10
b6d0504b50ebe97468f17ed74d56ec69  chunk_11
5078a8bc1fa789f4d98534a6d14dadd5  chunk_12
c0d8e09f05f476ec3e06505ba076da92  chunk_13
99224831341e3585b497f40c14238627  chunk_14
8c6ad4c2bdbfd7e0a37d31408d281867  chunk_15
EOF_MD5

# Environment snapshots for fast rebuild/debug
"$REPO/.venv-holoboxing/bin/python" -m pip freeze > "$B/env/pip-freeze-training.txt" 2>/dev/null || true
/mnt/workspace/miniforge3/envs/hbot-eval/bin/python -m pip freeze > "$B/env/pip-freeze-eval.txt" 2>/dev/null || true
/mnt/workspace/miniforge3/bin/conda list -n py311 --export > "$B/env/conda-py311.txt" 2>/dev/null || true
/mnt/workspace/miniforge3/bin/conda list -n hbot-eval --export > "$B/env/conda-hbot-eval.txt" 2>/dev/null || true

# Persistent cache locations for future installs
if ! grep -q 'PIP_CACHE_DIR' "$REPO/env.sh"; then
  cat >> "$REPO/env.sh" <<'EOF_ENV'
export PIP_CACHE_DIR=/mnt/workspace/.cache/pip
export CONDA_PKGS_DIRS=/mnt/workspace/.cache/conda-pkgs
EOF_ENV
fi

# Keep the recovery scripts on persistent storage (and mirrored in .remote-access)
for f in bootstrap_hbot.sh post_boot.sh verify_state.sh cache_persistent.sh; do
  [ -f "$R/$f" ] && cp -f "$R/$f" "$B/tools/$f"
done
chmod 700 "$R"/*.sh "$B/tools"/*.sh 2>/dev/null || true
chmod 600 "$B/checksums"/* "$B/env"/* "$R"/tailscaled.state "$R"/authorized_key.pub 2>/dev/null || true

ls -lh "$B/bin" "$B/apt" "$B/checksums" "$B/env"
echo CACHE_PERSISTENT_DONE
