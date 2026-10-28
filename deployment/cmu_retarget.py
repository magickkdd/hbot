"""CMU ASF/AMC boxing mocap -> G1 joint trajectories (legacy source npz).

Pipeline: parse ASF skeleton + AMC frames -> forward kinematics -> resample 50 Hz
-> map CMU body points onto G1 links -> per-frame damped-least-squares IK in
MuJoCo (root pose taken from CMU pelvis) -> write source npz accepted by
holoboxing.data.records.load_trajectory (legacy single-role schema).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCENE = "/root/hbot/HoloMotion/holoboxing/assets/arena/boxing.xml"
STANDING = "/root/hbot/HoloMotion/holoboxing/assets/initial-state.json"
ROBOT = "p1_"
DOF_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# CMU bone -> G1 body used as IK target, with solver weight.
MAP = {
    "lfemur": ("left_hip_roll_link", 2.0),
    "ltibia": ("left_knee_link", 3.0),
    "lfoot": ("left_ankle_roll_link", 3.0),
    "rfemur": ("right_hip_roll_link", 2.0),
    "rtibia": ("right_knee_link", 3.0),
    "rfoot": ("right_ankle_roll_link", 3.0),
    "upperback": ("torso_link", 1.0),
    "thorax": ("torso_link", 1.0),
    "upperneck": ("head_link", 1.0),
    "lhumerus": ("left_shoulder_roll_link", 1.0),
    "lradius": ("left_elbow_link", 2.0),
    "lwrist": ("left_wrist_roll_link", 3.0),
    "rhumerus": ("right_shoulder_roll_link", 1.0),
    "rradius": ("right_elbow_link", 2.0),
    "rwrist": ("right_wrist_roll_link", 3.0),
}


# ---------------------------------------------------------------- ASF / AMC
def parse_asf(path: Path):
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    parent, direction, length, dof = {}, {}, {}, {}

    # :bonedata  -> per-bone blocks (spans until :hierarchy)
    bi = lines.index(":bonedata")
    hi = lines.index(":hierarchy")
    name = None
    for line in lines[bi + 1:hi]:
        tok = line.split()
        if tok[0] == "name":
            name = tok[1]
        elif tok[0] == "direction" and name:
            direction[name] = np.array([float(v) for v in tok[1:4]])
        elif tok[0] == "length" and name:
            length[name] = float(tok[1])
        elif tok[0] == "axis" and name:
            label = tok[4].upper()
            if label == "XYZ":
                dof.setdefault(name, []).extend(["RX", "RY", "RZ"])
            else:
                dof.setdefault(name, []).append(label)

    # :hierarchy -> "root firstChild ..." then "parent children..."
    root = None
    for line in lines[hi + 1:]:
        tok = line.split()
        if tok[0] == "end":
            break
        if tok[0] == "begin":
            continue
        if tok[0] == "root":
            root = tok[1]
            for child in tok[2:]:
                parent[child] = root
        else:
            for child in tok[1:]:
                parent[child] = tok[0]
    return root, parent, direction, length, dof


def parse_amc(path: Path):
    """CMU AMC: integer frame marker, then 'root tx ty tz rx ry rz', then per-bone angles."""
    frames, current = [], None
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip().lstrip("\ufeff")
        if not s or s.startswith("#") or s.startswith(":"):
            continue
        tok = s.replace(",", " ").split()
        if len(tok) == 1 and tok[0].lstrip("+-").isdigit():
            if current is not None:
                frames.append(current)
            current = {"root_translation": None, "rot": {}}
            continue
        if current is None:
            continue
        vals = np.array([float(v) for v in tok[1:]])
        if tok[0] == "root":
            current["root_translation"] = vals[:3].copy()
            if vals.size > 3:
                current["rot"]["root"] = vals[3:].copy()
        else:
            current["rot"][tok[0]] = vals.copy()
    if current is not None:
        frames.append(current)
    for fr in frames:
        if fr["root_translation"] is None:
            fr["root_translation"] = np.zeros(3)
    return frames


def axis_rotation(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "RX":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "RY":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "RZ":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


# CMU/Vicon uses Y-up; rotate to MuJoCo's Z-up: (x, y, z) -> (x, -z, y).
_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def forward_kinematics(asf, frames):
    root, parent, direction, length, dof = asf
    order = []
    for name in direction:
        chain, cur = [], name
        while cur in parent:
            chain.append(cur)
            cur = parent[cur]
        chain.append(cur)
        order.append((len(chain), name))
    order = [n for _, n in sorted(order)]

    out_pos, out_rot = [], []
    for fr in frames:
        world_rot, world_pos = {}, {}
        for name in order:
            local = np.eye(3)
            axes = dof.get(name, [])
            if name == root:
                vals = fr["rot"].get("root")
            else:
                vals = fr["rot"].get(name)
            if axes:
                if vals is None:
                    vals = np.zeros(len(axes))
                if len(vals) != len(axes):
                    vals = np.resize(vals, len(axes))
                for ax, v in zip(axes, vals):
                    local = local @ axis_rotation(ax, float(v))
            if name == root:
                p = fr["root_translation"].copy()
                world_rot[name] = local
                world_pos[name] = p
            else:
                par = parent[name]
                p = world_pos[par] + world_rot[par] @ (direction[name] * length[name])
                world_rot[name] = world_rot[par] @ local
                world_pos[name] = p
        out_pos.append({k: _UP @ v for k, v in world_pos.items()})
        out_rot.append({k: _UP @ v for k, v in world_rot.items()})
    return out_pos, out_rot


def _smooth(values, win):
    if win <= 1 or len(values) < win:
        return values
    kernel = np.ones(win) / win
    pad = win // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])],
        1,
    )[: len(values)]


def _times(n, src_fps, dst_fps):
    t_src = np.arange(n) / src_fps
    t_dst = np.arange(int(np.floor(t_src[-1] * dst_fps))) / dst_fps
    return t_src, t_dst


def resample_positions(positions, src_fps, dst_fps):
    t_src, t_dst = _times(len(positions), src_fps, dst_fps)
    out = {}
    for name in positions[0]:
        flat = np.stack([p[name] for p in positions])
        out[name] = np.stack([np.interp(t_dst, t_src, flat[:, i]) for i in range(3)], 1)
    return out


def resample_rotations(rotmats, src_fps, dst_fps):
    t_src, t_dst = _times(len(rotmats), src_fps, dst_fps)
    out = {}
    for name in rotmats[0]:
        flat = np.stack([r[name] for r in rotmats])
        out[name] = np.stack(
            [np.stack([np.interp(t_dst, t_src, flat[:, i, j]) for j in range(3)], 1)
             for i in range(3)], 1)
    return out


# ---------------------------------------------------------------- retarget
def build_targets(positions, rot, scale):
    targets = {}
    for cmu_name, (g1_name, weight) in MAP.items():
        targets[g1_name] = (positions[cmu_name] * scale @ rot.T, weight)
    return targets


def root_pose(positions, rot, scale):
    pelvis = positions["lhipjoint"] * scale @ rot.T
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    quats = np.zeros((len(pelvis), 4))
    quats[:, 0] = np.cos(yaw / 2)
    quats[:, 3] = np.sin(yaw / 2)
    return pelvis, quats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asf", required=True)
    ap.add_argument("--amc-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--src-fps", type=float, default=120.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="from_idx", type=int, default=0, help="skip first N files")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files")
    ap.add_argument(
        "--freeze-lower",
        action="store_true",
        help="lock legs, torso and root to the official standing pose; retarget arms only",
    )
    args = ap.parse_args()

    import mujoco

    asf = parse_asf(Path(args.asf))
    files = sorted(Path(args.amc_dir).glob("*.amc"))
    files = files[args.from_idx:]
    if args.limit:
        files = files[: args.limit]
    if args.only:
        keep = args.only.split(",")
        files = [f for f in files if any(f.stem.endswith(k) for k in keep)]
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    joint_dof, joint_qpos = [], []
    for name in DOF_ORDER:
        jid = model.joint(ROBOT + name).id
        joint_dof.append(int(model.jnt_dofadr[jid]))
        joint_qpos.append(int(model.jnt_qposadr[jid]))
    body_ids = {n: model.body(ROBOT + n).id for n, _ in MAP.values()}
    ranges = np.array([model.jnt_range[model.joint(ROBOT + n).id] for n in DOF_ORDER])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"clips={len(files)}", flush=True)

    jacp = np.zeros((3, model.nv))
    written = 0
    for path in files:
        frames = parse_amc(path)
        positions, rotmats = forward_kinematics(asf, frames)
        positions = resample_positions(positions, args.src_fps, 50.0)
        rotmats = resample_rotations(rotmats, args.src_fps, 50.0)
        n = len(positions["lhipjoint"])
        if n < 20:
            print("skip short", path.name, n, flush=True)
            continue

        # yaw so the subject faces +x at the first frame
        hips = positions["lfemur"][0] - positions["rfemur"][0]
        right = hips / (np.linalg.norm(hips) + 1e-9)
        forward = np.cross(np.array([0, 0, 1.0]), right)
        yaw_ref = float(np.arctan2(forward[1], forward[0]))
        rot = np.array([[np.cos(yaw_ref), -np.sin(yaw_ref), 0],
                        [np.sin(yaw_ref), np.cos(yaw_ref), 0],
                        [0, 0, 1]])
        # height scale: CMU head-to-foot vs G1 nominal height
        feet = min(positions[k][0][2] for k in ("lfoot", "rfoot"))
        cmu_h = positions["head"][0][2] - feet
        scale = 1.30 / max(float(cmu_h), 1e-6)
        targets = build_targets(positions, rot, scale)
        pelvis, quats = root_pose(positions, rot, scale)
        base = int(model.jnt_qposadr[model.joint(ROBOT + "floating_base").id])
        # 脚底贴地：用官方站立姿态的踝高对齐首帧 CMU 踝高，整段恒定偏移
        data.qpos[:] = model.qpos0
        standing = json.loads(Path(STANDING).read_text())["qpos"][0]
        data.qpos[base:base + 7] = standing[:7]
        data.qpos[joint_qpos] = standing[7:]
        mujoco.mj_forward(model, data)
        za_rest = float(data.xpos[body_ids["left_ankle_roll_link"]][2])
        stand_pos = {name: data.xpos[bid].copy() for name, bid in body_ids.items()}
        stand_pos["pelvis"] = data.xpos[model.body(ROBOT + "pelvis").id].copy()
        cmu_ankle = min(targets["left_ankle_roll_link"][0][0, 2],
                        targets["right_ankle_roll_link"][0][0, 2])
        dz = za_rest - cmu_ankle
        if written == 0:
            print("DBG za_rest", round(za_rest, 3),
                  "cmu_ankle_l", round(float(targets["left_ankle_roll_link"][0][0, 2]), 3),
                  "cmu_ankle_r", round(float(targets["right_ankle_roll_link"][0][0, 2]), 3),
                  "pelvis0", round(float(pelvis[0, 2]), 3), "dz", round(float(dz), 3), flush=True)
        pelvis[:, 2] += dz
        for g1_name in targets:
            pts, weight = targets[g1_name]
            targets[g1_name] = (pts + np.array([0.0, 0.0, dz]), weight)
        if args.freeze_lower:
            # 冻结下肢与根位移：拳击示范只需上肢，冻结后冻结控制器不会因重心漂移摔倒
            for g1_name in ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
                            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
                            "torso_link"):
                pts, weight = targets[g1_name]
                targets[g1_name] = (np.tile(stand_pos[g1_name], (n, 1)), weight)
            pelvis[:] = stand_pos["pelvis"] + np.array([0.0, 0.0, 0.0])
            quats[:] = np.array([1.0, 0.0, 0.0, 0.0])

        qpos = np.zeros((n, 29), np.float64)
        prev = np.zeros(29)
        for t in range(n):
            data.qpos[:] = model.qpos0
            data.qpos[base:base + 3] = pelvis[t]
            data.qpos[base + 3:base + 7] = quats[t]
            data.qpos[joint_qpos] = prev
            for _ in range(12):
                mujoco.mj_forward(model, data)
                jtj = np.eye(29) * 1e-3
                jte = np.zeros(29)
                for g1_name, (pts, weight) in targets.items():
                    bid = body_ids[g1_name]
                    err = (pts[t] - data.xpos[bid]) * weight
                    mujoco.mj_jacBody(model, data, jacp, None, bid)
                    j = jacp[:, joint_dof] * weight
                    jtj += j.T @ j
                    jte += j.T @ err
                dq = np.linalg.solve(jtj + 1e-2 * np.eye(29), jte)
                prev = np.clip(prev + dq, ranges[:, 0], ranges[:, 1])
                data.qpos[joint_qpos] = prev
            qpos[t] = prev

        # velocities
        qpos = _smooth(qpos, 5)
        qpos = _smooth(qpos, 5)
        delta = np.diff(qpos, axis=0)
        delta = np.clip(delta, -0.10, 0.10)
        qpos = np.concatenate([qpos[:1], qpos[:1] + np.cumsum(delta, 0)], 0)
        jvel = np.zeros_like(qpos)
        jvel[1:] = (qpos[1:] - qpos[:-1]) * 50.0
        ang = np.zeros((n, 3))
        ang[:, 2] = 0.0

        stem = path.stem
        out = out_dir / f"{stem}.npz"
        np.savez_compressed(
            out,
            joint_names=np.asarray(DOF_ORDER),
            joint_pos=qpos.astype(np.float32),
            joint_vel=jvel.astype(np.float32),
            body_pos_w=pelvis[:, None, :].astype(np.float32),
            body_quat_w=quats[:, None, :].astype(np.float32),
            body_ang_vel_w=ang[:, None, :].astype(np.float32),
            fps=np.asarray(50),
        )
        written += 1
        if written % 5 == 0:
            print("written", written, stem, qpos.shape, flush=True)
    print("DONE", written)


if __name__ == "__main__":
    main()
