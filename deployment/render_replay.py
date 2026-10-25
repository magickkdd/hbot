"""Render a holoboxing match.json (frames of kinematic state) to PNG frames, offscreen (EGL).

Usage: python render_replay.py <match.json> <out_dir> [label]
Assumes scene XML at thirdparties/HoloMotion_assets/holoboxing/arena/boxing.xml
(robot order: red first, blue second; quaternion order wxyz — both verified 2026-10-25).
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    os.system(f"{sys.executable} -m pip install -q pillow")
    from PIL import Image, ImageDraw, ImageFont

MATCH = sys.argv[1]
OUT_DIR = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else "H2b (RED) vs RoboStriker (BLUE)"
XML = "/mnt/workspace/HoloMotion/thirdparties/HoloMotion_assets/holoboxing/arena/boxing.xml"

m = json.load(open(MATCH))
frames = m["frames"]
summary = m.get("summary", {})
winner = summary.get("winner")
end_reason = summary.get("end_reason", "")

model = mujoco.MjModel.from_xml_path(XML)
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
data = mujoco.MjData(model)
assert model.nq == 72, f"unexpected nq={model.nq}"
free = [model.jnt_qposadr[j] for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
assert len(free) == 2, f"expected 2 free joints, got {len(free)}"

print("frame0 root_z:", [round(frames[0]["root_pos"][k][2], 3) for k in (0, 1)])  # sanity: ~0.7-0.9

renderer = mujoco.Renderer(model, height=720, width=1280)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


f_small, f_big = font(22), font(72)
os.makedirs(OUT_DIR, exist_ok=True)
n = len(frames)

for i, fr in enumerate(frames):
    for k, adr in enumerate(free):
        data.qpos[adr:adr + 3] = fr["root_pos"][k]
        data.qpos[adr + 3:adr + 7] = fr["root_quat"][k]          # wxyz，直接写
        data.qpos[adr + 7:adr + 7 + 29] = fr["dof_pos"][k]
    p0 = np.array(fr["root_pos"][0]); p1 = np.array(fr["root_pos"][1])
    mid = (p0 + p1) / 2
    cam.lookat[:] = [mid[0] * 0.5, mid[1] * 0.5, 0.95]
    cam.distance = 5.2
    cam.elevation = -12
    cam.azimuth = 130
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam, opt)
    img = Image.fromarray(renderer.render())
    d = ImageDraw.Draw(img)
    sc = fr.get("score", [0, 0])
    d.text((16, 12), f"{LABEL}   t={i / 50:5.2f}s   RED {sc[0]} : {sc[1]} BLUE",
           font=f_small, fill=(255, 255, 0))
    if winner is not None and i >= n - 60:
        side = "RED WINS" if winner == 0 else "BLUE WINS"
        d.text((470, 320), f"K.O.  {side}", font=f_big,
               fill=(255, 70, 70) if winner == 0 else (80, 140, 255))
    elif winner is None and i >= n - 60 and end_reason:
        d.text((520, 320), end_reason, font=f_big, fill=(255, 255, 0))
    img.save(f"{OUT_DIR}/f{i:05d}.png")

renderer.close()
print("DONE frames:", n)
