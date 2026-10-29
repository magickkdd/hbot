"""Align the training reward with the competition scoring rule (v2).

Why v2: v1 projected the raw force magnitude, so grazing contacts (the arm
sliding along the opponent torso) counted as hits while the evaluator only
scores the contact-normal component.  Stage 04b therefore showed hits in
training but scored 0 in every native match.  v2 projects the measured
wrist->torso force onto the wrist->torso-centre direction and keeps only the
compressive part, which is much closer to ``impulse_hit_rules``.

Also:
* adds ``scored_hit_taken_reward`` (symmetric -1 per hit taken);
* drops ``impulse_strike_reward`` weight 200 -> 20 so it stays a dense
  gradient instead of the dominant objective.

Usage: python apply_scored_hit.py [repo_root]
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/root/hbot/HoloMotion")

IMPULSE_ANCHOR = """    per_hand = (force[:, :IMPULSE_WINDOW_STEPS, 0].norm(dim=-1) * PHYSICS_DT).sum(dim=1)  # [W,2]
    return per_hand.max(dim=-1).values
"""

V1_FUNC = '''
SCORED_HIT_THRESHOLD_NS = 5.0
HIT_COOLDOWN_STEPS = 40  # 0.2 s at the 50 Hz control step


def scored_hit_reward(env, sensor_cfg, asset_cfg=None, threshold=SCORED_HIT_THRESHOLD_NS):
    """Competition-scored hits: 20 ms impulse >= threshold with a per-hand cooldown."""
    force = env.scene.sensors[sensor_cfg.name].data.force_matrix_w_history  # [W,6,1,2,3]
    per_hand = (force[:, :IMPULSE_WINDOW_STEPS, 0].norm(dim=-1) * PHYSICS_DT).sum(dim=1)  # [W,2]
    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    # 按传感器分键：进攻（对方躯干）与被击中（自方躯干）各用各的冷却
    state = env.punch_state.setdefault("scored_" + sensor_cfg.name, {})
    if "hit_cooldown" not in state:
        state["hit_cooldown"] = torch.zeros_like(per_hand, dtype=torch.long)
    cooldown = state["hit_cooldown"]
    hit = (per_hand >= threshold) & (cooldown <= 0)
    cooldown.sub_(1).clamp_(min=0)
    cooldown[hit] = HIT_COOLDOWN_STEPS
    return hit.float().sum(dim=1)
'''

V2_FUNC = '''
SCORED_HIT_THRESHOLD_NS = 5.0
HIT_COOLDOWN_STEPS = 40  # 0.2 s at the 50 Hz control step
SCORED_HIT_V2 = True


def scored_hit_reward(
    env,
    sensor_cfg,
    asset_cfg=None,
    threshold=SCORED_HIT_THRESHOLD_NS,
    attacker="own",
):
    """Competition-scored hits: 20 ms *normal* impulse >= threshold.

    ``attacker="own"`` scores our punches on the opponent torso;
    ``attacker="other"`` scores the opponent's punches on ours.
    """
    own = env.scene[asset_cfg.name]
    other = env.get_opponent(asset_cfg.name)
    hitter, target = (own, other) if attacker == "own" else (other, own)
    torso = target.body_names.index("torso_link")
    hands = [
        hitter.body_names.index(name)
        for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    torso_pos = target.data.body_pos_w[:, torso, :3]
    wrist_pos = hitter.data.body_pos_w[:, hands, :3]
    direction = torso_pos[:, None, :] - wrist_pos
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-6)  # [W,2,3]
    force = env.scene.sensors[sensor_cfg.name].data.force_matrix_w_history  # [W,6,1,2,3]
    window = force[:, :IMPULSE_WINDOW_STEPS, 0]  # [W,4,2,3]
    # 压缩方向的分量（取绝对值，避免传感器力符号约定相反）
    normal = (window * direction[:, None]).sum(-1).abs()  # [W,4,2]
    per_hand = normal.sum(dim=1) * PHYSICS_DT  # [W,2]
    if not hasattr(env, "punch_state"):
        env.punch_state = {}
    state = env.punch_state.setdefault("scored_" + sensor_cfg.name, {})
    if "hit_cooldown" not in state:
        state["hit_cooldown"] = torch.zeros_like(per_hand, dtype=torch.long)
    cooldown = state["hit_cooldown"]
    hit = (per_hand >= threshold) & (cooldown <= 0)
    cooldown.sub_(1).clamp_(min=0)
    cooldown[hit] = HIT_COOLDOWN_STEPS
    return hit.float().sum(dim=1)
'''

WEIGHT_OLD = '''            "impulse_strike_reward",
            200.0,'''
WEIGHT_NEW = '''            "impulse_strike_reward",
            20.0,'''

V1_TERMS = '''        (
            "scored_hit_reward",
            50.0,
            rewards.scored_hit_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + other + "_torso"),
                threshold=5.0,
            ),
        ),
        (
            "scored_hit_taken_reward",
            -50.0,
            rewards.scored_hit_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + agent + "_torso"),
                threshold=5.0,
            ),
        ),
'''

V2_TERMS = '''        (
            "scored_hit_reward",
            50.0,
            rewards.scored_hit_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + other + "_torso"),
                threshold=5.0,
                attacker="own",
            ),
        ),
        (
            "scored_hit_taken_reward",
            -50.0,
            rewards.scored_hit_reward,
            dict(
                asset_cfg=own,
                sensor_cfg=entity("contact_sensor_" + agent + "_torso"),
                threshold=5.0,
                attacker="other",
            ),
        ),
'''

REGISTER_ANCHOR = '''        (
            "complete_punch_reward",'''


def patch(path, transform):
    text = path.read_text()
    updated, note = transform(text)
    if updated != text:
        path.write_text(updated)
    return note


def rewards_py(text):
    if "SCORED_HIT_V2 = True" in text:
        return text, "v2"
    if V1_FUNC in text:
        return text.replace(V1_FUNC, V2_FUNC, 1), "v1->v2"
    assert IMPULSE_ANCHOR in text, "rewards.py: impulse anchor missing"
    return text.replace(IMPULSE_ANCHOR, IMPULSE_ANCHOR + V2_FUNC, 1), "added"


def mujoco_rewards_py(text):
    if 'attacker="own"' in text:
        return text, "v2"
    if V1_TERMS in text:
        return text.replace(V1_TERMS, V2_TERMS, 1), "v1->v2"
    assert WEIGHT_OLD in text, "mujoco_rewards.py: impulse weight anchor missing"
    assert REGISTER_ANCHOR in text, "mujoco_rewards.py: register anchor missing"
    text = text.replace(WEIGHT_OLD, WEIGHT_NEW, 1)
    return text.replace(REGISTER_ANCHOR, V2_TERMS + REGISTER_ANCHOR, 1), "added"


notes = {
    "rewards.py": patch(
        ROOT / "holoboxing/training/tasks/combat/mdp/rewards.py", rewards_py
    ),
    "mujoco_rewards.py": patch(
        ROOT / "holoboxing/training/envs/mujoco_rewards.py", mujoco_rewards_py
    ),
}
print("SCORED_HIT", notes)
