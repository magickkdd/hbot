#!/usr/bin/env python3
"""match.json -> 事件级命中统计 + 倒地归因（B5 执行规格 §4.5，v2.1）。

用法: python3 fall_attrib.py <match.json> <我方robot索引0/1>
robot 索引：场景红方在前(0)、蓝方在后(1)；candidate_side 即我方索引。

要点（winrate.md ★★ 翻案教训）：
- 我方/对方有效命中直接数 HIT 事件（含 actor/target/peak_force），不做分数分解；
- KO 事件的 actor 是"受益方"不是"因果方"（对手自倒时 actor 也是我方）——
  因果判定 = FALL 前 2s 内我方 HIT ≥1 且倒地瞬间距离 <0.7m，否则记自倒；
- 同时检测我方倒地（FALL target == 我方）。
"""
import json
import math
import sys

HIT_WINDOW_S = 2.0   # FALL 前多少秒内的我方 HIT 算因果证据
CAUSAL_DIST_M = 0.7  # 倒地瞬间双方根部水平距离阈值


def main(path, us):
    m = json.load(open(path))
    fr = m["frames"]
    nat = 1 - us

    events = [e for f in fr for e in (f.get("events") or [])]
    hits = [e for e in events if e.get("type") == "HIT"]
    falls = [e for e in events if e.get("type") == "FALL"]
    kos = [e for e in events if e.get("type") == "KO"]
    our_hits = [e for e in hits if e.get("actor") == us]
    their_hits = [e for e in hits if e.get("actor") == nat]
    our_falls = [e for e in falls if e.get("target") == us]
    their_falls = [e for e in falls if e.get("target") == nat]

    score = fr[-1]["score"]
    print(f"frames={len(fr)} 比分 {score[us]}:{score[nat]}")
    print(f"有效命中(HIT事件): 我方={len(our_hits)} 对方={len(their_hits)}")
    if our_hits:
        pf = max(e.get("peak_force", 0) for e in our_hits)
        print(f"  我方命中峰值力={pf:.0f}N")
    print(f"倒地: 我方={len(our_falls)}次 对方={len(their_falls)}次；KO事件数={len(kos)}"
          f"（KO的actor是受益方，不代表因果）")
    print(f"我方 z_min={min(f['root_pos'][us][2] for f in fr):.3f}（倒地线 0.30）")

    for e in their_falls:
        t = e.get("round_elapsed_s") or e.get("sim_time_s")
        i = min(range(len(fr)), key=lambda k: abs(k / 50.0 - t))
        f = fr[i]
        d = math.hypot(
            f["root_pos"][us][0] - f["root_pos"][nat][0],
            f["root_pos"][us][1] - f["root_pos"][nat][1],
        )
        pre_hits = [h for h in our_hits
                    if 0 <= t - (h.get("round_elapsed_s") or h.get("sim_time_s", 0)) <= HIT_WINDOW_S]
        if d > 1.0:
            verdict = "自倒（距离>1.0m，接触不可能）"
        elif pre_hits and d <= CAUSAL_DIST_M:
            verdict = f"我方打出（倒地前{HIT_WINDOW_S:.0f}s内我方{len(pre_hits)}次命中+距离{d:.2f}m）"
        elif d <= CAUSAL_DIST_M:
            verdict = "存疑接触（近距但倒地前无我方命中，可能推挤）"
        else:
            verdict = "大概率自倒（0.7-1.0m 且无命中佐证）"
        print(f"对方倒地@{t:.2f}s 距离={d:.2f}m → {verdict}")

    if not their_falls:
        print("对方未倒地")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
