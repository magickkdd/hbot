import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else 'results/b5a-selfplay'
p = os.path.join(d, 'metrics.jsonl')
rows = [json.loads(line) for line in open(p)]
start = rows[0]['iteration'] - 1
last = rows[-1]
key = 'ego/reward/impulse_strike_reward'
vals = [r.get(key, 0.0) for r in rows]
n = len(vals)
nz = sum(v != 0 for v in vals)
recent = vals[-100:]
print('iter=%d delta=%d rows=%d nz=%d/%d (%.1f%%) last100=%d/%d' % (
    last['iteration'], last['iteration'] - start, n, nz, n,
    100.0 * nz / max(1, n), sum(v != 0 for v in recent), len(recent)))
print('shaping=%.5f vel=%.5f fallen=%.4f impulse=%.4f' % (
    last.get('ego/reward/active_hit_shaping_reward', 0.0),
    last.get('ego/reward/vel_reward', 0.0),
    last.get('ego/fallen_fraction', 0.0),
    last.get(key, 0.0)))

cpath = os.path.join(d, 'complete.json')
if os.path.exists(cpath):
    c = json.load(open(cpath))
    tr = c.get('training_result', {}) or {}
    prog = tr.get('progress', {}) or {}
    budget = tr.get('budget', {}) or {}
    print('complete updates_completed=%s progress_completed=%s stop_reason=%s' % (
        c.get('updates_completed'), prog.get('completed_updates'),
        budget.get('stop_reason')))
