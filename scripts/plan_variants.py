"""Pre-registration planner check: run the planner with a model saved by scripts/diagnose.py on the 16
validation days (ids 9000-9015), optionally with planner options (horizon=4, samples=16, ...).
Usage: python scripts/plan_variants.py TAG VARIANT D [key=value ...]"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from collections import Counter

import numpy as np

from noisefloor.agents.wm_agent import WMAgent
from noisefloor.config import sim_config
from noisefloor.data.collect import level_key, world_from_meta
from noisefloor.eval.harness import run_episodes
from noisefloor.govern.gate import Gate
from noisefloor.models.train import load
from noisefloor.sim.loads import make_profile

tag, variant, d = sys.argv[1], sys.argv[2], float(sys.argv[3])
opts = dict(kv.split("=") for kv in sys.argv[4:])
cfg = sim_config()
SCR = ROOT / ".checks"
mdir = SCR / "diag_models" / f"{tag}_{variant}_{level_key('main', d)}"
model = load(mdir)
world = world_from_meta(json.loads((mdir / "config.json").read_text())["world"])
kw = {k: (float(v) if "." in v else int(v)) for k, v in opts.items()}
agent = WMAgent(model, cfg, seed=0, **kw)
ids = list(range(9000, 9016))
fams = cfg["families"]
profiles = [make_profile(fams[i % 4], np.random.default_rng(i), cfg) for i in ids]
acts, planned = [], np.zeros((16, 287))
def rec(t, i, dec):
    acts.append(dec["executed"]); planned[i, t] = agent.last_pred_cost[i]
t0 = time.time()
res = run_episodes(agent, profiles, ids, np.full(16, 10), cfg, Gate.load("gate_v1.json"), world=world, record=rec)
w = 0.97 ** np.arange(8)
real = np.stack([(res.costs[:, t:t + 8] * w).sum(1) for t in range(279)], 1)
print(json.dumps({"variant": variant, "d": d, "opts": opts, "J": round(float(np.mean([r.J for r in res.results])), 1),
                  "viol": float(np.mean([r.violation_steps for r in res.results])),
                  "replicas": round(float(np.mean([r.replica_steps for r in res.results])) / 287, 1),
                  "r/p": round(float(real.mean() / planned[:, :279].mean()), 2), "secs": round(time.time() - t0, 1),
                  "acts": dict(Counter(acts).most_common())}))
