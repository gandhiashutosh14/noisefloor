"""Pre-registration planning check on validation days (spec section 14: the harness must be able to
show a difference before the confirmatory grid runs).

Models are trained on the pilot data seed, not the grid's data seed, and evaluated on the 16
validation days used for the headroom check (ids 9000-9015), which never appear in the evaluation
set. Any planner change made after this check applies to every learned model and is disclosed in
PREREGISTRATION.md.

    python scripts/plan_check.py --variant jepa --d 0.8 [--tag check1]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import noisefloor  # noqa: E402,F401
from noisefloor.agents.wm_agent import WMAgent  # noqa: E402
from noisefloor.config import sim_config  # noqa: E402
from noisefloor.data.collect import build_world, collect, level_key, render  # noqa: E402
from noisefloor.data.lake import load_training_arrays, write_observations, write_trajectories  # noqa: E402
from noisefloor.eval.harness import run_episodes  # noqa: E402
from noisefloor.govern.gate import Gate  # noqa: E402
from noisefloor.models.train import TrainConfig, train  # noqa: E402
from noisefloor.sim.loads import make_profile  # noqa: E402

PILOT_SEED = 777001
VAL_IDS = list(range(9000, 9016))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="jepa")
    ap.add_argument("--d", type=float, default=0.8)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="check1")
    a = ap.parse_args()
    cfg = sim_config()
    key = level_key(a.arm, a.d)
    lake = ROOT / f"lake_check_{a.tag}_{a.variant}_{key}_s{a.seed}"
    shutil.rmtree(lake, ignore_errors=True)
    tr = collect(cfg, seed=PILOT_SEED)
    write_trajectories(lake / "trajectories", tr, PILOT_SEED)
    world = build_world(tr, cfg, a.arm, a.d)
    obs, xi = render(tr, cfg, world)
    write_observations(lake / "observations", key, tr.episode_id, obs, xi)
    out = train(load_training_arrays(lake, key), TrainConfig(variant=a.variant, seed=a.seed))
    agent = WMAgent(out["model"], cfg, seed=a.seed)
    fams = cfg["families"]
    profiles = [make_profile(fams[i % 4], np.random.default_rng(i), cfg) for i in VAL_IDS]
    planned = np.zeros((16, cfg["steps_per_episode"] - 1))

    def record(t, i, dec):
        planned[i, t] = agent.last_pred_cost[i]

    t0 = time.time()
    res = run_episodes(agent, profiles, VAL_IDS, np.full(16, 10), cfg, Gate.load("gate_v1.json"), world=world,
                       arm=a.arm, record=record)
    secs = time.time() - t0
    J = np.array([r.J for r in res.results])
    w = 0.97 ** np.arange(8)
    real = np.stack([(res.costs[:, t:t + 8] * w).sum(1) for t in range(res.costs.shape[1] - 8)], 1)
    ratio = float(real.mean() / planned[:, :real.shape[1]].mean())
    by = {f: round(float(np.mean([r.J for r in res.results if r.family == f])), 1) for f in fams}
    row = {"variant": a.variant, "level": key, "seed": a.seed, "J": round(float(J.mean()), 1), "by_family": by,
           "violation_steps": float(np.mean([r.violation_steps for r in res.results])),
           "failovers": int(sum(r.failovers for r in res.results)), "requests": int(sum(r.requests for r in res.results)),
           "realised_over_planned": round(ratio, 2), "eval_seconds": round(secs, 1),
           "train_seconds": round(out["train_seconds"], 1), "plan_ms_p50": res.results[0].plan_ms_p50}
    print(json.dumps(row), flush=True)
    d = ROOT / "reports" / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"plan-{a.tag}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    shutil.rmtree(lake, ignore_errors=True)


if __name__ == "__main__":
    main()
