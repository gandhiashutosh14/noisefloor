"""Timing benchmark (spec section 14, hour 2): 100 training steps per model variant and 100 planning
decisions over 16 environments, on this machine's CPU with 2 threads. Output: reports/benchmark.json.

    python scripts/benchmark.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import noisefloor  # noqa: E402,F401
from noisefloor.agents.rules import Context  # noqa: E402
from noisefloor.agents.wm_agent import WMAgent  # noqa: E402
from noisefloor.config import sim_config  # noqa: E402
from noisefloor.data.collect import build_world, collect, level_key, render  # noqa: E402
from noisefloor.data.lake import load_training_arrays, write_observations, write_trajectories  # noqa: E402
from noisefloor.models.train import TrainConfig, train  # noqa: E402
from noisefloor.sim.fleet import allowed_mask, new_actuator  # noqa: E402


def main() -> None:
    cfg = sim_config()
    lake = ROOT / ".checks" / "bench_lake"
    shutil.rmtree(lake, ignore_errors=True)
    tr = collect(cfg, seed=4242)
    write_trajectories(lake / "trajectories", tr, 4242)
    world = build_world(tr, cfg, "main", 0.8)
    obs, xi = render(tr, cfg, world)
    write_observations(lake / "observations", level_key("main", 0.8), tr.episode_id, obs, xi)
    arrays = load_training_arrays(lake, "main_d80")
    out = {"cpu": platform.processor() or platform.machine(), "threads": 2, "torch": torch.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "train_100_steps_s": {}}
    model = None
    for variant in ("jepa", "recon", "ema", "costonly"):
        r = train(arrays, TrainConfig(variant=variant, steps=100, log_every=1000))
        out["train_100_steps_s"][variant] = round(r["train_seconds"], 2)
        if variant == "jepa":
            model = r["model"]
    agent = WMAgent(model, cfg, seed=0)
    agent.reset(16, cfg)
    act = new_actuator(np.full(16, 10), cfg)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for t in range(100):
        ctx = Context(t=t, obs=rng.standard_normal((16, 128)).astype(np.float32), golden={}, actuator=act,
                      act_vec=act.vector(cfg), mask=allowed_mask(act, cfg))
        agent.propose(ctx)
    dt = time.perf_counter() - t0
    out["decisions_100x16_s"] = round(dt, 2)
    out["ms_per_env_decision"] = round(1000 * dt / 1600, 2)
    out["estimated_1200_step_training_min"] = {k: round(12 * v / 60, 1) for k, v in out["train_100_steps_s"].items()}
    out["estimated_eval_16_days_min"] = round(dt / 100 * 287 / 60, 2)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    shutil.rmtree(lake, ignore_errors=True)


if __name__ == "__main__":
    main()
