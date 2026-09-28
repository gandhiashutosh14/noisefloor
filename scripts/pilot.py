"""Hour-2 pilot (spec section 14): probe-only comparison of JEPA-SIGReg and the reconstruction model
at d = 0.95 on a separate data seed, before any planning metric exists. It also times 1,200 training
steps for each model. Output: reports/pilot.json and reports/pilot.md.

    python scripts/pilot.py [--k 16]      (k = number of distractor sources; default from configs/sim.json)

The first pilot ran with k = 128 (reports/pilot-k128.*) and showed no JEPA advantage; the spec's one
permitted pre-registration change is the distractor dimensionality, re-tested here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import noisefloor  # noqa: E402,F401  (CPU guard)
from noisefloor.config import sim_config  # noqa: E402
from noisefloor.data.collect import build_world, collect, level_key, render  # noqa: E402
from noisefloor.data.lake import load_probe_arrays, load_training_arrays, write_observations, write_trajectories  # noqa: E402
from noisefloor.eval.probes import kstep_error, probe  # noqa: E402
from noisefloor.models.train import TrainConfig, train  # noqa: E402

PILOT_SEED = 777001
LEVEL = ("main", 0.95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tag", default="pilot", help="output name: reports/pilot/<tag>-k<k>.{json,md}")
    args = ap.parse_args()
    cfg = sim_config()
    if args.k is not None:
        cfg["distractor_dims"] = args.k
    k = cfg.get("distractor_dims", 128)
    lake = ROOT / f"lake_pilot_{args.tag}_k{k}"
    shutil.rmtree(lake, ignore_errors=True)
    print(f"pilot at {LEVEL} with k = {k}", flush=True)
    tr = collect(cfg, seed=PILOT_SEED)
    write_trajectories(lake / "trajectories", tr, PILOT_SEED)
    world = build_world(tr, cfg, *LEVEL)
    obs, xi = render(tr, cfg, world)
    key = level_key(*LEVEL)
    write_observations(lake / "observations", key, tr.episode_id, obs, xi)
    arrays = load_training_arrays(lake, key)
    val = load_training_arrays(lake, key, splits=("val",))
    fit, test = load_probe_arrays(lake, key, splits=("train",)), load_probe_arrays(lake, key, splits=("val",))
    rows = {}
    for variant in ("jepa", "recon"):
        out = train(arrays, TrainConfig(variant=variant, seed=0))
        rows[variant] = {"train_seconds": round(out["train_seconds"], 1), "params": out["params"],
                         **{k: round(v, 4) for k, v in probe(out["model"], fit, test).items()},
                         **{k: round(v, 4) for k, v in kstep_error(out["model"], val).items()}}
        print(variant, rows[variant], flush=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "uncommitted"
    report = {"level": key, "distractor_dims": k, "data_seed": PILOT_SEED, "steps": 1200, "commit": sha, "results": rows,
              "generated": time.strftime("%Y-%m-%d")}
    out_dir = ROOT / "reports" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}-k{k}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Pilot: probes at {key}, k = {k} distractor sources (data seed {PILOT_SEED}, 1,200 steps, commit {sha})", "",
             "Probe-only comparison run before pre-registration; no planning metric was computed.", "",
             "| model | state R^2 | distractor R^2 | erank | 1-step err | 4-step err | train s |", "|---|---|---|---|---|---|---|"]
    for v, r in rows.items():
        lines.append(f"| {v} | {r['probe_state_r2']:.3f} | {r['probe_xi_r2']:.3f} | {r['erank']:.1f} | "
                     f"{r['kstep_err_1']:.3f} | {r['kstep_err_4']:.3f} | {r['train_seconds']} |")
    (out_dir / f"{args.tag}-k{k}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(lake, ignore_errors=True)


if __name__ == "__main__":
    main()
