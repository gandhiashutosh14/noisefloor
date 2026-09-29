"""The experiment grid: data, training cells, evaluation, baselines, and sharding across processes.

A cell is one (variant, arm, d, seed, z_dim) training run followed by its evaluations. Results go to
Delta tables in the shard's own lake (``results``, ``eval_steps``, ``train_runs``); the aggregate
step reads every shard's tables and appends them into one lake, never copying partition folders.
The data is regenerated deterministically in every shard (seconds on CPU), and each shard records
the content hash of its trajectories so the aggregate step can check that all shards saw identical
data.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import tracking
from .agents.wm_agent import WMAgent
from .audit.envelopes import EnvelopeRecorder, git_sha
from .config import config_hash, load_json
from .data.collect import DATA_SEED, LEVELS, build_world, collect, level_key, render, save_meta
from .data.lake import append_rows, load_probe_arrays, load_training_arrays, trajectory_hash, write_observations, write_trajectories
from .eval.probes import kstep_error, probe
from .eval.run import baselines, eval_set, load_world, oracle, result_rows, run_agent, tune_hpa
from .govern.gate import Gate
from .models.train import K, TrainConfig, train
from .models.train import load as load_model
from .models.train import save as save_model
from .tracking import DecisionTracer


@dataclass(frozen=True)
class Cell:
    variant: str
    arm: str
    d: float
    seed: int
    z_dim: int = 16
    evals: tuple = ("main",)          # "main", "ood" (unseen load families), "h4" (planning horizon 4)

    @property
    def level(self) -> str:
        return level_key(self.arm, self.d)

    @property
    def name(self) -> str:
        z = "" if self.z_dim == 16 else f"-z{self.z_dim}"
        return f"{self.variant}{z}-{self.level}-s{self.seed}"

    def minutes(self) -> float:
        train = {"jepa": 2.4, "lambda0": 1.3, "ema": 1.2, "recon": 1.4, "costonly": 1.1}[self.variant]   # reports/benchmark.json
        return train * (1.6 if self.z_dim > 16 else 1.0) + 0.3 * len(self.evals)


def eval_days(ood: bool = False) -> int:
    days = load_json("grid.json").get("eval_days", {"main": 16, "ood": 16})
    return int(days["ood" if ood else "main"])


def grid_cells(tier: int, spec: Optional[Dict] = None) -> List[Cell]:
    spec = spec or load_json("grid.json")
    cells: List[Cell] = []
    for block in spec["blocks"]:
        if block["tier"] > tier:
            continue
        for variant in block["variants"]:
            for arm, d in block["levels"]:
                for seed in range(block["seeds"]):
                    cells.append(Cell(variant, arm, float(d), seed, block.get("z_dim", 16), tuple(block.get("evals", ["main"]))))
    seen, out = set(), []
    for c in cells:                                   # later blocks may add evals to an existing cell
        key = (c.variant, c.arm, c.d, c.seed, c.z_dim)
        if key in seen:
            i = next(j for j, o in enumerate(out) if (o.variant, o.arm, o.d, o.seed, o.z_dim) == key)
            out[i] = Cell(*key, evals=tuple(dict.fromkeys(out[i].evals + c.evals)))
        else:
            seen.add(key)
            out.append(c)
    return out


def shard(cells: List[Cell], index: int, count: int) -> List[Cell]:
    """Greedy longest-first assignment to ``count`` shards; returns shard ``index``."""
    loads = [0.0] * count
    assign: List[List[Cell]] = [[] for _ in range(count)]
    for c in sorted(cells, key=lambda c: (-c.minutes(), c.name)):
        j = int(np.argmin(loads))
        loads[j] += c.minutes()
        assign[j].append(c)
    return sorted(assign[index], key=lambda c: c.name)


# ---------------------------------------------------------------------------------------------- data
def build_lake(lake: Path, cfg: Dict, levels: Iterable = LEVELS) -> Dict:
    t0 = time.perf_counter()
    tr = collect(cfg, seed=DATA_SEED)
    data_hash = trajectory_hash(tr)
    version = write_trajectories(lake / "trajectories", tr, DATA_SEED, data_hash)
    info = {"data_seed": DATA_SEED, "trajectories_version": version, "data_hash": data_hash,
            "config_hash": config_hash(cfg), "levels": {}}
    for arm, d in levels:
        world = build_world(tr, cfg, arm, d)
        obs, xi = render(tr, cfg, world)
        key = level_key(arm, d)
        save_meta(lake / "meta" / f"{key}.json", world)
        info["levels"][key] = write_observations(lake / "observations", key, tr.episode_id, obs, xi, data_hash)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    (lake / "meta" / "data.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------------------------------------- cells
def train_cell(cell: Cell, lake: Path, runs: Path, steps: int = 1200, log=print) -> Dict:
    info = json.loads((lake / "meta" / "data.json").read_text(encoding="utf-8"))
    arrays = load_training_arrays(lake, cell.level)
    tc = TrainConfig(variant=cell.variant, z_dim=cell.z_dim, seed=cell.seed, steps=steps)
    params = {**asdict(tc), "level": cell.level, "git_sha": git_sha(), "data_hash": info["data_hash"],
              "delta_trajectories_version": arrays["traj_version"], "delta_observations_version": arrays["obs_version"],
              "sim_config_hash": info["config_hash"], "K": K, "M": tc.slices}
    with tracking.run("noisefloor-train", cell.name, params, tags={"variant": cell.variant, "level": cell.level}) as rid:
        out = train(arrays, tc, logger=lambda s, m: tracking.log_metrics(m, step=s))
        model = out["model"]
        fit = load_probe_arrays(lake, cell.level, splits=("train",))
        test = load_probe_arrays(lake, cell.level, splits=("val",))
        val = load_training_arrays(lake, cell.level, splits=("val",))
        diag = {**probe(model, fit, test), **kstep_error(model, val)}
        tracking.log_metrics(diag)
        tracking.log_dict(params, "config.json")
        path = runs / cell.name
        save_model(model, tc, path, extra={"cell": asdict(cell), "diagnostics": diag, "mlflow_run_id": rid,
                                          "train_seconds": out["train_seconds"], "params": out["params"],
                                          "data_hash": info["data_hash"]})
        tracking.log_artifact(path / "model.pt")
        try:
            tracking.log_encoder(model)
        except Exception as e:  # logging the model flavour is best-effort
            log(f"  (encoder model not logged: {e})")
    row = {"run": cell.name, "variant": cell.variant, "arm": cell.arm, "d": cell.d, "seed": cell.seed,
           "z_dim": cell.z_dim, "train_seconds": out["train_seconds"], "params": out["params"],
           "mlflow_run_id": rid or "", "data_hash": info["data_hash"], "git_sha": params["git_sha"],
           **{k: float(v) for k, v in diag.items()}}
    append_rows(lake / "train_runs", [row])
    log(f"  trained {cell.name} in {out['train_seconds']:.0f}s: state R2 {diag['probe_state_r2']:.3f}, "
        f"xi R2 {diag['probe_xi_r2']:.3f}, erank {diag['erank']:.1f}")
    return row


def _realised_vs_planned(costs: np.ndarray, planned: np.ndarray, H: int = 8, disc: float = 0.97) -> float:
    """Mean realised discounted H-step cost divided by the mean planned cost, over decisions with a full horizon."""
    E, T = costs.shape
    w = disc ** np.arange(H)
    real = np.stack([(costs[:, t:t + H] * w).sum(1) for t in range(T - H)], axis=1)
    return float(real.mean() / max(planned[:, :T - H].mean(), 1e-9))


def eval_cell(cell: Cell, lake: Path, runs: Path, cfg: Dict, gate_v1: Gate, log=print, n_eval: Optional[int] = None) -> List[Dict]:
    model = load_model(runs / cell.name)
    meta = json.loads((runs / cell.name / "config.json").read_text(encoding="utf-8"))
    world = load_world(lake, cell.level)
    info = json.loads((lake / "meta" / "data.json").read_text(encoding="utf-8"))
    rows = []
    for ev in cell.evals:
        agent = WMAgent(model, cfg, seed=cell.seed, horizon=4 if ev == "h4" else 8)
        gate = gate_v1
        n = n_eval or eval_days(ev == "ood")
        profiles, ids = eval_set(cfg, n=n, ood=(ev == "ood"))
        recorder = EnvelopeRecorder(model.variant, ev, cell.seed, cell.arm, cell.d, ids, agent=agent, z_dim=cell.z_dim,
                                    policy_id=gate.policy_id,
                                    extra={"mlflow_run_id": meta.get("mlflow_run_id") or "",
                                           "delta_version": info["levels"][cell.level]})
        planned = np.zeros((len(ids), cfg["steps_per_episode"] - 1))
        label = f"{cell.name}-{ev}"
        tracer = DecisionTracer(ids[::8], label)           # MLflow spans for 1 in 8 evaluation days

        def record(t, i, dec):
            recorder(t, i, dec)
            if agent.last_pred_cost is not None:
                planned[i, t] = agent.last_pred_cost[i]
                tracer.record(t, i, ids[i], dec, float(agent.last_pred_cost[i]), float(agent.last_spread[i]))

        with tracking.run("noisefloor-eval", label, {"cell": cell.name, "eval": ev, "train_run_id": meta.get("mlflow_run_id") or ""}):
            t0 = time.perf_counter()
            out = run_agent(agent, cfg, gate, world=world, arm=cell.arm, ood=(ev == "ood"), n=n, record=record)
            secs = time.perf_counter() - t0
            tracer.flush()
            ratio = _realised_vs_planned(out.costs, planned)
            res = result_rows(out, run=label, model=model.variant, variant=ev, train_seed=cell.seed, arm=cell.arm,
                              d=cell.d, z_dim=cell.z_dim, split="ood" if ev == "ood" else "test", git_sha=git_sha(),
                              data_hash=info["data_hash"], realised_over_planned=ratio)
            tracking.log_metrics({"J_mean": float(np.mean([r["J"] for r in res])), "realised_over_planned": ratio,
                                  "eval_seconds": secs})
        rows.extend(res)
        steps = [{"run": label, "episode_id": int(d["run_id"].rsplit("-e", 1)[1]), "t": d["seq"] - 1,
                  "proposed": ",".join(d["data"]["proposed"] or []), "executed": d["data"]["executed"],
                  "verdict": d["data"]["verdict"], "reason": d["data"]["reason"],
                  "predicted_cost": float(d["data"].get("predicted_cost", float("nan"))),
                  "realized_cost": float(out.costs[ids.index(int(d["run_id"].rsplit("-e", 1)[1])), d["seq"] - 1])}
                 for d in recorder.rows]
        append_rows(lake / "eval_steps", steps)
        env_dir = runs / cell.name / "envelopes"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / f"{ev}.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in recorder.rows[:2000]) + "\n",
                                            encoding="utf-8")
        log(f"  eval {label}: J {np.mean([r['J'] for r in res]):.1f}, realised/planned {ratio:.2f}, {secs:.0f}s")
    append_rows(lake / "results", rows, partition_by=["arm"])
    return rows


def run_baselines(lake: Path, cfg: Dict, gate_v1: Gate, log=print, n_eval: Optional[int] = None) -> List[Dict]:
    """Rule agents and the oracle read clean channels, so their results do not depend on d."""
    tuned = {"hpa": tune_hpa(cfg, gate_v1), "runbook": tune_hpa(cfg, gate_v1, runbook=True)}
    (lake / "meta").mkdir(parents=True, exist_ok=True)
    (lake / "meta" / "hpa_tuning.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    agents = baselines(cfg, gate_v1, tuned)
    rows = []
    info = json.loads((lake / "meta" / "data.json").read_text(encoding="utf-8"))
    for ood in (False, True):
        n = n_eval or eval_days(ood)
        for name, agent in list(agents.items()) + [("oracle", oracle(cfg, ood=ood, n=n))]:
            t0 = time.perf_counter()
            out = run_agent(agent, cfg, gate_v1, ood=ood, n=n)
            res = result_rows(out, run=f"{name}{'-ood' if ood else ''}", model=name, variant="ood" if ood else "main",
                              train_seed=0, arm="clean", d=-1.0, z_dim=0, split="ood" if ood else "test",
                              git_sha=git_sha(), data_hash=info["data_hash"], realised_over_planned=float("nan"))
            rows.extend(res)
            log(f"  {name}{' (OOD)' if ood else ''}: J {np.mean([r['J'] for r in res]):.1f} ({time.perf_counter() - t0:.0f}s)")
    append_rows(lake / "results", rows, partition_by=["arm"])
    return rows
