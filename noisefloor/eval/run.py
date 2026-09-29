"""Evaluation sets, baseline construction and one entry point that runs any agent on any level.

Held-out days: 32 per level (8 per family; configs/grid.json), drawn with EVAL_SEED, episode ids 100000+. Every agent
sees the same loads, incidents and distractor draws on the same day (common random numbers), so all
comparisons are paired by episode. OOD days (launch plateau, double spike) use ids 200000+.
HPA is tuned on training-seed days, never on evaluation days.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..agents.rules import HPA, Predictive, Runbook, Static, training_profile
from ..data.collect import DATA_SEED, world_from_meta
from ..govern.gate import Gate
from ..plan.oracle import OracleMPC
from ..sim.loads import LoadProfile, make_profile
from ..sim.telemetry import TelemetryWorld
from .harness import RunOutput, run_episodes

EVAL_SEED = 5150
EVAL_ID0 = 100_000
OOD_ID0 = 200_000
TUNE_ID0 = 300_000
R0 = 10


def eval_set(cfg: Dict, n: int = 16, ood: bool = False, seed: int = EVAL_SEED) -> Tuple[List[LoadProfile], List[int]]:
    fams = cfg["ood_families"] if ood else cfg["families"]
    id0 = OOD_ID0 if ood else EVAL_ID0
    ids = [id0 + i for i in range(n)]
    profiles = [make_profile(fams[i % len(fams)], np.random.default_rng([seed, eid]), cfg) for i, eid in enumerate(ids)]
    return profiles, ids


def tuning_set(cfg: Dict, n: int = 32) -> Tuple[List[LoadProfile], List[int]]:
    fams = cfg["families"]
    ids = [TUNE_ID0 + i for i in range(n)]
    return [make_profile(fams[i % len(fams)], np.random.default_rng([DATA_SEED, 1, eid]), cfg) for i, eid in enumerate(ids)], ids


def mean_J(out: RunOutput) -> float:
    return float(np.mean([r.J for r in out.results]))


def tune_hpa(cfg: Dict, gate: Gate, runbook: bool = False) -> Dict:
    """Grid target x window on training-seed days; returns the best setting and the whole grid."""
    profiles, ids = tuning_set(cfg)
    grid = []
    for target, window in itertools.product((0.5, 0.6, 0.7), (1, 3, 6)):
        agent = Runbook(target, window) if runbook else HPA(target, window)
        J = mean_J(run_episodes(agent, profiles, ids, np.full(len(ids), R0), cfg, gate))
        grid.append({"target": target, "window": window, "J": round(J, 2)})
    best = min(grid, key=lambda g: g["J"])
    return {"best": best, "grid": grid}


def static_replicas(cfg: Dict) -> int:
    profiles, _ = tuning_set(cfg)
    p99 = float(np.percentile(np.concatenate([p.load for p in profiles]), 99))
    return int(min(cfg["replicas_max"], np.ceil(p99 / (0.7 * cfg["mu"]))))


def baselines(cfg: Dict, gate: Gate, tuned: Optional[Dict] = None) -> Dict[str, object]:
    """Rule agents keyed by name. ``tuned`` holds {"hpa": {...}, "runbook": {...}} from tune_hpa."""
    tuned = tuned or {"hpa": tune_hpa(cfg, gate), "runbook": tune_hpa(cfg, gate, runbook=True)}
    h, rb = tuned["hpa"]["best"], tuned["runbook"]["best"]
    profiles, _ = tuning_set(cfg)
    return {
        "static": Static(static_replicas(cfg)),
        "hpa": HPA(h["target"], h["window"]),
        "runbook": Runbook(rb["target"], rb["window"]),
        "predictive": Predictive(training_profile(cfg, np.stack([p.load for p in profiles])), target=h["target"]),
    }


def run_agent(agent, cfg: Dict, gate: Gate, *, world: Optional[TelemetryWorld] = None, arm: str = "main",
              ood: bool = False, n: Optional[int] = None, record=None, keep_decisions: bool = False,
              approver_health: bool = False) -> RunOutput:
    """Run an agent on the evaluation days (all of them by default, per configs/grid.json)."""
    if n is None:
        from ..grid import eval_days
        n = eval_days(ood)
    profiles, ids = eval_set(cfg, n=n, ood=ood)
    if isinstance(agent, OracleMPC) or getattr(agent, "name", "") == "oracle":
        agent.profiles = profiles
    return run_episodes(agent, profiles, ids, np.full(len(ids), R0), cfg, gate, world=world, arm=arm,
                        record=record, keep_decisions=keep_decisions, approver_health=approver_health)


def oracle(cfg: Dict, seed: int = 1, ood: bool = False, n: int = 16) -> OracleMPC:
    profiles, _ = eval_set(cfg, n=n, ood=ood)
    return OracleMPC(profiles, cfg, seed=seed)


def load_world(lake: Path, level: str) -> TelemetryWorld:
    return world_from_meta(json.loads((lake / "meta" / f"{level}.json").read_text(encoding="utf-8")))


def result_rows(out: RunOutput, **tags) -> List[Dict]:
    return [{**tags, "episode_id": r.episode_id, "family": r.family, "J": r.J,
             "violation_min": 5 * r.violation_steps, "replica_hours": r.replica_steps * 5 / 60,
             "shed_frac": r.shed_mean, "failovers": r.failovers, "denials": r.denials, "approvals": r.approvals,
             "requests": r.requests, "refused_approvals": r.refused_approvals, "churn": r.churn,
             "plan_ms_p50": r.plan_ms_p50, "plan_ms_p99": r.plan_ms_p99}
            for r in out.results]
