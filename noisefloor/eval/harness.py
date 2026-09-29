"""The control loop every agent runs in, learned or not.

At each 5-minute step:
  1. The approver answers last step's failover requests; an approved failover executes now if the
     gate still allows it (the action budget may have been spent meanwhile; such a refusal is recorded).
  2. Otherwise the agent proposes up to three candidate actions, best first.
  3. Each candidate goes through the gate: allow -> execute; needs-approval -> file the request and
     try the next candidate; deny -> try the next. If nothing is allowed, noop. Every candidate's
     verdict is kept in the decision record, not just the executed one's.
  4. The fleet steps; cost, violations and the decision record are kept.

Observations are rendered from the true state with the episode's own distractor stream, so every
agent sees the same noise on the same day (common random numbers). The gate's limits drive both the
mask the agents plan with and the fleet's action budget window, so planner, harness and gate agree.

Metrics per episode: J (total cost), SLO-violation steps, replica-steps (ready + pending, as billed),
mean shed level (as billed), failovers executed, denials (candidates the gate refused), approvals
granted and requests filed, refused approvals (granted but no longer allowed), churn (a scale in the
opposite direction within CHURN_WINDOW steps of the previous scale), and planning time. ``plan_ms``
is the wall-clock of one batched planning call divided by the number of environments, so its p50/p99
are over the 287 calls of a run, not over individual decisions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..agents.rules import Context
from ..govern.approver import Approver
from ..govern.gate import Gate
from ..sim.fleet import ACTIONS, DOWN2, FAILOVER, NOOP, UP2, UP6, Fleet, allowed_mask
from ..sim.loads import LoadProfile
from ..sim.telemetry import DistractorStream, TelemetryWorld, features

CHURN_WINDOW = 3


@dataclass
class EpisodeResult:
    episode_id: int
    family: str
    J: float
    violation_steps: int
    replica_steps: float
    shed_mean: float
    failovers: int
    denials: int
    approvals: int
    requests: int
    churn: int
    plan_ms_p50: float
    plan_ms_p99: float
    refused_approvals: int = 0


@dataclass
class RunOutput:
    results: List[EpisodeResult]
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    costs: Optional[np.ndarray] = None       # (E, T-1)


def render_obs(fleet: Fleet, world: Optional[TelemetryWorld], streams: Optional[List[DistractorStream]]) -> Optional[np.ndarray]:
    if world is None:
        return None
    phi = features(fleet.true_state(), fleet.last["rho"], np.full(fleet.E, fleet.t), fleet.cfg)
    xi = np.stack([s.next() for s in streams])
    eps = np.stack([s.noise() for s in streams])
    return world.render(phi, xi, eps)


def run_episodes(agent, profiles: List[LoadProfile], episode_ids: List[int], r0: np.ndarray, cfg: Dict, gate: Gate, *,
                 world: Optional[TelemetryWorld] = None, arm: str = "main", record: Optional[Callable] = None,
                 keep_decisions: bool = False, approver_health: bool = False) -> RunOutput:
    """Run one agent on a batch of episodes in lockstep. ``record(t, i, decision)`` is called per decision.
    ``approver_health`` selects the approver that also requires degraded region health (post-hoc variant)."""
    E = len(profiles)
    limits = gate.limits(cfg)
    fleet = Fleet.start(profiles, r0, cfg, budget_window=gate.window)
    if hasattr(agent, "fleet"):
        agent.fleet = fleet
    agent.reset(E, cfg)
    approver = Approver(E, require_degraded_health=approver_health)
    streams = [DistractorStream(world.world_seed, eid, arm, n=world.k) for eid in episode_ids] if world is not None else None
    T = cfg["steps_per_episode"]
    J = np.zeros(E)
    viol = np.zeros(E, dtype=np.int64)
    rep = np.zeros(E)
    shed = np.zeros(E)
    fails = np.zeros(E, dtype=np.int64)
    denials = np.zeros(E, dtype=np.int64)
    refused = np.zeros(E, dtype=np.int64)
    churn = np.zeros(E, dtype=np.int64)
    last_dir = np.zeros(E, dtype=np.int64)          # +1 after a scale-up, -1 after a scale-down
    last_dir_t = np.full(E, -10 ** 6)
    plan_ms: List[float] = []
    costs = np.zeros((E, T - 1))
    decisions: List[Dict[str, Any]] = []
    while not fleet.done:
        t = fleet.t
        obs = render_obs(fleet, world, streams)
        golden = fleet.golden()
        approver.observe(golden["error_rate"], golden["health"])
        approved = approver.decide()
        mask = allowed_mask(fleet.act, cfg, limits)
        ctx = Context(t=t, obs=obs, golden=golden, actuator=fleet.act, act_vec=fleet.act.vector(cfg), mask=mask,
                      true_state=fleet.true_state(), limits=limits, pending=approver.pending.copy())
        t0 = time.perf_counter()
        ranked = agent.propose(ctx)
        plan_ms.append((time.perf_counter() - t0) * 1000.0 / E)
        chosen = np.full(E, NOOP, dtype=np.int64)
        for i in range(E):
            record_i: Dict[str, Any] = {"proposed": [ACTIONS[a] for a in ranked[i]], "executed": "noop", "verdict": "allow",
                                        "reason": "", "effect": "reversible", "args": {}, "approval": None,
                                        "requested": False, "candidates": []}
            if i in approved:
                if approved[i]:
                    v = gate.check("failover", fleet.act, i, cfg, approved=True)
                    if v.verdict == "allow":
                        chosen[i] = FAILOVER
                        record_i.update(executed="failover", verdict="allow", effect=v.effect, args=v.args, approval="granted")
                        record_i["candidates"].append({"action": "failover", "verdict": "allow", "reason": "approved"})
                        _finish(record_i, t, i, decisions, keep_decisions, record)
                        continue
                    refused[i] += 1
                    record_i["approval"] = "granted-refused"
                    record_i["candidates"].append({"action": "failover", "verdict": v.verdict, "reason": v.reason})
                else:
                    record_i["approval"] = "refused"
            for a in ranked[i]:
                name = ACTIONS[int(a)]
                v = gate.check(name, fleet.act, i, cfg)
                record_i["candidates"].append({"action": name, "verdict": v.verdict, "reason": v.reason})
                if v.verdict == "allow":
                    chosen[i] = int(a)
                    record_i.update(executed=name, verdict="allow", effect=v.effect, args=v.args, reason="")
                    break
                if v.verdict == "needs-approval":
                    approver.request(i)
                    record_i["requested"] = True
                    continue
                denials[i] += 1
            if record_i["executed"] == "noop" and record_i["candidates"]:
                last = record_i["candidates"][-1]
                if last["verdict"] != "allow":
                    record_i.update(verdict=last["verdict"], reason=last["reason"])
            _finish(record_i, t, i, decisions, keep_decisions, record)
        direction = np.where(np.isin(chosen, (UP2, UP6)), 1, np.where(chosen == DOWN2, -1, 0))
        reversal = (direction != 0) & (direction == -last_dir) & (t - last_dir_t <= CHURN_WINDOW)
        churn += reversal
        last_dir = np.where(direction != 0, direction, last_dir)
        last_dir_t = np.where(direction != 0, t, last_dir_t)
        out = fleet.step(chosen)
        J += out["cost"]
        costs[:, t] = out["cost"]
        viol += out["violation"].astype(np.int64)
        rep += fleet.act.r + fleet.act.pending
        shed += out["shed_level"]
        fails += out["failover"].astype(np.int64)
    steps = T - 1
    p50, p99 = float(np.percentile(plan_ms, 50)), float(np.percentile(plan_ms, 99))
    results = [EpisodeResult(int(episode_ids[i]), profiles[i].family, float(J[i]), int(viol[i]), float(rep[i]),
                             float(shed[i] / steps), int(fails[i]), int(denials[i]), int(approver.approvals[i]),
                             int(approver.requests[i]), int(churn[i]), p50, p99, int(refused[i])) for i in range(E)]
    return RunOutput(results, decisions, costs)


def _finish(record_i: Dict[str, Any], t: int, i: int, decisions: List[Dict[str, Any]], keep: bool, record: Optional[Callable]) -> None:
    if keep:
        decisions.append({"t": t, "i": i, **record_i})
    if record:
        record(t, i, record_i)
