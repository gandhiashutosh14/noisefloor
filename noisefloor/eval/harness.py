"""The control loop every agent runs in, learned or not.

At each 5-minute step:
  1. The approver answers last step's failover requests; an approved failover executes now.
  2. Otherwise the agent proposes up to three candidate actions, best first.
  3. Each candidate goes through the gate: allow -> execute; needs-approval -> file the request and
     try the next candidate; deny -> try the next. If nothing is allowed, noop.
  4. The fleet steps; cost, violations and the decision record are kept.

Observations are rendered from the true state with the episode's own distractor stream, so every
agent sees the same noise on the same day (common random numbers).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..agents.rules import Context
from ..govern.approver import Approver
from ..govern.gate import Gate
from ..sim.fleet import ACTIONS, FAILOVER, NOOP, Fleet, allowed_mask
from ..sim.loads import LoadProfile
from ..sim.telemetry import DistractorStream, TelemetryWorld, features


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
                 keep_decisions: bool = False) -> RunOutput:
    """Run one agent on a batch of episodes in lockstep. ``record(t, i, decision)`` is called per decision."""
    E = len(profiles)
    fleet = Fleet.start(profiles, r0, cfg)
    if hasattr(agent, "fleet"):
        agent.fleet = fleet
    agent.reset(E, cfg)
    approver = Approver(E)
    streams = [DistractorStream(world.world_seed, eid, arm, n=world.k) for eid in episode_ids] if world is not None else None
    T = cfg["steps_per_episode"]
    J = np.zeros(E)
    viol = np.zeros(E, dtype=np.int64)
    rep = np.zeros(E)
    shed = np.zeros(E)
    fails = np.zeros(E, dtype=np.int64)
    denials = np.zeros(E, dtype=np.int64)
    churn = np.zeros(E, dtype=np.int64)
    plan_ms: List[float] = []
    costs = np.zeros((E, T - 1))
    decisions: List[Dict[str, Any]] = []
    prev_action = np.zeros(E, dtype=np.int64)
    while not fleet.done:
        t = fleet.t
        obs = render_obs(fleet, world, streams)
        golden = fleet.golden()
        approver.observe(golden["error_rate"])
        approved = approver.decide()
        mask = allowed_mask(fleet.act, cfg)
        ctx = Context(t=t, obs=obs, golden=golden, actuator=fleet.act, act_vec=fleet.act.vector(cfg), mask=mask,
                      true_state=fleet.true_state())
        t0 = time.perf_counter()
        ranked = agent.propose(ctx)
        plan_ms.append((time.perf_counter() - t0) * 1000.0 / E)
        chosen = np.full(E, NOOP, dtype=np.int64)
        for i in range(E):
            record_i = {"proposed": [ACTIONS[a] for a in ranked[i]], "executed": "noop", "verdict": "allow",
                        "reason": "", "effect": "reversible", "args": {}, "approval": None}
            if approved.get(i):
                v = gate.check("failover", fleet.act, i, cfg, approved=True)
                record_i["approval"] = "granted"
                if v.verdict == "allow":
                    chosen[i] = FAILOVER
                    record_i.update(executed="failover", verdict="allow", effect=v.effect, args=v.args)
                    decisions.append({"t": t, "i": i, **record_i}) if keep_decisions else None
                    if record:
                        record(t, i, record_i)
                    continue
            elif i in approved:
                record_i["approval"] = "refused"
            for a in ranked[i]:
                name = ACTIONS[int(a)]
                v = gate.check(name, fleet.act, i, cfg)
                if v.verdict == "allow":
                    chosen[i] = int(a)
                    record_i.update(executed=name, verdict="allow", effect=v.effect, args=v.args, reason="")
                    break
                if v.verdict == "needs-approval":
                    approver.request(i)
                    record_i.update(verdict="needs-approval", reason=v.reason, effect=v.effect, args=v.args)
                    continue
                denials[i] += 1
                record_i.update(verdict="deny", reason=v.reason, effect=v.effect, args=v.args)
            if chosen[i] == NOOP and record_i["executed"] == "noop" and record_i["verdict"] != "allow":
                record_i["executed"] = "noop"
            if keep_decisions:
                decisions.append({"t": t, "i": i, **record_i})
            if record:
                record(t, i, record_i)
        churn += (chosen != NOOP) & (chosen != prev_action) & (prev_action != NOOP)
        prev_action = chosen
        out = fleet.step(chosen)
        J += out["cost"]
        costs[:, t] = out["cost"]
        viol += out["violation"].astype(np.int64)
        rep += fleet.act.r + fleet.act.pending
        shed += fleet.act.shed
        fails += out["failover"].astype(np.int64)
    steps = T - 1
    p50, p99 = float(np.percentile(plan_ms, 50)), float(np.percentile(plan_ms, 99))
    results = [EpisodeResult(int(episode_ids[i]), profiles[i].family, float(J[i]), int(viol[i]), float(rep[i]),
                             float(shed[i] / steps), int(fails[i]), int(denials[i]), int(approver.approvals[i]),
                             int(approver.requests[i]), int(churn[i]), p50, p99) for i in range(E)]
    return RunOutput(results, decisions, costs)
