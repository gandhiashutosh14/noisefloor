"""The simulated service fleet: vectorised over E environments, 5-minute steps.

Physics per step (all arrays shape (E,)):
    capacity   kappa = r * mu * (0.7 + 0.3 c) * h            ready replicas only
    shed load  lam_s = lam * (1 - shed);  rho = lam_s / kappa
    queue      q'    = max(0, q + 300 (lam_s - kappa))       requests waiting
    latency    l95   = S_eff + 3 W_q + q' / kappa            (Sakasegawa L_q, Little's law, x3 for p95)
               W_q   = rho~^sqrt(2(r+1)) / ((1 - rho~) lam_s),  rho~ = min(rho, 0.98)
    errors     e     = 0.05 (1 - h) + 0.2 sigmoid(40 (rho - 1.05))
    cost       c_t   = 0.05 (r + pending) + 1.0 [l95 > 250 ms or e > 1%] + 2.0 shed + 0.1 [warm] + 20 [failover]

The actuator state (replicas ready and pending, shed, cooldowns, lockout, action budget) is known
exactly to every controller; `allowed_mask` and `apply_actions` are shared by the environment, the
governance gate and the planner's rollouts so all three agree on what an action does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .loads import LoadProfile

ACTIONS = ["noop", "scale_up_2", "scale_up_6", "scale_down_2", "cache_warm", "shed_10", "failover"]
NOOP, UP2, UP6, DOWN2, WARM, SHED, FAILOVER = range(7)
N_ACTIONS = len(ACTIONS)
STATE_NAMES = ["load", "queue", "replicas", "pending", "cache", "health", "p95_ms", "error_rate"]


@dataclass
class Actuator:
    """Controller-visible state that changes only through actions and time."""
    r: np.ndarray                 # ready replicas
    p1: np.ndarray                # pending, ready next step
    p2: np.ndarray                # pending, ready in two steps
    shed: np.ndarray              # current shed fraction
    shed_expiry: np.ndarray       # (E, restore_steps): shed amount that expires k+1 steps from now
    since_warm: np.ndarray        # steps since the last cache warm
    lockout: np.ndarray           # failover lockout steps remaining
    failovers: np.ndarray         # failovers used today
    recent: np.ndarray            # (E, window): 1 where a non-noop action was taken

    @property
    def pending(self) -> np.ndarray:
        return self.p1 + self.p2

    @property
    def budget_used(self) -> np.ndarray:
        return self.recent.sum(axis=1)

    def copy(self) -> "Actuator":
        return Actuator(*(getattr(self, f).copy() for f in self.__dataclass_fields__))

    def repeat(self, n: int) -> "Actuator":
        """Each environment repeated n times along axis 0 (for batched planning rollouts)."""
        return Actuator(*(np.repeat(getattr(self, f), n, axis=0) for f in self.__dataclass_fields__))

    def vector(self, cfg: Dict) -> np.ndarray:
        """(E, 6) normalised actuator features for the models: r, pending, shed, cooldown, lockout, budget."""
        return np.stack([
            self.r / cfg["replicas_max"],
            self.pending / cfg["replicas_max"],
            self.shed / cfg["shed_max"],
            np.minimum(self.since_warm, cfg["cache_warm_every"] * 2) / (cfg["cache_warm_every"] * 2),
            self.lockout / cfg["failover_lockout_steps"],
            self.budget_used / cfg["budget_actions"],
        ], axis=1).astype(np.float32)


def new_actuator(r0: np.ndarray, cfg: Dict) -> Actuator:
    E = len(r0)
    return Actuator(
        r=np.asarray(r0, dtype=np.int64).copy(), p1=np.zeros(E, np.int64), p2=np.zeros(E, np.int64),
        shed=np.zeros(E), shed_expiry=np.zeros((E, cfg["shed_restore_steps"])),
        since_warm=np.full(E, 10 * cfg["cache_warm_every"], np.int64), lockout=np.zeros(E, np.int64),
        failovers=np.zeros(E, np.int64), recent=np.zeros((E, cfg["budget_window"]), np.int64))


def allowed_mask(act: Actuator, cfg: Dict) -> np.ndarray:
    """(E, 7) bool: which actions the gate's constraints and the action budget permit right now.
    Approval for failover is separate (the gate returns needs-approval, not allow)."""
    E = len(act.r)
    total = act.r + act.pending
    m = np.ones((E, N_ACTIONS), dtype=bool)
    m[:, UP2] = total + 2 <= cfg["replicas_max"]
    m[:, UP6] = total + 6 <= cfg["replicas_max"]
    m[:, DOWN2] = act.r - 2 >= cfg["replicas_min"]
    m[:, WARM] = act.since_warm >= cfg["cache_warm_every"]
    m[:, SHED] = act.shed + cfg["shed_step"] <= cfg["shed_max"] + 1e-9
    m[:, FAILOVER] = (act.lockout == 0) & (act.failovers < cfg["failovers_per_day"])
    over = act.budget_used >= cfg["budget_actions"]
    m[over, 1:] = False
    return m


def apply_actions(act: Actuator, a: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    """Apply chosen actions in place (callers pass only allowed actions). Returns per-env flags."""
    a = np.asarray(a)
    up = np.where(a == UP2, 2, np.where(a == UP6, 6, 0))
    act.p2 += up
    down = a == DOWN2
    act.r = np.where(down, act.r - 2, act.r)
    warm = a == WARM
    act.since_warm = np.where(warm, 0, act.since_warm)
    shed = a == SHED
    act.shed = np.where(shed, act.shed + cfg["shed_step"], act.shed)
    act.shed_expiry[shed, -1] += cfg["shed_step"]
    fail = a == FAILOVER
    act.lockout = np.where(fail, cfg["failover_lockout_steps"], act.lockout)
    act.failovers = act.failovers + fail
    act.recent = np.roll(act.recent, -1, axis=1)
    act.recent[:, -1] = (a != NOOP).astype(np.int64)
    return {"warm": warm, "failover": fail, "scale_up": up > 0, "scale_down": down, "shed": shed}


def advance_time(act: Actuator) -> None:
    """End of a step: warm-up pipeline moves, shed increments expire, cooldowns tick."""
    act.r = act.r + act.p1
    act.p1 = act.p2
    act.p2 = np.zeros_like(act.p2)
    act.shed = np.maximum(act.shed - act.shed_expiry[:, 0], 0.0)
    act.shed_expiry = np.roll(act.shed_expiry, -1, axis=1)
    act.shed_expiry[:, -1] = 0.0
    act.since_warm = act.since_warm + 1
    act.lockout = np.maximum(act.lockout - 1, 0)


def physics(load: np.ndarray, r: np.ndarray, cache: np.ndarray, health: np.ndarray, shed: np.ndarray,
            queue: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    mu = cfg["mu"]
    speed = mu * (0.7 + 0.3 * cache) * health
    kappa = np.maximum(r, 1) * speed
    lam_s = load * (1.0 - shed)
    rho = lam_s / kappa
    q_raw = np.maximum(0.0, queue + cfg["step_seconds"] * (lam_s - kappa))
    # clients time out: backlog beyond `timeout_s` of capacity is dropped and counted as errors
    q_max = cfg["timeout_s"] * kappa
    q_next = np.minimum(q_raw, q_max)
    dropped = (q_raw - q_next) / (cfg["step_seconds"] * np.maximum(lam_s, 1e-6))
    rho_t = np.minimum(rho, cfg["rho_cap"])
    wq = rho_t ** np.sqrt(2.0 * (r + 1.0)) / ((1.0 - rho_t) * np.maximum(lam_s, 1e-6))
    p95_ms = 1000.0 * (1.0 / speed + 3.0 * wq + q_next / kappa)
    err = np.minimum(0.05 * (1.0 - health) + 0.2 / (1.0 + np.exp(-40.0 * (rho - 1.05))) + dropped, 1.0)
    return {"rho": rho, "queue": q_next, "p95_ms": p95_ms, "error_rate": err}


def known_cost(act: Actuator, flags: Dict[str, np.ndarray], cfg: Dict) -> np.ndarray:
    """The part of a step's cost that follows from the actuator state and the action alone (replicas,
    shed traffic, cache warms, failovers). Every controller can compute it exactly."""
    return (cfg["cost_replica_step"] * (act.r + act.pending)
            + cfg["cost_shed_fraction"] * act.shed
            + cfg["cost_cache_warm"] * flags["warm"]
            + cfg["cost_failover"] * flags["failover"])


def step_cost(act: Actuator, flags: Dict[str, np.ndarray], p95_ms: np.ndarray, err: np.ndarray, cfg: Dict):
    """Step cost = known cost + the SLO-violation penalty, the only part that depends on the hidden state."""
    violation = (p95_ms > cfg["slo_latency_ms"]) | (err > cfg["slo_error_rate"])
    return known_cost(act, flags, cfg) + cfg["cost_violation_step"] * violation, violation


@dataclass
class Fleet:
    """A batch of E environments stepping through their load profiles."""
    cfg: Dict
    profiles: List[LoadProfile]
    act: Actuator
    t: int = 0
    queue: np.ndarray = field(default=None)
    cache: np.ndarray = field(default=None)
    failed_over: np.ndarray = field(default=None)
    last: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def start(cls, profiles: List[LoadProfile], r0: np.ndarray, cfg: Dict) -> "Fleet":
        E = len(profiles)
        f = cls(cfg=cfg, profiles=profiles, act=new_actuator(np.asarray(r0), cfg))
        f.queue = np.zeros(E)
        f.cache = np.full(E, 0.5)
        f.failed_over = np.zeros(E, dtype=bool)
        f._measure(np.zeros(E, dtype=int))
        return f

    @property
    def E(self) -> int:
        return len(self.profiles)

    def health(self, t: Optional[int] = None) -> np.ndarray:
        t = self.t if t is None else t
        h = np.ones(self.E)
        for i, p in enumerate(self.profiles):
            if p.incident_start >= 0 and p.incident_start <= t < p.incident_start + p.incident_steps:
                h[i] = self.cfg["incident_health"]
        return np.where(self.failed_over, 1.0, h)

    def load(self, t: Optional[int] = None) -> np.ndarray:
        t = self.t if t is None else t
        T = self.cfg["steps_per_episode"]
        return np.array([p.load[min(t, T - 1)] for p in self.profiles])

    def _measure(self, _a) -> None:
        h = self.health()
        ph = physics(self.load(), self.act.r, self.cache, h, self.act.shed, self.queue, self.cfg)
        self.last = {"load": self.load(), "health": h, **ph}

    def true_state(self) -> np.ndarray:
        """(E, 8): load, queue, replicas, pending, cache, health, p95_ms, error_rate."""
        L = self.last
        return np.stack([L["load"], L["queue"], self.act.r, self.act.pending, self.cache, L["health"],
                         L["p95_ms"], L["error_rate"]], axis=1).astype(np.float64)

    def golden(self) -> Dict[str, np.ndarray]:
        """Clean utilisation, latency and error rate: read only by rule baselines and the approver."""
        return {"rho": self.last["rho"].copy(), "p95_ms": self.last["p95_ms"].copy(),
                "error_rate": self.last["error_rate"].copy(), "load": self.last["load"].copy()}

    def step(self, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply one action per environment (must be allowed), then advance one 5-minute step."""
        cfg = self.cfg
        a = np.asarray(actions)
        mask = allowed_mask(self.act, cfg)
        if not mask[np.arange(self.E), a].all():
            bad = [(i, ACTIONS[a[i]]) for i in range(self.E) if not mask[i, a[i]]]
            raise ValueError(f"disallowed actions reached the fleet: {bad[:5]}")
        flags = apply_actions(self.act, a, cfg)
        self.cache = np.where(flags["warm"], self.cache + cfg["cache_warm_gain"] * (1 - self.cache), self.cache)
        self.failed_over = self.failed_over | flags["failover"]
        self.cache = np.where(flags["failover"], cfg["cache_after_failover"], self.cache)
        self.t += 1
        h = self.health()
        load = self.load()
        ph = physics(load, self.act.r, self.cache, h, self.act.shed, self.queue, cfg)
        cost, violation = step_cost(self.act, flags, ph["p95_ms"], ph["error_rate"], cfg)
        self.queue = ph["queue"]
        self.last = {"load": load, "health": h, **ph}
        advance_time(self.act)
        self.cache = self.cache * cfg["cache_decay"]
        return {"cost": cost, "violation": violation, **flags}

    @property
    def done(self) -> bool:
        return self.t >= self.cfg["steps_per_episode"] - 1


def simulate_open_loop(start_queue, start_cache, start_failed, act: Actuator, loads: np.ndarray,
                       health: np.ndarray, action_seqs: np.ndarray, cfg: Dict) -> np.ndarray:
    """Roll the true dynamics for N parallel copies under fixed action sequences (oracle planning).

    loads, health: (N, H); action_seqs: (N, H). Disallowed actions become noop. A failover chosen at
    step k restores health from step k+1, which models the one-step approval delay. Returns (N,) cost
    sums discounted by 0.97 per step."""
    N, H = action_seqs.shape
    act = act.copy()
    queue, cache, failed = start_queue.copy(), start_cache.copy(), start_failed.copy()
    total = np.zeros(N)
    for k in range(H):
        a = action_seqs[:, k].copy()
        m = allowed_mask(act, cfg)
        a = np.where(m[np.arange(N), a], a, NOOP)
        flags = apply_actions(act, a, cfg)
        cache = np.where(flags["warm"], cache + cfg["cache_warm_gain"] * (1 - cache), cache)
        h = np.where(failed, 1.0, health[:, k])
        failed = failed | flags["failover"]
        cache = np.where(flags["failover"], cfg["cache_after_failover"], cache)
        ph = physics(loads[:, k], act.r, cache, h, act.shed, queue, cfg)
        cost, _ = step_cost(act, flags, ph["p95_ms"], ph["error_rate"], cfg)
        total += (0.97 ** k) * cost
        queue = ph["queue"]
        advance_time(act)
        cache = cache * cfg["cache_decay"]
    return total
