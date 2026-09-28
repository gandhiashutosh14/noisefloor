"""Rule-based controllers. They read the clean ("golden") utilisation, latency and error channels,
which the learned agents never see: a declared handicap in favour of the baselines.

HPA follows the Kubernetes Horizontal Pod Autoscaler algorithm:
    desired = ceil(current_replicas * current_metric / target_metric)
with a 10% tolerance (no change when |metric/target - 1| <= 0.1) and a scale-down stabilisation
window (the highest recommendation of the last W steps is used when scaling down; W = 1 step is the
Kubernetes default of 300 s at 5-minute steps). The desired count maps onto the nearest scale action.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from ..sim.fleet import DOWN2, FAILOVER, NOOP, SHED, UP2, UP6
from ..sim.loads import diurnal


@dataclass
class Context:
    """What an agent may read at a decision point."""
    t: int
    obs: Optional[np.ndarray]          # (E, 128) telemetry, for learned agents
    golden: Dict[str, np.ndarray]      # clean channels, for rule agents and the approver
    actuator: "object"                 # sim.fleet.Actuator
    act_vec: np.ndarray                # (E, 6)
    mask: np.ndarray                   # (E, 7) gate-allowed actions
    true_state: Optional[np.ndarray] = None   # only the oracle reads this


def _scale_action(desired: np.ndarray, total: np.ndarray) -> np.ndarray:
    gap = desired - total
    return np.where(gap >= 4, UP6, np.where(gap >= 1, UP2, np.where(gap <= -2, DOWN2, NOOP)))


class HPA:
    name = "hpa"

    def __init__(self, target: float = 0.6, window: int = 1, tolerance: float = 0.1):
        self.target, self.window, self.tolerance = target, window, tolerance
        self.history: List[Deque[int]] = []

    def reset(self, E: int, cfg: Dict) -> None:
        self.history = [deque(maxlen=self.window) for _ in range(E)]

    def desired(self, ctx: Context) -> np.ndarray:
        act = ctx.actuator
        rho = ctx.golden["rho"]
        total = act.r + act.pending
        raw = np.ceil(np.maximum(act.r, 1) * rho / self.target).astype(np.int64)
        within = np.abs(rho / self.target - 1.0) <= self.tolerance
        rec = np.where(within, total, raw)
        out = np.empty_like(rec)
        for i, v in enumerate(rec):
            self.history[i].append(int(v))
            # scale-down stabilisation: never go below the highest recent recommendation
            out[i] = v if v >= total[i] else min(total[i], max(self.history[i]))
        return out

    def propose(self, ctx: Context) -> np.ndarray:
        total = ctx.actuator.r + ctx.actuator.pending
        a = _scale_action(self.desired(ctx), total)
        return np.stack([a, np.full_like(a, NOOP), np.full_like(a, NOOP)], axis=1)


class Runbook(HPA):
    """HPA plus the two things an on-call runbook adds: shed load when utilisation is far over
    capacity, and ask for failover when errors stay above 2% for two steps."""
    name = "runbook"

    def reset(self, E: int, cfg: Dict) -> None:
        super().reset(E, cfg)
        self.err_hist = [deque(maxlen=2) for _ in range(E)]

    def propose(self, ctx: Context) -> np.ndarray:
        base = super().propose(ctx)
        out = base.copy()
        for i in range(len(out)):
            self.err_hist[i].append(float(ctx.golden["error_rate"][i]))
            cands = []
            if len(self.err_hist[i]) == 2 and all(e > 0.02 for e in self.err_hist[i]):
                cands.append(FAILOVER)
            if ctx.golden["rho"][i] > 1.1:
                cands.append(SHED)
            cands.append(int(base[i, 0]))
            cands = (cands + [NOOP, NOOP, NOOP])[:3]
            out[i] = cands
        return out


class Predictive:
    """Forecast load two steps ahead from a time-of-day profile learned on training days, corrected by
    a Holt linear trend on the recent ratio of observed to profile load; size replicas for it."""
    name = "predictive"

    def __init__(self, profile: np.ndarray, target: float = 0.7, alpha: float = 0.5, beta: float = 0.3):
        self.profile, self.target, self.alpha, self.beta = profile, target, alpha, beta

    def reset(self, E: int, cfg: Dict) -> None:
        self.level = np.ones(E)
        self.trend = np.zeros(E)
        self.cfg = cfg

    def propose(self, ctx: Context) -> np.ndarray:
        T = len(self.profile)
        load = ctx.golden["load"]
        ratio = load / self.profile[min(ctx.t, T - 1)]
        prev = self.level
        self.level = self.alpha * ratio + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev) + (1 - self.beta) * self.trend
        fut = self.profile[min(ctx.t + 2, T - 1)] * (self.level + 2 * self.trend)
        act = ctx.actuator
        speed = load / np.maximum(ctx.golden["rho"] * np.maximum(act.r, 1), 1e-6)   # per-replica req/s now
        desired = np.ceil(fut / (self.target * np.maximum(speed, 1e-6))).astype(np.int64)
        a = _scale_action(desired, act.r + act.pending)
        return np.stack([a, np.full_like(a, NOOP), np.full_like(a, NOOP)], axis=1)


class Static:
    """Provision for the 99th percentile of training load and hold."""
    name = "static"

    def __init__(self, replicas: int):
        self.replicas = replicas

    def reset(self, E: int, cfg: Dict) -> None:
        pass

    def propose(self, ctx: Context) -> np.ndarray:
        total = ctx.actuator.r + ctx.actuator.pending
        a = _scale_action(np.full_like(total, self.replicas), total)
        return np.stack([a, np.full_like(a, NOOP), np.full_like(a, NOOP)], axis=1)


def training_profile(cfg: Dict, loads: np.ndarray) -> np.ndarray:
    """Mean training load per time of day (loads: (episodes, T)); falls back to the diurnal shape."""
    if loads is None or len(loads) == 0:
        return cfg["mean_load"] * diurnal(cfg["steps_per_episode"], cfg["diurnal_amplitude"])
    return loads.mean(0)
