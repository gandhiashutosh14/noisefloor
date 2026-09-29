"""Oracle MPC: the same CEM, but planning on copies of the true simulator from the true state.

It is a reference, not an upper bound on what is achievable: it knows the dynamics, the current
true state and the forecastable part of the load (diurnal and trend), but not the random spikes or
incidents, which it samples from the prior (4 load paths per candidate sequence). Current region
health is assumed to persist over the horizon unless the plan fails over.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..agents.rules import Context
from ..sim.fleet import N_ACTIONS, Actuator, simulate_open_loop
from ..sim.loads import LoadProfile, sample_future_load
from .cem import cem_plan, shift


class OracleMPC:
    name = "oracle"

    def __init__(self, profiles: List[LoadProfile], cfg: Dict, seed: int = 0, horizon: int = 8, samples: int = 128,
                 elites: int = 16, iterations: int = 3, copies: int = 4):
        self.profiles, self.cfg = profiles, cfg
        self.rng = np.random.default_rng(seed)
        self.h, self.s, self.el, self.it, self.copies = horizon, samples, elites, iterations, copies
        self.p = None
        self.fleet = None   # set by the harness: the oracle may read the fleet's queue, cache and health

    def reset(self, E: int, cfg: Dict) -> None:
        self.p = None

    def propose(self, ctx: Context) -> np.ndarray:
        fleet = self.fleet
        E, H, S, C = len(self.profiles), self.h, self.s, self.copies
        loads = np.stack([sample_future_load(p.expected, ctx.t, H, float(ctx.golden["load"][i]), self.rng, self.cfg, C)
                          for i, p in enumerate(self.profiles)])                        # (E, C, H)
        health = np.repeat(fleet.last["health"][:, None], H, axis=1)                   # (E, H)

        def cost_fn(seqs: np.ndarray) -> np.ndarray:        # seqs (E, S, H)
            n = E * S * C
            act: Actuator = ctx.actuator.repeat(S * C)
            a = np.repeat(seqs.reshape(E * S, H), C, axis=0)
            L = np.tile(loads, (1, S, 1)).reshape(n, H)     # index (e, s, c) -> loads[e, c]
            hh = np.repeat(health, S * C, axis=0)
            q = np.repeat(fleet.queue, S * C)
            c = np.repeat(fleet.cache, S * C)
            f = np.repeat(fleet.failed_over, S * C)
            total = simulate_open_loop(q, c, f, act, L, hh, a, self.cfg, ctx.limits)
            return total.reshape(E, S, C).mean(-1)

        init = None if self.p is None else shift(self.p)
        ranked, _, self.p = cem_plan(E, N_ACTIONS, cost_fn, self.rng, horizon=H, samples=S, elites=self.el,
                                     iterations=self.it, init=init, first_mask=ctx.mask)
        return ranked
