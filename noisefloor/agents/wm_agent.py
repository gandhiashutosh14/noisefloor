"""The learned agent: categorical CEM inside a world model's latent space.

At each step the agent encodes the current telemetry frame, keeps the last three latents as the
predictor's context, and scores candidate 8-step action sequences by

    sum_h 0.97^h [known(u_h, a_h) + 5 * sigmoid(R(z_h, u_h, a_h))],   z_0 = E(o_t),  z_{h+1} = P(z_{h-2..h}, u_h, a_h)

known() is the exact replica, shed, warm and failover cost; R is the learned violation risk.

The actuator state u (replicas, pending, shed, cooldowns, lockout, action budget) is known exactly,
so it is rolled forward with the simulator's own actuator code: an action the gate would refuse at
step h becomes a noop at step h, exactly as it would in the real loop. A failover chosen at step h
is filed for approval and takes effect at h + 1 (the approver's one-step delay).

The agent never sees the true state or the clean golden channels. It returns the three best distinct
first actions; the harness passes them through the gate in order.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from ..models.nets import HISTORY, WorldModel
from ..plan.cem import cem_plan, shift
from ..sim.fleet import FAILOVER, N_ACTIONS, NOOP, advance_time, allowed_mask, apply_actions, known_cost
from .rules import Context


def rollout_actuator(actuator, seqs: np.ndarray, cfg: Dict):
    """seqs (N, H) proposed actions -> (U (N, H, 6) actuator features before each step, A (N, H) the
    actions that would actually execute, C (N, H) their exactly known cost)."""
    N, H = seqs.shape
    act = actuator.copy()
    U = np.zeros((N, H, 6), np.float32)
    A = np.zeros((N, H), np.int64)
    C = np.zeros((N, H), np.float32)
    pending_fail = np.zeros(N, bool)
    rows = np.arange(N)
    for h in range(H):
        U[:, h] = act.vector(cfg)
        mask = allowed_mask(act, cfg)
        a = seqs[:, h]
        ok = mask[rows, a]
        eff = np.where(ok, a, NOOP)
        file_fail = (eff == FAILOVER) & ~pending_fail
        eff = np.where(file_fail, NOOP, eff)                 # requested now, executed next step
        run_fail = pending_fail & mask[:, FAILOVER]
        eff = np.where(run_fail, FAILOVER, eff)
        pending_fail = file_fail
        A[:, h] = eff
        flags = apply_actions(act, eff, cfg)
        C[:, h] = known_cost(act, flags, cfg)
        advance_time(act)
    return U, A, C


class WMAgent:
    """``model`` is a trained WorldModel; one agent instance drives a batch of E environments."""

    def __init__(self, model: WorldModel, cfg: Dict, seed: int = 0, horizon: int = 8, samples: int = 128,
                 elites: int = 16, iterations: int = 3, discount: float = 0.97, name: Optional[str] = None,
                 filter_alpha: Optional[float] = None, violation_scale: float = 1.0):
        self.model, self.cfg = model.eval(), cfg
        self.rng = np.random.default_rng(seed)
        self.h, self.s, self.el, self.it, self.discount = horizon, samples, elites, iterations, discount
        self.name = name or f"wm-{model.variant}"
        self.violation_cost = float(cfg["cost_violation_step"]) * violation_scale
        self.filter_alpha = filter_alpha
        self.hist: Optional[torch.Tensor] = None
        self.p = None
        self.last_pred_cost: Optional[np.ndarray] = None
        self.last_spread: Optional[np.ndarray] = None
        self.last_latent: Optional[np.ndarray] = None

    def reset(self, E: int, cfg: Dict) -> None:
        self.hist, self.p = None, None

    @torch.no_grad()
    def propose(self, ctx: Context) -> np.ndarray:
        if ctx.obs is None:
            raise ValueError("the world-model agent needs telemetry observations")
        z = self.model.encoder(torch.from_numpy(np.asarray(ctx.obs, np.float32)))
        if self.filter_alpha is not None and self.hist is not None:
            z = self.filter_alpha * z + (1 - self.filter_alpha) * self.hist[:, -1]
        self.hist = z[:, None].repeat(1, HISTORY, 1) if self.hist is None else torch.cat([self.hist[:, 1:], z[:, None]], 1)
        self.last_latent = z.numpy()
        E, S, H = z.shape[0], self.s, self.h
        disc = torch.tensor([self.discount ** k for k in range(H)], dtype=torch.float32)
        depth = torch.arange(H)

        def score(seqs: np.ndarray) -> np.ndarray:                             # (E, n, H) -> (E, n)
            n = seqs.shape[1]
            hist = self.hist.repeat_interleave(n, dim=0)
            U, A, C = rollout_actuator(ctx.actuator.repeat(n), seqs.reshape(E * n, H), self.cfg)
            Ut, At = torch.from_numpy(U), torch.from_numpy(A)
            z_img = self.model.rollout(hist, Ut[:, :-1], At[:, :-1])          # z_1 .. z_{H-1}
            zs = torch.cat([hist[:, -1:], z_img], dim=1)                      # z_0 .. z_{H-1}
            risk = torch.sigmoid(self.model.risk(zs, Ut, At, depth))          # (E*n, H), depth 0 .. H-1
            c = torch.from_numpy(C) + self.violation_cost * risk
            return (c * disc).sum(1).reshape(E, n).numpy()

        init = None if self.p is None else shift(self.p)
        ranked, spread, self.p = cem_plan(E, N_ACTIONS, score, self.rng, horizon=H, samples=S, elites=self.el,
                                          iterations=self.it, init=init)
        self.last_spread = spread
        self.last_pred_cost = score(self.p.argmax(-1)[:, None, :])[:, 0]      # cost of the most likely plan
        return ranked
