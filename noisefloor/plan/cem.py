"""Categorical cross-entropy method over discrete action sequences, batched across environments.

For each environment keep a distribution p (H x A) over actions at each horizon step. Each iteration
samples S sequences, scores them with ``cost_fn`` (lower is better), keeps the E_lite cheapest, and
moves p toward the elites' action frequencies: p <- (1 - alpha) p + alpha p_elite (alpha = 0.7).
The result is the ranked list of distinct first actions of the final elites, plus the spread of
their costs (a cheap uncertainty signal). ``warm`` shifts the previous distribution by one step.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def cem_plan(E: int, n_actions: int, cost_fn: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator, *,
             horizon: int = 8, samples: int = 128, elites: int = 16, iterations: int = 3, alpha: float = 0.7,
             init: Optional[np.ndarray] = None, top_k: int = 3,
             first_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cost_fn(seqs (E, S, H) int) -> (E, S) costs. Returns (ranked first actions (E, top_k),
    elite cost spread (E,), final distribution (E, H, A)).

    Two guards against a sticky search: a warm-start distribution is blended 50/50 with uniform, and
    every iteration also scores the n_actions constant sequences ("do a every step"), so the simplest
    plans are never missed because the sampler stopped proposing them.

    ``first_mask`` (E, A) bool says which first actions can actually execute now. The rollouts already
    score a refused action as a no-op, so without the mask a refused action ties with the no-op plan
    and can be ranked first; the ranking and its de-duplication are therefore done on the *effective*
    first action (refused -> action 0, the no-op)."""
    uniform = np.full((E, horizon, n_actions), 1.0 / n_actions)
    p = uniform if init is None else 0.5 * init + 0.5 * uniform
    constant = np.broadcast_to(np.arange(n_actions)[None, :, None], (E, n_actions, horizon))
    for _ in range(iterations):
        cum = p.cumsum(-1)
        u = rng.random((E, samples - n_actions, horizon, 1))
        sampled = (u > cum[:, None, :, :]).sum(-1).clip(0, n_actions - 1)       # (E, S - A, H)
        seqs = np.concatenate([constant, sampled], axis=1)                       # (E, S, H)
        costs = cost_fn(seqs)
        order = np.argsort(costs, axis=1)[:, :elites]                            # (E, elites)
        elite_seqs = np.take_along_axis(seqs, order[:, :, None], axis=1)         # (E, elites, H)
        freq = np.zeros_like(p)
        for a in range(n_actions):
            freq[..., a] = (elite_seqs == a).mean(1)
        p = (1 - alpha) * p + alpha * freq
    elite_costs = np.take_along_axis(costs, order, axis=1)
    spread = elite_costs.std(1)
    first = seqs[:, :, 0]
    if first_mask is not None:
        first = np.where(np.take_along_axis(first_mask, first, axis=1), first, 0)
    ranked = np.zeros((E, top_k), dtype=np.int64)
    for e in range(E):
        seen = []
        for s in order[e]:
            a0 = int(first[e, s])
            if a0 not in seen:
                seen.append(a0)
            if len(seen) == top_k:
                break
        while len(seen) < top_k:
            seen.append(0)
        ranked[e] = seen
    return ranked, spread, p


def shift(p: np.ndarray) -> np.ndarray:
    """Warm start: drop the first horizon step and append a uniform one."""
    out = np.roll(p, -1, axis=1)
    out[:, -1, :] = 1.0 / p.shape[-1]
    return out
