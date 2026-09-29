"""Diagnostics on a trained encoder: what the latent code contains and how well it predicts itself.

  state R^2      ridge probe z -> true state (8 transformed variables), fitted on training episodes,
                 scored on validation episodes; mean R^2 over variables
  distractor R^2 the same probe onto the first 16 distractor values xi (should be near 0)
  erank          effective rank of the validation latents
  k-step error   ||z_hat_{t+k} - E(o_{t+k})||^2 / total latent variance, k = 1..8

Probes read the true state and the distractors; they are diagnostics only and never touch training.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..models.nets import WorldModel
from ..models.sigreg import effective_rank
from ..models.train import CONTEXT, K, Windows

STATE_PROBE_NAMES = ["load", "log_queue", "replicas", "pending", "cache", "health", "log_p95", "error_rate"]


def transform_state(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64).copy()
    s[:, 1] = np.log1p(s[:, 1])
    s[:, 6] = np.log(np.maximum(s[:, 6], 1.0))
    return s


@torch.no_grad()
def encode(model: WorldModel, obs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(obs.reshape(-1, obs.shape[-1]), dtype=np.float32))
    return torch.cat([model.encoder(x[i:i + chunk]) for i in range(0, len(x), chunk)]).numpy()


def ridge_r2(x_tr, y_tr, x_te, y_te, alpha: float = 1.0) -> np.ndarray:
    mx, sx = x_tr.mean(0), x_tr.std(0) + 1e-8
    my, sy = y_tr.mean(0), y_tr.std(0) + 1e-8
    Xa, Ya = (x_tr - mx) / sx, (y_tr - my) / sy
    W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(Xa.shape[1]), Xa.T @ Ya)
    pred = ((x_te - mx) / sx) @ W * sy + my
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


def probe(model: WorldModel, fit: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Dict[str, float]:
    z_fit, z_te = encode(model, fit["obs"]), encode(model, test["obs"])
    s_fit, s_te = transform_state(fit["state"]), transform_state(test["state"])
    keep = s_fit.std(0) > 1e-6
    r2_state = ridge_r2(z_fit, s_fit[:, keep], z_te, s_te[:, keep])
    r2_xi = ridge_r2(z_fit, fit["xi16"], z_te, test["xi16"])
    names = [n for n, k in zip(STATE_PROBE_NAMES, keep) if k]
    out = {"probe_state_r2": float(np.mean(r2_state)), "probe_xi_r2": float(np.mean(r2_xi)),
           "erank": effective_rank(torch.from_numpy(z_te[:20000])),
           "z_std": float(z_te.std(0).mean())}      # effective rank ignores scale; a shrunk code shows here
    out.update({f"probe_r2_{n}": float(v) for n, v in zip(names, r2_state)})
    return out


@torch.no_grad()
def kstep_error(model: WorldModel, arrays: Dict[str, np.ndarray], n: int = 2048, seed: int = 0) -> Dict[str, float]:
    win = Windows(arrays)
    idx = np.random.default_rng(seed).integers(0, len(win), n)
    o, u, a, _ = win.batch(idx)
    B = o.shape[0]
    z = model.encoder(o.reshape(B * o.shape[1], -1)).reshape(B, o.shape[1], -1)
    z_hat = model.rollout(z[:, :CONTEXT], u[:, CONTEXT - 1:CONTEXT - 1 + K], a[:, CONTEXT - 1:CONTEXT - 1 + K])
    var = float(z.reshape(-1, z.shape[-1]).var(0).sum())
    err = ((z_hat - z[:, CONTEXT:]) ** 2).sum(-1).mean(0).numpy() / max(var, 1e-12)
    return {f"kstep_err_{k + 1}": float(e) for k, e in enumerate(err)}
