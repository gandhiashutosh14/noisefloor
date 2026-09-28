"""SIGReg: sketched isotropic Gaussian regularisation (LeJEPA, Balestriero & LeCun, arXiv 2511.08544).

Written from the paper's equations, not copied from the LeJEPA reference code (CC BY-NC). The
numerical choices follow the MIT-licensed LeWorldModel implementation (github.com/lucas-maes/le-wm,
module.SIGReg), which is the world-model setting reproduced here:

  * the statistic is computed separately for each time step over the batch (N = batch size) and
    averaged over time steps and directions, not pooled over every frame of every window;
  * the integral runs over t in [-3, 3], evaluated on 17 knots of [0, 3] with doubled trapezoid
    weights (the integrand is even), window w(t) = exp(-t^2 / 2).

For embeddings Z (N x K) and M random unit directions a_m, with x = Z a_m:

    EP(x) = N * integral |phi_hat_x(t) - exp(-t^2/2)|^2 w(t) dt,   phi_hat_x(t) = (1/N) sum_j exp(i t x_j)

SIGReg is the mean of EP over directions (and time steps). Under the null (Z ~ N(0, I)) the
expectation is integral (1 - e^{-t^2}) e^{-t^2/2} dt ~ 1.0, independent of N. By the Cramer-Wold
argument in the paper, driving every 1-D marginal to N(0, 1) drives the embeddings to N(0, I), which
rules out collapse without EMA teachers or stop-gradients. Directions are redrawn every call.
"""
from __future__ import annotations

from typing import Optional

import torch


def sigreg(z: torch.Tensor, num_slices: int = 256, t_max: float = 3.0, num_points: int = 17,
           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sliced Epps-Pulley distance from N(0, I). z is (N, K) or (T, N, K) (one statistic per time
    step over the batch dimension N, then averaged). Returns a scalar."""
    if z.dim() == 2:
        z = z.unsqueeze(0)
    if z.dim() != 3:
        raise ValueError(f"expected (N, K) or (T, N, K) embeddings, got shape {tuple(z.shape)}")
    n, k = z.shape[-2], z.shape[-1]
    a = torch.randn(k, num_slices, device=z.device, dtype=z.dtype, generator=generator)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(0.0, t_max, num_points, device=z.device, dtype=z.dtype)
    dt = t_max / (num_points - 1)
    w = torch.full((num_points,), 2 * dt, device=z.device, dtype=z.dtype)
    w[0] = w[-1] = dt                                      # 2 x trapezoid on [0, t_max] = integral on [-t_max, t_max]
    target = torch.exp(-0.5 * t ** 2)                      # characteristic function of N(0, 1) = window
    xt = (z @ a).unsqueeze(-1) * t                          # (T, N, M, P)
    err = (torch.cos(xt).mean(-3) - target) ** 2 + torch.sin(xt).mean(-3) ** 2   # (T, M, P)
    ep = n * (err @ (w * target))                          # (T, M)
    return ep.mean()


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """exp(entropy of the normalised singular values): 1 for a collapsed code, K for an isotropic one."""
    zc = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(zc.double())
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))
