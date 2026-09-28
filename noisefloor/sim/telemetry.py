"""Telemetry: what the learned agents see instead of the true state.

    u_t = A phi(s_t) + beta_d Q xi_t
    o_t = softplus(u_t) + 0.05 eps_t,   then z-scored with training statistics

phi(s) are 12 normalised state features (load, queue, log1p queue, replicas, pending, cache, health,
p95, error rate, utilisation, sin/cos time of day). A (128 x 12) is a fixed Gaussian mixing, so
every channel carries every feature. Q (128 x k) has k orthonormal random columns: k independent
distractor sources (think batch jobs, garbage collection, noisy neighbours) each leak into every
channel, so no channel is clean and dropping channels removes nothing. beta_d is set so that
distractors make up a share d of the mean channel variance of u.

k (``distractor_dims`` in configs/sim.json) is 16. The first pilot used k = 128 (isotropic noise
spread over every direction) and found no advantage for JEPA: isotropic noise has little variance
per direction, so a reconstruction bottleneck discards it for free. Changing k is the one
pre-registration change the spec allows after a flat pilot; both pilots are reported.

Two distractor arms, both independent of the state and of the actions:
  main (unpredictable): 50% Student-t(3), 25% sparse bursts Bernoulli(0.05) x Exp(1), 25% AR(1)
                        with rho <= 0.2. Lag-1 autocorrelation stays below 0.2, so the past does not
                        predict the next value.
  predictable (falsification arm): 50% AR(1) with rho in [0.95, 0.99], 25% random walks that reset,
                        25% sawtooth waves with random periods. A predictor can forecast these.
  isotropic (robustness arm): the main arm's sources, but k = 128 of them, so the noise is spread
                        evenly over every direction of observation space (the original design).

The distractor stream for an episode depends only on (world seed, episode id), so every agent in
an evaluation sees exactly the same noise (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

N_CHANNELS = 128
N_FEATURES = 12


def features(state: np.ndarray, rho: np.ndarray, t: np.ndarray, cfg: Dict) -> np.ndarray:
    """(E, 12) normalised features from the true state (E, 8), utilisation and the step index."""
    load, queue, r, pending, cache, health, p95, err = state.T
    T = cfg["steps_per_episode"]
    tod = 2 * np.pi * np.asarray(t) / T
    return np.stack([
        load / cfg["mean_load"] - 1.0,
        np.minimum(queue / 1e4, 5.0),
        np.log1p(queue) / 10.0,
        r / cfg["replicas_max"],
        pending / cfg["replicas_max"],
        cache,
        health,
        np.log(np.maximum(p95, 1.0)) - np.log(cfg["slo_latency_ms"]),
        np.minimum(err * 20.0, 5.0),
        np.minimum(rho, 3.0),
        np.sin(tod) * np.ones_like(load),
        np.cos(tod) * np.ones_like(load),
    ], axis=1)


@dataclass
class TelemetryWorld:
    """The fixed mixing matrices and scaling for one world seed and one distractor share."""
    A: np.ndarray
    Q: np.ndarray
    beta: float
    d: float
    arm: str
    world_seed: int
    k: int = N_CHANNELS
    mean: Optional[np.ndarray] = None     # z-score statistics, fitted on training observations
    std: Optional[np.ndarray] = None

    @classmethod
    def build(cls, world_seed: int, d: float, arm: str, feature_var: float, k: int = N_CHANNELS,
              source_var: float = 1.0) -> "TelemetryWorld":
        """feature_var: mean channel variance of A phi(s) over training states (for the share d).
        k sources of mean variance source_var through orthonormal columns add k * source_var / 128
        variance per channel on average."""
        rng = np.random.default_rng(world_seed)
        A = rng.standard_normal((N_CHANNELS, N_FEATURES)) / np.sqrt(N_FEATURES)
        Q, _ = np.linalg.qr(rng.standard_normal((N_CHANNELS, k)))
        beta = 0.0 if d <= 0 else float(np.sqrt(d / (1.0 - d) * feature_var * N_CHANNELS / (k * source_var)))
        return cls(A=A, Q=Q, beta=beta, d=d, arm=arm, world_seed=world_seed, k=k)

    def render(self, phi: np.ndarray, xi: np.ndarray, eps: np.ndarray) -> np.ndarray:
        u = phi @ self.A.T + self.beta * (xi @ self.Q.T)
        o = np.logaddexp(0.0, u) + 0.05 * eps
        if self.mean is not None:
            o = (o - self.mean) / self.std
        return o.astype(np.float32)

    def fit_normaliser(self, raw: np.ndarray) -> None:
        self.mean = raw.mean(0)
        self.std = raw.std(0) + 1e-6


def mean_feature_variance(A_seed: int, phi: np.ndarray) -> float:
    rng = np.random.default_rng(A_seed)
    A = rng.standard_normal((N_CHANNELS, N_FEATURES)) / np.sqrt(N_FEATURES)
    return float((phi @ A.T).var(0).mean())


class DistractorStream:
    """Standardised distractors xi_t (128,) for one episode; deterministic in (world_seed, episode_id, arm)."""

    def __init__(self, world_seed: int, episode_id: int, arm: str, n: int = N_CHANNELS):
        if arm not in ("main", "predictable", "isotropic"):
            raise ValueError(f"unknown distractor arm {arm!r}")
        arm = "main" if arm == "isotropic" else arm          # isotropic = main-arm sources, k = 128 of them
        self.rng = np.random.default_rng([world_seed, 7919, episode_id, 0 if arm == "main" else 1])
        self.arm = arm
        self.n = n
        g = self.rng
        if arm == "main":
            self.n_t, self.n_b = n // 2, n // 4
            self.n_ar = n - self.n_t - self.n_b
            self.phi = g.uniform(0.0, 0.2, self.n_ar)
            self.ar = g.standard_normal(self.n_ar)
        else:
            self.n_ar, self.n_rw = n // 2, n // 4
            self.n_saw = n - self.n_ar - self.n_rw
            self.phi = g.uniform(0.95, 0.99, self.n_ar)
            self.ar = g.standard_normal(self.n_ar)
            self.rw = g.standard_normal(self.n_rw)
            self.period = g.integers(12, 96, self.n_saw)
            self.phase = g.integers(0, 96, self.n_saw)
            self.t = 0

    def next(self) -> np.ndarray:
        g = self.rng
        if self.arm == "main":
            t3 = g.standard_t(3, self.n_t) / np.sqrt(3.0)                      # unit variance
            bursts = (g.random(self.n_b) < 0.05) * g.exponential(1.0, self.n_b)
            bursts = (bursts - 0.05) / np.sqrt(0.05 * 2 - 0.05 ** 2)           # E=0.05, Var=0.0975
            self.ar = self.phi * self.ar + np.sqrt(1 - self.phi ** 2) * g.standard_normal(self.n_ar)
            return np.concatenate([t3, bursts, self.ar])
        self.ar = self.phi * self.ar + np.sqrt(1 - self.phi ** 2) * g.standard_normal(self.n_ar)
        reset = g.random(self.n_rw) < 0.01
        self.rw = np.where(reset, 0.0, self.rw + 0.15 * g.standard_normal(self.n_rw))
        rw = np.clip(self.rw, -3, 3)
        saw = ((self.t + self.phase) % self.period) / self.period
        saw = (saw - 0.5) * np.sqrt(12.0)
        self.t += 1
        return np.concatenate([self.ar, rw, saw])

    def noise(self, n: int = N_CHANNELS) -> np.ndarray:
        """Small per-channel observation noise eps_t (independent of xi)."""
        return self.rng.standard_normal(n)
