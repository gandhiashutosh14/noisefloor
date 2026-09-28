"""Load profiles: one day of request rate per episode, plus the incident schedule.

lambda_t = mean_load * b_t * trend_t * (1 + sum_j A_j exp(-(t - t_j)/tau) 1[t >= t_j]) * (1 + 0.05 eta_t)
b_t = 1 + 0.5 sin(2 pi t / 288 - pi/2)   (a day: trough at midnight, peak at noon)

Families (equal mix in training): diurnal, flash crowd (Poisson spikes), ramp (growth through the
day), incident (region health drops to 0.4 for up to 36 steps unless the agent fails over). Two
out-of-distribution families are only ever used for evaluation: a 24-step x2.5 launch plateau and
a double spike.

``expected_profile`` is what a forecaster may know: the diurnal and trend terms, without the random
spikes or the incident. The non-clairvoyant oracle plans on it plus spikes sampled from the prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LoadProfile:
    family: str
    load: np.ndarray            # (T,) requests per second
    expected: np.ndarray        # (T,) the forecastable part: diurnal x trend
    incident_start: int         # -1 when there is no incident
    incident_steps: int


def diurnal(T: int, amplitude: float) -> np.ndarray:
    t = np.arange(T)
    return 1.0 + amplitude * np.sin(2 * np.pi * t / T - np.pi / 2)


def _spikes(T: int, times, amps, tau: float) -> np.ndarray:
    out = np.zeros(T)
    t = np.arange(T)
    for tj, aj in zip(times, amps):
        mask = t >= tj
        out[mask] += aj * np.exp(-(t[mask] - tj) / tau)
    return out


def make_profile(family: str, rng: np.random.Generator, cfg: Dict) -> LoadProfile:
    T = int(cfg["steps_per_episode"])
    base = cfg["mean_load"] * diurnal(T, cfg["diurnal_amplitude"])
    trend = np.ones(T)
    spikes = np.zeros(T)
    plateau = np.ones(T)
    inc_start, inc_steps = -1, 0
    if family == "flash":
        n = rng.poisson(cfg["flash_spikes_per_day"])
        times = rng.integers(0, T, size=n)
        amps = rng.uniform(*cfg["flash_amplitude"], size=n)
        spikes = _spikes(T, times, amps, cfg["spike_decay_steps"])
    elif family == "ramp":
        trend = 1.0 + cfg["ramp_growth"] * np.arange(T) / T
    elif family == "incident":
        inc_start = int(rng.integers(cfg["incident_start"][0], cfg["incident_start"][1] + 1))
        inc_steps = int(cfg["incident_steps"])
    elif family == "launch":                       # OOD: a sustained plateau never seen in training
        start = int(rng.integers(60, 200))
        plateau[start:start + 24] = 2.5
    elif family == "double_spike":                 # OOD: two large spikes close together
        t0 = int(rng.integers(40, 230))
        spikes = _spikes(T, [t0, t0 + int(rng.integers(3, 10))], rng.uniform(1.5, 2.0, size=2),
                         cfg["spike_decay_steps"])
    elif family != "diurnal":
        raise ValueError(f"unknown load family {family!r}")
    noise = 1.0 + cfg["load_noise"] * rng.standard_normal(T)
    expected = base * trend
    load = np.maximum(expected * plateau * (1.0 + spikes) * noise, 1.0)
    return LoadProfile(family, load, expected, inc_start, inc_steps)


def sample_future_load(expected: np.ndarray, t: int, horizon: int, current: float, rng: np.random.Generator,
                       cfg: Dict, n: int) -> np.ndarray:
    """(n, horizon) load paths for the oracle: the expected profile, anchored to the current level,
    with spikes drawn from the flash-crowd prior. The oracle never sees the realised future."""
    T = len(expected)
    idx = np.minimum(np.arange(t + 1, t + 1 + horizon), T - 1)
    ratio = current / max(expected[min(t, T - 1)], 1.0)
    decay = np.exp(-np.arange(1, horizon + 1) / cfg["spike_decay_steps"])
    # the current excess over the forecast decays like a spike; new spikes arrive at the prior rate
    level = expected[idx][None, :] * (1.0 + (ratio - 1.0) * decay[None, :])
    p_spike = cfg["flash_spikes_per_day"] / (4.0 * T)        # a quarter of episodes are flash crowds
    arrivals = rng.random((n, horizon)) < p_spike
    amps = rng.uniform(*cfg["flash_amplitude"], size=(n, horizon)) * arrivals
    extra = np.zeros((n, horizon))
    for k in range(horizon):
        extra[:, k:] += amps[:, k:k + 1] * np.exp(-np.arange(horizon - k) / cfg["spike_decay_steps"])[None, :]
    noise = 1.0 + cfg["load_noise"] * rng.standard_normal((n, horizon))
    return np.maximum(level * (1.0 + extra) * noise, 1.0)
