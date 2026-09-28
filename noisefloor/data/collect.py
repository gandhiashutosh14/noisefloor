"""Offline data: behaviour trajectories once, rendered as telemetry at every noise level.

Behaviour policy: the clean-channel HPA with a utilisation target drawn per day from U(0.4, 1.3), 30%
of steps replaced by a random gate-allowed action, random initial replicas r0 ~ U{2..40}, and a 2%
chance per step of resetting the replica count to a random value. The per-day target makes the data
a mix of cautious and reckless operators: a planner considers running hot, so the data must show
what running hot costs. (With a fixed target of 0.6, few replicas almost always meant low load, and
every learned model concluded that scaling down was safe; see PREREGISTRATION.md, planning checks.) Each
trajectory is generated once and rendered at every (arm, d); levels therefore differ only in
observation noise, never in the underlying days.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..agents.rules import HPA, Context
from ..sim.fleet import N_ACTIONS, NOOP, Fleet, allowed_mask
from ..sim.loads import make_profile
from ..sim.telemetry import DistractorStream, TelemetryWorld, features

WORLD_SEED = 1234
DATA_SEED = 20260929
LEVELS: Tuple[Tuple[str, float], ...] = (("main", 0.0), ("main", 0.5), ("main", 0.8), ("main", 0.95), ("predictable", 0.8),
                                         ("isotropic", 0.8))


@dataclass
class Trajectories:
    episode_id: np.ndarray       # (N,)
    family: List[str]
    split: List[str]
    state: np.ndarray            # (N, T, 8) true state, never a model input
    rho: np.ndarray              # (N, T)
    act_vec: np.ndarray          # (N, T, 6) actuator features
    action: np.ndarray           # (N, T) action taken at t
    cost: np.ndarray             # (N, T) cost of step t
    reset: np.ndarray            # (N, T) replica reset happened after step t
    violation: np.ndarray        # (N, T) the SLO was violated during step t (the cost's only hidden part)
    loads: np.ndarray            # (N, T) load profile (for the predictive baseline)

    @property
    def T(self) -> int:
        return self.state.shape[1]


def collect(cfg: Dict, n_train: int = 200, n_val: int = 50, seed: int = DATA_SEED, explore: float = 0.3,
            reset_p: float = 0.02, target_range: Tuple[float, float] = (0.4, 1.3)) -> Trajectories:
    N = n_train + n_val
    rng = np.random.default_rng(seed)
    fams = cfg["families"]
    families = [fams[i % len(fams)] for i in range(N)]
    profiles = [make_profile(f, np.random.default_rng([seed, i]), cfg) for i, f in enumerate(families)]
    r0 = rng.integers(cfg["replicas_min"], cfg["replicas_max"] + 1, size=N)
    fleet = Fleet.start(profiles, r0, cfg)
    hpa = HPA(rng.uniform(*target_range, size=N), 1)
    hpa.reset(N, cfg)
    T = cfg["steps_per_episode"] - 1
    state = np.zeros((N, T, 8), np.float32)
    rho = np.zeros((N, T), np.float32)
    act_vec = np.zeros((N, T, 6), np.float32)
    action = np.zeros((N, T), np.int8)
    cost = np.zeros((N, T), np.float32)
    reset = np.zeros((N, T), bool)
    violation = np.zeros((N, T), bool)
    for t in range(T):
        state[:, t] = fleet.true_state()
        rho[:, t] = fleet.last["rho"]
        act_vec[:, t] = fleet.act.vector(cfg)
        mask = allowed_mask(fleet.act, cfg)
        mask[:, -1] = False       # the behaviour policy never fails over on its own
        ctx = Context(t=t, obs=None, golden=fleet.golden(), actuator=fleet.act, act_vec=act_vec[:, t], mask=mask)
        a = hpa.propose(ctx)[:, 0]
        a = np.where(mask[np.arange(N), a], a, NOOP)
        explore_now = rng.random(N) < explore
        rand = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
        a = np.where(explore_now, rand, a)
        # an incident day gets a failover opportunity: 20% of steps in the window try it
        inc = fleet.last["health"] < 1.0
        allowed_fail = allowed_mask(fleet.act, cfg)[:, -1]
        try_fail = inc & allowed_fail & (rng.random(N) < 0.2)
        a = np.where(try_fail, N_ACTIONS - 1, a)
        out = fleet.step(a)
        action[:, t] = a
        cost[:, t] = out["cost"]
        violation[:, t] = out["violation"]
        do_reset = rng.random(N) < reset_p
        if do_reset.any():
            new_r = rng.integers(cfg["replicas_min"], cfg["replicas_max"] + 1, size=N)
            fleet.act.r = np.where(do_reset, new_r, fleet.act.r)
            fleet.act.p1 = np.where(do_reset, 0, fleet.act.p1)
            fleet.act.p2 = np.where(do_reset, 0, fleet.act.p2)
        reset[:, t] = do_reset
    split = ["train"] * n_train + ["val"] * n_val
    return Trajectories(np.arange(N), families, split, state, rho, act_vec, action, cost, reset, violation,
                        np.stack([p.load for p in profiles]).astype(np.float32))


def feature_matrix(tr: Trajectories, cfg: Dict) -> np.ndarray:
    N, T = tr.state.shape[:2]
    t = np.broadcast_to(np.arange(T)[None, :], (N, T)).reshape(-1)
    return features(tr.state.reshape(-1, 8).astype(np.float64), tr.rho.reshape(-1).astype(np.float64), t, cfg).reshape(N, T, -1)


def build_world(tr: Trajectories, cfg: Dict, arm: str, d: float, world_seed: int = WORLD_SEED) -> TelemetryWorld:
    phi = feature_matrix(tr, cfg)
    train = np.array([s == "train" for s in tr.split])
    rng = np.random.default_rng(world_seed)
    A = rng.standard_normal((128, phi.shape[-1])) / np.sqrt(phi.shape[-1])
    var = float((phi[train].reshape(-1, phi.shape[-1]) @ A.T).var(0).mean())
    k = 128 if arm == "isotropic" else cfg.get("distractor_dims", 128)
    return TelemetryWorld.build(world_seed, d, arm, var, k=k, source_var=source_variance(world_seed, arm, k, tr.T))


def source_variance(world_seed: int, arm: str, k: int, T: int, n_streams: int = 32) -> float:
    """Mean variance of the distractor sources over day-long streams (calibration ids 900000+, never
    used for training or evaluation). Slow sources (random walks) exceed unit variance within a day, so
    beta is calibrated on what a day of the arm actually looks like."""
    xs = []
    for e in range(n_streams):
        s = DistractorStream(world_seed, 900_000 + e, arm, n=k)
        xs.append(np.stack([s.next() for _ in range(T)]))
    return float(np.concatenate(xs).var(0).mean())


def render(tr: Trajectories, cfg: Dict, world: TelemetryWorld, episode_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(N, T, 128) observations and (N, T, 16) of the distractor values (for probes only)."""
    phi = feature_matrix(tr, cfg)
    N, T, _ = phi.shape
    streams = [DistractorStream(world.world_seed, int(eid) + episode_offset, world.arm, n=world.k) for eid in tr.episode_id]
    xi = np.zeros((N, T, world.k))
    eps = np.zeros((N, T, 128))
    for t in range(T):
        xi[:, t] = np.stack([s.next() for s in streams])
        eps[:, t] = np.stack([s.noise(128) for s in streams])
    saved = (world.mean, world.std)
    world.mean, world.std = None, None
    raw = world.render(phi.reshape(-1, phi.shape[-1]), xi.reshape(-1, world.k), eps.reshape(-1, 128)).reshape(N, T, 128)
    world.mean, world.std = saved
    if world.mean is None:
        train = np.array([s == "train" for s in tr.split])
        world.fit_normaliser(raw[train].reshape(-1, 128))
    obs = ((raw - world.mean) / world.std).astype(np.float32)
    return obs, xi[..., :16].astype(np.float32)


def world_meta(world: TelemetryWorld) -> Dict:
    return {"world_seed": world.world_seed, "d": world.d, "arm": world.arm, "beta": world.beta, "k": world.k,
            "mean": world.mean.tolist(), "std": world.std.tolist()}


def world_from_meta(meta: Dict) -> TelemetryWorld:
    w = TelemetryWorld.build(meta["world_seed"], meta["d"], meta["arm"], 1.0, k=meta.get("k", 128))
    w.beta = meta["beta"]
    w.mean = np.asarray(meta["mean"], np.float64)
    w.std = np.asarray(meta["std"], np.float64)
    return w


def level_key(arm: str, d: float) -> str:
    return f"{arm}_d{int(round(d * 100)):02d}"


def save_meta(path: Path, world: TelemetryWorld) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world_meta(world)), encoding="utf-8")
