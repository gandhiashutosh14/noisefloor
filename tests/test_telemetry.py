import numpy as np
import pytest

from noisefloor.config import sim_config
from noisefloor.data.collect import build_world, collect, feature_matrix, render
from noisefloor.sim.telemetry import DistractorStream

CFG = sim_config()


@pytest.fixture(scope="module")
def trajectories():
    return collect(CFG, n_train=24, n_val=8, seed=99)


@pytest.mark.parametrize("arm,d", [("main", 0.5), ("main", 0.8), ("main", 0.95), ("predictable", 0.8), ("isotropic", 0.8)])
def test_distractor_share_of_variance_matches_d(trajectories, arm, d):
    world = build_world(trajectories, CFG, arm, d)
    phi = feature_matrix(trajectories, CFG).reshape(-1, 12)
    signal = phi @ world.A.T
    n = phi.shape[0]
    xi = np.concatenate([np.stack([s.next() for _ in range(n // 16 + 1)])
                         for s in [DistractorStream(world.world_seed, 50_000 + e, arm, n=world.k) for e in range(16)]])[:n]
    noise = world.beta * (xi @ world.Q.T)
    share = noise.var(0).mean() / (noise.var(0).mean() + signal.var(0).mean())
    assert abs(share - d) < 0.02


def test_main_arm_is_unpredictable_and_predictable_arm_is_not():
    main = DistractorStream(1, 5, "main", n=16).next()[None]
    s = DistractorStream(1, 5, "main", n=16)
    x = np.stack([s.next() for _ in range(4000)])
    lag1 = [np.corrcoef(x[:-1, j], x[1:, j])[0, 1] for j in range(16)]
    assert max(abs(v) for v in lag1) <= 0.2 + 0.05     # AR components have rho <= 0.2; sampling slack
    p = DistractorStream(1, 5, "predictable", n=16)
    y = np.stack([p.next() for _ in range(4000)])
    lag1p = [np.corrcoef(y[:-1, j], y[1:, j])[0, 1] for j in range(16)]
    assert np.median(lag1p) > 0.9
    assert main.shape == (1, 16)


def test_distractors_are_independent_of_the_state(trajectories):
    world = build_world(trajectories, CFG, "main", 0.8)
    _, xi = render(trajectories, CFG, world)
    state = trajectories.state.reshape(-1, 8)
    x = xi.reshape(-1, xi.shape[-1])
    keep = state.std(0) > 1e-9
    c = np.corrcoef(np.concatenate([state[:, keep], x], axis=1).T)[: keep.sum(), keep.sum():]
    assert np.abs(c).max() < 0.05


def test_levels_share_identical_true_states(trajectories):
    """Rendering never touches the simulated days: every level sees the same states."""
    before = trajectories.state.copy()
    for arm, d in [("main", 0.0), ("main", 0.95), ("predictable", 0.8)]:
        render(trajectories, CFG, build_world(trajectories, CFG, arm, d))
    assert np.array_equal(before, trajectories.state)


def test_observations_are_zscored_with_training_statistics(trajectories):
    world = build_world(trajectories, CFG, "main", 0.8)
    obs, _ = render(trajectories, CFG, world)
    train = np.array([s == "train" for s in trajectories.split])
    o = obs[train].reshape(-1, 128)
    assert np.abs(o.mean(0)).max() < 1e-3
    assert np.abs(o.std(0) - 1).max() < 1e-2
