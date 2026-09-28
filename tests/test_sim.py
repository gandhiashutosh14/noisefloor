import numpy as np
import pytest

from noisefloor.config import sim_config
from noisefloor.sim.fleet import DOWN2, FAILOVER, NOOP, UP2, UP6, Fleet, allowed_mask, new_actuator, physics, simulate_open_loop
from noisefloor.sim.loads import make_profile

CFG = sim_config()


def _fleet(n=4, r0=10, family="diurnal", seed=0):
    profiles = [make_profile(family, np.random.default_rng([seed, i]), CFG) for i in range(n)]
    return Fleet.start(profiles, np.full(n, r0), CFG)


def test_queue_is_never_negative_and_replicas_stay_in_bounds():
    rng = np.random.default_rng(0)
    f = _fleet(8, r0=4, family="flash")
    while not f.done:
        m = allowed_mask(f.act, CFG)
        a = np.array([rng.choice(np.flatnonzero(row)) for row in m])
        f.step(a)
        assert (f.queue >= 0).all()
        assert (f.act.r >= CFG["replicas_min"]).all()
        assert (f.act.r + f.act.pending <= CFG["replicas_max"]).all()


def test_cost_rises_with_replicas_when_nothing_is_violated():
    load = np.full(3, 100.0)
    ph = [physics(load, np.full(3, r), np.full(3, 1.0), np.ones(3), np.zeros(3), np.zeros(3), CFG) for r in (12, 20, 30)]
    assert all(p["p95_ms"].max() < CFG["slo_latency_ms"] for p in ph)
    act = [new_actuator(np.full(3, r), CFG) for r in (12, 20, 30)]
    costs = [simulate_open_loop(np.zeros(3), np.ones(3), np.zeros(3, bool), a, np.full((3, 4), 100.0),
                                np.ones((3, 4)), np.zeros((3, 4), np.int64), CFG).mean() for a in act]
    assert costs[0] < costs[1] < costs[2]


def test_scale_up_takes_two_steps_and_scale_down_is_immediate():
    """A scale-up chosen at step t serves traffic from step t + 2; a scale-down removes capacity at once."""
    f = _fleet(2, r0=10)
    rho0 = f.last["rho"].copy()
    f.step(np.array([UP2, DOWN2]))                    # step t: capacity 10 and 8
    assert list(f.act.r) == [10, 8] and list(f.act.pending) == [2, 0]
    f.step(np.array([NOOP, NOOP]))                    # step t + 1: still 10 ready replicas
    assert list(f.act.r) == [12, 8] and list(f.act.pending) == [0, 0]   # ready for step t + 2
    assert rho0.shape == (2,)


def test_failover_restores_health_and_resets_cache():
    profiles = [make_profile("incident", np.random.default_rng([3, i]), CFG) for i in range(2)]
    f = Fleet.start(profiles, np.full(2, 10), CFG)
    start = profiles[0].incident_start
    while f.t < start + 1:
        f.step(np.zeros(2, np.int64))
    assert f.last["health"][0] == pytest.approx(CFG["incident_health"])
    f.step(np.array([FAILOVER, NOOP]))
    assert f.last["health"][0] == 1.0
    assert f.cache[0] == pytest.approx(CFG["cache_after_failover"] * CFG["cache_decay"])
    assert not allowed_mask(f.act, CFG)[0, FAILOVER]         # lockout and the one-per-day limit


def test_disallowed_actions_are_rejected_by_the_fleet():
    f = _fleet(1, r0=38)
    with pytest.raises(ValueError):
        f.step(np.array([UP6]))


def test_simulation_is_deterministic_in_its_seeds():
    runs = []
    for _ in range(2):
        f = _fleet(4, family="flash", seed=11)
        J = 0.0
        while not f.done:
            J += f.step(np.full(4, NOOP))["cost"].sum()
        runs.append(J)
    assert runs[0] == runs[1]
