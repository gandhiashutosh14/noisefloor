import os
import shutil

import numpy as np
import pytest
import torch

from noisefloor.config import sim_config
from noisefloor.data.collect import build_world, collect, level_key, render
from noisefloor.data.lake import load_probe_arrays, load_training_arrays, write_observations, write_trajectories
from noisefloor.models.nets import WorldModel, count_params
from noisefloor.models.train import WINDOW, TrainConfig, Windows, train

CFG = sim_config()


def test_cuda_is_hidden_and_never_initialised():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("variant,expected", [("jepa", 139_000), ("recon", 242_000)])
def test_parameter_counts_match_the_spec(variant, expected):
    n = count_params(WorldModel(variant))
    assert abs(n - expected) / expected < 0.10


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    path = tmp_path_factory.mktemp("lake")
    tr = collect(CFG, n_train=12, n_val=4, seed=5)
    v0 = write_trajectories(path / "trajectories", tr, 5)
    world = build_world(tr, CFG, "main", 0.8)
    obs, xi = render(tr, CFG, world)
    write_observations(path / "observations", level_key("main", 0.8), tr.episode_id, obs, xi)
    world2 = build_world(tr, CFG, "main", 0.0)
    obs2, xi2 = render(tr, CFG, world2)
    write_observations(path / "observations", level_key("main", 0.0), tr.episode_id, obs2, xi2)
    yield path, tr, obs, v0
    shutil.rmtree(path, ignore_errors=True)


def test_delta_round_trip_append_and_time_travel(lake):
    path, tr, obs, v0 = lake
    arr = load_training_arrays(path, "main_d80")
    train_mask = np.array([s == "train" for s in tr.split])
    assert np.allclose(arr["obs"], obs[train_mask])
    assert np.array_equal(arr["action"], tr.action[train_mask])
    assert arr["obs_version"] == 1                          # two appends: versions 0 and 1
    old = load_training_arrays(path, "main_d80", obs_version=0)
    assert np.allclose(old["obs"], obs[train_mask])        # version 0 already held this level
    assert v0 == 0


def test_training_loader_refuses_state_and_distractors(lake):
    path, *_ = lake
    for bad in ("true_state", "xi16"):
        with pytest.raises(ValueError):
            load_training_arrays(path, "main_d80", columns=("obs", bad))
    probe = load_probe_arrays(path, "main_d80", splits=("val",))
    assert probe["state"].shape[1] == 8 and probe["xi16"].shape[1] == 16


def test_windows_skip_behaviour_resets(lake):
    path, tr, *_ = lake
    arr = load_training_arrays(path, "main_d80")
    win = Windows(arr)
    for e, s in win.starts[:500]:
        assert not arr["reset"][e, s:s + WINDOW - 1].any()


@pytest.mark.parametrize("variant", ["jepa", "lambda0", "ema", "recon", "costonly"])
def test_every_variant_trains_and_stays_finite(lake, variant):
    path, *_ = lake
    out = train(load_training_arrays(path, "main_d80"), TrainConfig(variant=variant, steps=6, batch=32, log_every=3))
    assert np.isfinite(out["history"][-1]["total"])
    assert out["history"][-1]["step"] == 6
