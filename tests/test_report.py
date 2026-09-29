import numpy as np
import pandas as pd
import pytest

from noisefloor.eval.report import DIRECTIONS, gap_closed, matrix, verdict

W_LOWER = {"win": True, "loss": False}
W_HIGHER = {"win": False, "loss": True}
W_NONE = {"win": False, "loss": False}


def test_verdict_follows_the_registered_direction_and_the_holm_adjustment():
    assert verdict(W_LOWER, -1, 0.01) == "supported"
    assert verdict(W_HIGHER, -1, 0.01) == "reversed"
    assert verdict(W_HIGHER, +1, 0.01) == "supported"          # a positive difference-in-differences was the prediction
    assert verdict(W_LOWER, +1, 0.01) == "reversed"
    assert verdict(W_LOWER, -1, 0.2) == "not supported (Holm-adjusted p >= 0.05)"
    assert verdict(W_NONE, -1, 0.01) == "not supported (interval includes 0)"
    assert verdict(W_HIGHER, 0, 0.01) == "reference lower (no direction was registered)"
    assert verdict(W_LOWER, 0, 0.01) == "candidate lower (no direction was registered)"
    assert verdict(W_LOWER, -1) == "supported"                 # primary tests carry no Holm adjustment


def test_registered_directions_match_the_preregistration():
    assert DIRECTIONS["jepa_vs_ema_d80"] == 0 and DIRECTIONS["did_predictable_minus_main"] == +1
    assert all(v == -1 for k, v in DIRECTIONS.items() if k not in ("jepa_vs_ema_d80", "did_predictable_minus_main"))


def test_matrix_refuses_duplicate_rows_and_gap_closed_checks_pairing():
    rows = [{"model": "jepa", "arm": "main", "d": 0.8, "variant": "main", "z_dim": 16, "train_seed": s, "episode_id": e, "J": 1.0}
            for s in range(2) for e in range(3)]
    res = pd.DataFrame(rows)
    assert matrix(res, "jepa", "main", 0.8).shape == (2, 3)
    with pytest.raises(ValueError, match="duplicate"):
        matrix(pd.concat([res, res.iloc[:1]]), "jepa", "main", 0.8)
    with pytest.raises(ValueError, match="pairs"):
        gap_closed(np.ones((2, 3)), np.ones(4), np.ones(3), n_boot=10)
