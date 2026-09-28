"""Statistics for paired comparisons across training seeds and evaluation days.

Data layout: a (seeds x episodes) matrix of per-episode differences (or values) where every column is
the same evaluation day for every seed (common random numbers).

  hierarchical_bootstrap  resample seeds, then episodes within the resampled set (10,000 draws);
                          95% percentile interval of the mean
  seed_t_interval         t interval on per-seed means (df = seeds - 1); a win needs both intervals
                          to exclude zero
  tost                    two one-sided tests for equivalence within +/- margin (90% interval inside)
  holm                    Holm-Bonferroni step-down adjustment
  iqm                     interquartile mean, reported beside the mean
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def hierarchical_bootstrap(x: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    S, E = x.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, S, (n_boot, S))
    ei = rng.integers(0, E, (n_boot, E))
    means = x[si[:, :, None], ei[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}


def seed_t_interval(x: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    m = np.atleast_2d(np.asarray(x, dtype=np.float64)).mean(axis=1)
    n = len(m)
    if n < 2:
        return {"mean": float(m.mean()), "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    se = m.std(ddof=1) / np.sqrt(n)
    t = sps.t.ppf(1 - alpha / 2, n - 1)
    p = float(2 * sps.t.sf(abs(m.mean() / se), n - 1)) if se > 0 else 0.0
    return {"mean": float(m.mean()), "lo": float(m.mean() - t * se), "hi": float(m.mean() + t * se), "p_two_sided": p}


def win(diff: np.ndarray, **kw) -> Dict:
    """diff = candidate - reference (lower J is better). A win needs both intervals below zero."""
    b, t = hierarchical_bootstrap(diff, **kw), seed_t_interval(diff)
    return {"bootstrap": b, "seed_t": t, "win": bool(b["hi"] < 0 and t["hi"] < 0),
            "loss": bool(b["lo"] > 0 and t["lo"] > 0)}


def tost(rel_diff: np.ndarray, margin: float = 0.03, n_boot: int = 10_000, seed: int = 0) -> Dict:
    """Equivalence of a relative difference within +/- margin: the 90% bootstrap interval must lie inside."""
    b = hierarchical_bootstrap(rel_diff, n_boot=n_boot, seed=seed, alpha=0.10)
    return {"interval90": [b["lo"], b["hi"]], "margin": margin, "mean": b["mean"],
            "equivalent": bool(-margin < b["lo"] and b["hi"] < margin)}


def holm(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def iqm(x: np.ndarray) -> float:
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = len(v)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())


def gap_closed(J: np.ndarray, J_runbook: np.ndarray, J_oracle: np.ndarray) -> np.ndarray:
    """Per-episode share of the runbook-to-oracle gap closed; aggregated as ratio of means by callers."""
    return (J_runbook - J) / (J_runbook - J_oracle)
