"""Delta Lake storage for trajectories, observations, evaluation steps and results (delta-rs, no JVM).

Tables (all local directories; paths must not contain spaces, see delta-rs issue #2425):
  trajectories   one row per (episode, t): true state, actuator, action, cost, reset flag
  observations   partitioned by level: one row per (episode, t) with the 128-channel telemetry
  results        one row per (run, episode): the evaluation outcome
  eval_steps     one row per decision (optional, large)

Every training run records the Delta version it read, and ``DeltaTable(path, version=v)`` reproduces
exactly that data later. The training loader reads only an allow-list of columns, so the true state
and the distractor values cannot leak into a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .collect import Trajectories

MODEL_COLUMNS = frozenset({"episode_id", "t", "obs", "actuator", "action", "cost", "violation", "reset", "split"})


def _list_col(x: np.ndarray) -> pa.Array:
    flat = pa.array(x.reshape(-1).astype(np.float32))
    return pa.FixedSizeListArray.from_arrays(flat, x.shape[-1])


def write_trajectories(path: Path, tr: Trajectories, data_seed: int) -> int:
    N, T = tr.action.shape
    tbl = pa.table({
        "episode_id": np.repeat(tr.episode_id, T).astype(np.int32),
        "split": np.repeat(np.array(tr.split), T),
        "family": np.repeat(np.array(tr.family), T),
        "t": np.tile(np.arange(T), N).astype(np.int16),
        "true_state": _list_col(tr.state.reshape(N * T, -1)),
        "rho": tr.rho.reshape(-1).astype(np.float32),
        "actuator": _list_col(tr.act_vec.reshape(N * T, -1)),
        "action": tr.action.reshape(-1).astype(np.int8),
        "cost": tr.cost.reshape(-1).astype(np.float32),
        "reset": tr.reset.reshape(-1),
        "violation": tr.violation.reshape(-1),
        "data_seed": np.full(N * T, data_seed, np.int64),
    })
    write_deltalake(str(path), tbl, mode="overwrite")
    return DeltaTable(str(path)).version()


def write_observations(path: Path, level: str, episode_id: np.ndarray, obs: np.ndarray, xi16: np.ndarray) -> int:
    N, T, _ = obs.shape
    tbl = pa.table({
        "level": np.full(N * T, level),
        "episode_id": np.repeat(episode_id, T).astype(np.int32),
        "t": np.tile(np.arange(T), N).astype(np.int16),
        "obs": _list_col(obs.reshape(N * T, -1)),
        "xi16": _list_col(xi16.reshape(N * T, -1)),
    })
    write_deltalake(str(path), tbl, mode="append", partition_by=["level"])
    return DeltaTable(str(path)).version()


def _to_array(col: pa.ChunkedArray, width: int) -> np.ndarray:
    arr = col.combine_chunks()
    return np.asarray(arr.flatten() if hasattr(arr, "flatten") else arr.values).reshape(-1, width)


def load_training_arrays(lake: Path, level: str, splits: Sequence[str] = ("train",), traj_version: Optional[int] = None,
                         obs_version: Optional[int] = None,
                         columns: Sequence[str] = ("obs", "actuator", "action", "cost", "violation", "reset")) -> Dict:
    """Arrays shaped (episodes, T, ...) for the requested splits; refuses anything outside MODEL_COLUMNS."""
    bad = set(columns) - MODEL_COLUMNS
    if bad:
        raise ValueError(f"columns not allowed in a training loader: {sorted(bad)}")
    traj = DeltaTable(str(lake / "trajectories"), version=traj_version)
    obs_t = DeltaTable(str(lake / "observations"), version=obs_version)
    tcols = ["episode_id", "t", "split"] + [c for c in ("actuator", "action", "cost", "violation", "reset") if c in columns]
    tt = traj.to_pyarrow_table(columns=tcols)
    split_mask = np.isin(np.asarray(tt.column("split")), list(splits))
    ep = np.asarray(tt.column("episode_id"))[split_mask]
    tsteps = np.asarray(tt.column("t"))[split_mask]
    order = np.lexsort((tsteps, ep))
    eps = np.unique(ep)
    T = int(tsteps.max()) + 1
    out: Dict = {"episode_id": eps, "T": T,
                 "traj_version": traj.version(), "obs_version": obs_t.version()}
    if "actuator" in columns:
        out["actuator"] = _to_array(tt.column("actuator"), 6)[split_mask][order].reshape(len(eps), T, 6)
    for c in ("action", "cost", "violation", "reset"):
        if c in columns:
            out[c] = np.asarray(tt.column(c))[split_mask][order].reshape(len(eps), T)
    if "obs" in columns:
        ot = obs_t.to_pyarrow_table(columns=["episode_id", "t", "obs"], filters=[("level", "=", level)])
        oep = np.asarray(ot.column("episode_id"))
        keep = np.isin(oep, eps)
        ots = np.asarray(ot.column("t"))[keep]
        oep = oep[keep]
        o = _to_array(ot.column("obs"), 128)[keep][np.lexsort((ots, oep))]
        out["obs"] = o.reshape(len(eps), T, 128)
    return out


def load_probe_arrays(lake: Path, level: str, splits: Sequence[str] = ("val",)) -> Dict:
    """Obs with the true state and distractor values, for probes and diagnostics only (never training)."""
    traj = DeltaTable(str(lake / "trajectories")).to_pyarrow_table(columns=["episode_id", "t", "split", "true_state"])
    keep = np.isin(np.asarray(traj.column("split")), list(splits))
    ep, ts = np.asarray(traj.column("episode_id"))[keep], np.asarray(traj.column("t"))[keep]
    order = np.lexsort((ts, ep))
    state = _to_array(traj.column("true_state"), 8)[keep][order]
    ot = DeltaTable(str(lake / "observations")).to_pyarrow_table(columns=["episode_id", "t", "obs", "xi16"],
                                                                  filters=[("level", "=", level)])
    oep, ots = np.asarray(ot.column("episode_id")), np.asarray(ot.column("t"))
    k2 = np.isin(oep, np.unique(ep))
    o2 = np.lexsort((ots[k2], oep[k2]))
    return {"state": state, "obs": _to_array(ot.column("obs"), 128)[k2][o2], "xi16": _to_array(ot.column("xi16"), 16)[k2][o2]}


def append_rows(path: Path, rows: List[Dict], partition_by: Optional[List[str]] = None) -> int:
    if not rows:
        return -1
    tbl = pa.Table.from_pylist(rows)
    write_deltalake(str(path), tbl, mode="append", partition_by=partition_by, schema_mode="merge")
    return DeltaTable(str(path)).version()
