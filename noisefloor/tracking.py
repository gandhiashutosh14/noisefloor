"""MLflow tracking, one SQLite store per process.

Each process writes to ``mlruns/{job}-{proc}.db`` so parallel CI jobs never share a SQLite file
(MLflow issue #6013, "database is locked"). Stores are kept as CI artifacts and never merged; the
aggregate report reads Delta tables only. Tracking is optional: with NOISEFLOOR_MLFLOW=0 (or mlflow
missing) every call becomes a no-op, so tests and quick runs need no store.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("PYTHONUTF8", "1")

try:
    import mlflow  # type: ignore
    HAVE_MLFLOW = True
except Exception:  # pragma: no cover
    mlflow = None
    HAVE_MLFLOW = False

ROOT = Path(__file__).resolve().parent.parent


def enabled() -> bool:
    return HAVE_MLFLOW and os.environ.get("NOISEFLOOR_MLFLOW", "1") != "0"


def configure(job: str, proc: int = 0, root: Optional[Path] = None) -> Optional[str]:
    if not enabled():
        return None
    store = (root or ROOT) / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (store / f"{job}-{proc}.db").as_posix()
    mlflow.set_tracking_uri(uri)
    return uri


@contextlib.contextmanager
def run(experiment: str, name: str, params: Dict[str, Any], tags: Optional[Dict[str, str]] = None) -> Iterator[Optional[str]]:
    if not enabled():
        yield None
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name, tags=tags or {}) as r:
        mlflow.log_params({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()})
        yield r.info.run_id


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if enabled() and mlflow.active_run() is not None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_dict(d: Dict, name: str) -> None:
    if enabled() and mlflow.active_run() is not None:
        mlflow.log_dict(d, name)


def log_artifact(path: Path) -> None:
    if enabled() and mlflow.active_run() is not None:
        mlflow.log_artifact(str(path))


def log_encoder(model, name: str = "encoder") -> None:
    """Log the encoder as an MLflow PyTorch model with an input example (128 telemetry channels)."""
    if not (enabled() and mlflow.active_run() is not None):
        return
    import numpy as np
    from mlflow import pytorch as mlflow_pytorch  # type: ignore
    mlflow_pytorch.log_model(model.encoder, name=name, input_example=np.zeros((2, 128), np.float32))


class DecisionTracer:
    """MLflow tracing for sampled evaluation episodes: one AGENT span per decision with child spans for
    encode, cem, gate and act. Only episodes listed in ``episodes`` are traced (1 in 8 by default)."""

    def __init__(self, episodes, run_label: str):
        self.episodes = set(episodes)
        self.label = run_label
        self.active = enabled() and bool(self.episodes)

    def record(self, t: int, i: int, episode: int, decision: Dict[str, Any], plan_ms: float,
               predicted_cost: Optional[float], spread: Optional[float]) -> None:
        if not self.active or episode not in self.episodes:
            return
        with mlflow.start_span(name="decision", span_type="AGENT") as span:
            span.set_inputs({"run": self.label, "episode": episode, "t": t})
            with mlflow.start_span(name="encode", span_type="CHAIN") as s:
                s.set_outputs({"frames": 1})
            with mlflow.start_span(name="cem", span_type="CHAIN") as s:
                s.set_outputs({"proposed": decision.get("proposed"), "predicted_cost": predicted_cost,
                               "elite_spread": spread, "plan_ms": plan_ms})
            with mlflow.start_span(name="gate", span_type="GUARDRAIL") as s:
                s.set_outputs({"verdict": decision.get("verdict"), "reason": decision.get("reason"),
                               "effect": decision.get("effect")})
            with mlflow.start_span(name="act", span_type="TOOL") as s:
                s.set_outputs({"executed": decision.get("executed")})
            span.set_outputs({"executed": decision.get("executed")})

    def flush(self) -> None:
        if self.active:
            with contextlib.suppress(Exception):
                mlflow.flush_trace_async_logging()
