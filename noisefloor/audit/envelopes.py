"""Every decision the agent makes becomes a TRACEWAKE TraceEnvelope.

    run_id   "{model}-{variant}-s{seed}-{arm}-d{d}-e{episode}"
    seq      t + 1 (TRACEWAKE sequence numbers start at 1)
    type     "decision"
    producer "noisefloor/agent@{git sha}"
    policy   "gate-v1" (the catalog the gate enforced)
    data     proposed, executed, effect_class, args, verdict, reason, approval, predicted_cost,
             elite_spread, mlflow_run_id, delta_version

With the ``audit`` extra installed (``pip install -e .[audit]``) the envelopes are real
tracewake.envelope.TraceEnvelope objects, published to a TRACEWAKE bus and ingested by a WakeLedger,
which is idempotent on (run_id, seq). Without it, ``envelope_dict`` produces the same JSON shape.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # optional dependency
    from tracewake.envelope import TraceEnvelope  # type: ignore
    HAVE_TRACEWAKE = True
except Exception:  # pragma: no cover - exercised when the extra is absent
    TraceEnvelope = None
    HAVE_TRACEWAKE = False

ROOT = Path(__file__).resolve().parent.parent.parent
EVENT_TYPE = "decision"
EPOCH = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def run_id(model: str, variant: str, seed: int, arm: str, d: float, episode: int) -> str:
    return f"{model}-{variant}-s{seed}-{arm}-d{int(round(d * 100)):02d}-e{episode}"


def sim_timestamp(episode: int, t: int, step_seconds: int = 300) -> str:
    """Simulated wall clock: each episode is its own day after EPOCH, each step five minutes."""
    ts = EPOCH + _dt.timedelta(days=episode % 3650, seconds=t * step_seconds)
    return ts.isoformat().replace("+00:00", "Z")


def envelope_dict(rid: str, t: int, episode: int, decision: Dict[str, Any], *, sha: str, policy_id: str = "gate-v1",
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = {"proposed": decision.get("proposed"), "executed": decision.get("executed"),
            "effect_class": decision.get("effect"), "args": decision.get("args") or {},
            "verdict": decision.get("verdict"), "reason": decision.get("reason") or "",
            "approval": decision.get("approval")}
    data.update(extra or {})
    return {"run_id": rid, "seq": t + 1, "ts": sim_timestamp(episode, t), "type": EVENT_TYPE, "data": data,
            "policy_id": policy_id, "producer": f"noisefloor/agent@{sha}", "envelope_version": "1"}


def to_envelope(d: Dict[str, Any]):
    if not HAVE_TRACEWAKE:
        raise RuntimeError("tracewake is not installed; pip install -e .[audit]")
    return TraceEnvelope(run_id=d["run_id"], seq=d["seq"], ts=d["ts"], type=d["type"], data=d["data"],
                         policy_id=d["policy_id"], producer=d["producer"])


class EnvelopeRecorder:
    """Collects envelope dicts from the harness's ``record(t, i, decision)`` callback."""

    def __init__(self, model: str, variant: str, seed: int, arm: str, d: float, episode_ids: List[int],
                 agent=None, sha: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        self.ids = [run_id(model, variant, seed, arm, d, e) for e in episode_ids]
        self.episode_ids = episode_ids
        self.agent, self.sha, self.extra = agent, sha or git_sha(), extra or {}
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, t: int, i: int, decision: Dict[str, Any]) -> None:
        extra = dict(self.extra)
        if self.agent is not None and getattr(self.agent, "last_pred_cost", None) is not None:
            extra["predicted_cost"] = round(float(self.agent.last_pred_cost[i]), 4)
            extra["elite_spread"] = round(float(self.agent.last_spread[i]), 4)
        self.rows.append(envelope_dict(self.ids[i], t, self.episode_ids[i], decision, sha=self.sha, extra=extra))

    def publish(self, bus, topic: str = "noisefloor.decisions") -> int:
        """Publish every envelope to a TRACEWAKE bus (MemoryBus or the Kafka/AutoMQ bus)."""
        bus.ensure_topic(topic)
        for d in self.rows:
            env = to_envelope(d)
            bus.publish(topic, env.key(), env.to_json())
        bus.flush()
        return len(self.rows)
