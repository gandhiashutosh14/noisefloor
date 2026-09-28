"""Audit a few episodes end to end: envelopes -> log -> ledger -> replay under a changed policy.

1. A trained agent runs evaluation days; every decision becomes a TRACEWAKE TraceEnvelope.
2. The envelopes are published to a bus (in memory, or a Kafka-compatible broker such as AutoMQ),
   twice, to show that the ledger is idempotent on (run_id, seq).
3. A WakeLedger is rebuilt from the log alone.
4. Every recorded decision is replayed against a stricter catalog (gate-v2: at most 24 replicas, no
   failover) using the arguments recorded at decision time, and the decisions that would flip are
   listed. The action budget depends on the whole run's history and is not replayed.

Requires the audit extra (tracewake).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..govern.gate import Gate, check_constraints


def replay_verdict(gate: Gate, action: str, args: Dict, approved: bool = False) -> Dict[str, str]:
    """The verdict a catalog gives to a recorded decision, from its recorded arguments."""
    cap = gate.caps.get(action)
    if cap is None:
        return {"verdict": "deny", "reason": f"'{action}' is not in policy {gate.policy_id}"}
    violations = check_constraints(cap.get("constraints", {}), args or {})
    if violations:
        return {"verdict": "deny", "reason": " ".join(violations)}
    if cap.get("requires_approval") and not approved:
        return {"verdict": "needs-approval", "reason": f"'{action}' needs a human"}
    return {"verdict": "allow", "reason": ""}


def replay_ledger(ledger, gate_new: Gate) -> Dict:
    flips: List[Dict] = []
    total = executed = 0
    for rid in ledger.runs():
        for env in ledger.envelopes(rid):
            total += 1
            d = env.data
            action = d.get("executed") or "noop"
            if action == "noop":
                continue
            executed += 1
            approved = d.get("approval") == "granted"
            new = replay_verdict(gate_new, action, d.get("args") or {}, approved=approved)
            if new["verdict"] != "allow":
                flips.append({"run_id": rid, "seq": env.seq, "action": action, "args": d.get("args"),
                              "old": "allow", "new": new["verdict"], "reason": new["reason"]})
    by_action: Dict[str, int] = {}
    for f in flips:
        by_action[f["action"]] = by_action.get(f["action"], 0) + 1
    return {"decisions": total, "executed_non_noop": executed, "flips": len(flips), "flips_by_action": by_action,
            "examples": flips[:12]}


def run_audit(recorder, bus, out_md: Path, out_json: Optional[Path] = None, topic: str = "noisefloor.decisions",
              policy_new: str = "gate_v2.json", describe: str = "in-memory") -> Dict:
    from tracewake.ledger import WakeLedger
    n1 = recorder.publish(bus, topic)
    n2 = recorder.publish(bus, topic)
    ledger = WakeLedger()
    stats = ledger.ingest(bus, topic)
    gaps = {rid: ledger.gaps(rid) for rid in ledger.runs()}
    rep = replay_ledger(ledger, Gate.load(policy_new))
    result = {"bus": describe, "published": n1 + n2, "ingest": stats.to_dict(), "runs": len(ledger.runs()),
              "events": ledger.count(), "gaps": {k: v for k, v in gaps.items() if v},
              "idempotent": stats.inserted == n1 and stats.duplicates == n2,
              "replay_policy": Gate.load(policy_new).policy_id, **rep}
    lines = [f"# Audit: {result['runs']} episodes through TRACEWAKE ({describe})", "",
             f"- envelopes published: {n1}, then the same {n2} again",
             f"- ledger rebuilt from the log: {stats.inserted} inserted, {stats.duplicates} duplicates ignored, "
             f"{stats.invalid} invalid; idempotent: **{result['idempotent']}**; sequence gaps: {len(result['gaps'])}",
             f"- replayed {rep['executed_non_noop']} executed actions under `{result['replay_policy']}` "
             f"(max 24 replicas, no failover): **{rep['flips']} would flip** {rep['flips_by_action']}", "",
             "| run | seq | action | recorded args | under gate-v2 | reason |", "|---|---|---|---|---|---|"]
    for f in rep["examples"]:
        lines.append(f"| {f['run_id']} | {f['seq']} | {f['action']} | `{json.dumps(f['args'])}` | {f['new']} | {f['reason']} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out_json:
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
