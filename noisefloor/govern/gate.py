"""The governance gate every agent's actions pass through, HPA included.

The catalog uses the governed-agent-orchestrator's capability schema: an effect class (reversible,
compensable, irreversible), whether a human must approve, the compensating capability, and argument
constraints with the orchestrator's check_constraints semantics (min, max, allowed, pattern,
max_length). Loading fails for an irreversible capability that does not require approval, or a
compensable one without a known compensation. A run-level budget caps non-noop actions in a rolling
window.

`check` returns one of allow / deny / needs-approval with the reason, and is the record that goes
into the audit envelope. The vectorised `allowed_mask` in sim/fleet.py implements the same rules for
planning rollouts; a test holds the two in agreement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..sim.fleet import ACTIONS, Actuator

EFFECTS = ("reversible", "compensable", "irreversible")


def check_constraints(constraints: Dict[str, Dict[str, Any]], inputs: Dict[str, Any]) -> List[str]:
    """Same rules as the orchestrator's guard: every violated constraint, as a sentence."""
    out: List[str] = []
    for name, spec in constraints.items():
        if name not in inputs:
            continue
        value = inputs[name]
        if "min" in spec or "max" in spec:
            try:
                number = float(value)
            except (TypeError, ValueError):
                out.append(f"Input '{name}' must be a number, got {value!r}.")
                continue
            if "min" in spec and number < float(spec["min"]):
                out.append(f"Input '{name}' is {value!r}; the minimum is {spec['min']}.")
            if "max" in spec and number > float(spec["max"]) + 1e-9:
                out.append(f"Input '{name}' is {value!r}; the maximum is {spec['max']}.")
        if "allowed" in spec and value not in spec["allowed"]:
            out.append(f"Input '{name}' is {value!r}; allowed values: {', '.join(map(str, spec['allowed']))}.")
        if "pattern" in spec and not re.fullmatch(str(spec["pattern"]), str(value)):
            out.append(f"Input '{name}' is {value!r}; it must match /{spec['pattern']}/.")
        if "max_length" in spec and len(str(value)) > int(spec["max_length"]):
            out.append(f"Input '{name}' is too long; the maximum is {spec['max_length']}.")
    return out


@dataclass
class Verdict:
    action: str
    verdict: str            # allow | deny | needs-approval
    reason: str
    effect: str
    args: Dict[str, Any]


class Gate:
    def __init__(self, catalog: Dict[str, Any], source: str = "<memory>"):
        self.policy_id = catalog.get("policy_id", "gate")
        self.source = source
        self.budget = catalog.get("budget", {"actions": 10 ** 9, "window_steps": 1})
        self.caps = {c["name"]: c for c in catalog["capabilities"]}
        errors: List[str] = []
        for c in self.caps.values():
            eff = c.get("effect", "reversible")
            if eff not in EFFECTS:
                errors.append(f"{c['name']}: unknown effect {eff!r}")
            if eff == "irreversible" and not c.get("requires_approval"):
                errors.append(f"{c['name']}: irreversible capabilities must require approval")
            if eff == "compensable" and c.get("compensation") not in self.caps:
                errors.append(f"{c['name']}: compensable without a known compensation")
        if errors:
            raise ValueError("Invalid gate catalog: " + "; ".join(errors))

    @classmethod
    def load(cls, path: str) -> "Gate":
        p = Path(path)
        if not p.exists():
            p = Path(__file__).resolve().parents[2] / "configs" / path
        return cls(json.loads(p.read_text(encoding="utf-8")), source=p.name)

    @staticmethod
    def arguments(action: str, act: Actuator, i: int, cfg: Dict) -> Dict[str, Any]:
        total = int(act.r[i] + act.pending[i])
        if action == "scale_up_2":
            return {"replicas_after": total + 2}
        if action == "scale_up_6":
            return {"replicas_after": total + 6}
        if action == "scale_down_2":
            return {"replicas_after": int(act.r[i]) - 2}
        if action == "cache_warm":
            return {"steps_since_warm": int(act.since_warm[i])}
        if action == "shed_10":
            return {"shed_after": round(float(act.shed[i]) + cfg["shed_step"], 6)}
        if action == "failover":
            return {"failovers_today": int(act.failovers[i]), "lockout_steps": int(act.lockout[i])}
        return {}

    def check(self, action: str, act: Actuator, i: int, cfg: Dict, approved: bool = False) -> Verdict:
        args = self.arguments(action, act, i, cfg)
        cap = self.caps.get(action)
        if cap is None:
            return Verdict(action, "deny", f"'{action}' is not in policy {self.policy_id}", "unknown", args)
        effect = cap.get("effect", "reversible")
        if action != "noop" and int(act.recent[i].sum()) >= int(self.budget["actions"]):
            return Verdict(action, "deny", f"action budget of {self.budget['actions']} per "
                           f"{self.budget['window_steps']} steps is used up", effect, args)
        violations = check_constraints(cap.get("constraints", {}), args)
        if violations:
            return Verdict(action, "deny", " ".join(violations), effect, args)
        if cap.get("requires_approval") and not approved:
            return Verdict(action, "needs-approval", f"'{action}' is {effect} and needs a human", effect, args)
        return Verdict(action, "allow", "", effect, args)


ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
