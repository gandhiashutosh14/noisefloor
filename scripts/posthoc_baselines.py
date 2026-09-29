"""Post-hoc re-run of the rule baselines and the oracle after the review of 2026-09-29 (disclosed in
RESULTS.md as a post-registration analysis; the pre-registered numbers are not changed).

Two corrections are applied to the reference agents only; the learned agents' published day-by-day
costs are taken from reports/results.json and are not re-run:
  (a) HPA, runbook and predictive size demand from the replica count the utilisation was measured on
      (the v0.1 code multiplied a stale utilisation by the current count and over-scaled on 6.6% of
      decisions);
  (b) additionally, the approver requires degraded region health before approving a failover, so a
      flash crowd no longer triggers a useless failover (the runbook's rule and the approver's are both
      error-based in the registered environment).
The oracle is re-run under (b) as well, with the v0.2 planner (ranking on effective first actions).

    python scripts/posthoc_baselines.py [--out reports/posthoc-baselines.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import noisefloor  # noqa: E402,F401
from noisefloor.audit.envelopes import git_sha  # noqa: E402
from noisefloor.config import sim_config  # noqa: E402
from noisefloor.eval.run import baselines, eval_set, oracle, run_agent, tune_hpa  # noqa: E402
from noisefloor.eval.stats import hierarchical_bootstrap  # noqa: E402
from noisefloor.govern.gate import Gate  # noqa: E402


def gap(J: np.ndarray, Jrb: np.ndarray, Jor: np.ndarray) -> float:
    return float((Jrb.mean() - J.mean()) / (Jrb.mean() - Jor.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "posthoc-baselines.json"))
    a = ap.parse_args()
    cfg, gate = sim_config(), Gate.load("gate_v1.json")
    published = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    pub = {r["model"]: r for r in published["table_d80"]}
    _, ids = eval_set(cfg)
    out = {"git_sha": git_sha(), "published_commit": published["git_sha"], "episodes": ids, "runs": {}}
    tuned = {"hpa": tune_hpa(cfg, gate), "runbook": tune_hpa(cfg, gate, runbook=True)}
    out["tuning"] = tuned
    agents = baselines(cfg, gate, tuned)
    for setting, health in (("count_fix", False), ("count_fix_and_health_aware_approver", True)):
        runs = {}
        for name, agent in list(agents.items()) + [("oracle", oracle(cfg))]:
            if name == "oracle" and not health:
                continue                                     # the registered oracle numbers stand under (a)
            res = run_agent(agent, cfg, gate, approver_health=health)
            J = np.array([r.J for r in res.results])
            runs[name] = {"J": [float(x) for x in J], "J_mean": float(J.mean()), "J_ci": hierarchical_bootstrap(J[None, :]),
                          "failovers": int(sum(r.failovers for r in res.results)),
                          "failovers_by_family": {f: int(sum(r.failovers for r in res.results if r.family == f)) for f in cfg["families"]},
                          "violation_min_per_day": float(np.mean([5 * r.violation_steps for r in res.results])),
                          "replica_hours_per_day": float(np.mean([r.replica_steps * 5 / 60 for r in res.results]))}
            print(f"{setting:40s} {name:10s} J {J.mean():7.1f}  failovers {runs[name]['failovers']}", flush=True)
        out["runs"][setting] = runs
    # corrected "gap closed" for the published learned agents (their day-level J is not in results.json,
    # so the point estimate uses their published mean; intervals are not recomputed)
    Jor_pub = pub["oracle"]["J"]["mean"]
    for setting, runs in out["runs"].items():
        Jrb = runs["runbook"]["J_mean"]
        Jor = runs["oracle"]["J_mean"] if "oracle" in runs else Jor_pub
        runs["gap_closed_published_models"] = {m: float((Jrb - pub[m]["J"]["mean"]) / (Jrb - Jor))
                                               for m in ("jepa", "recon", "ema", "costonly", "lambda0") if m in pub}
        runs["reference"] = {"runbook": Jrb, "oracle": Jor, "oracle_source": "re-run" if "oracle" in runs else "published"}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({s: {"runbook": r["runbook"]["J_mean"], "oracle": r["reference"]["oracle"], "gap": r["gap_closed_published_models"]}
                      for s, r in out["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
