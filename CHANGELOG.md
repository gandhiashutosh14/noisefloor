# Changelog

## Unreleased (2026-10-02)

The audit-envelope dependency is now STERNWATCH (`pip install -e .[audit]` installs v0.4.0-alpha; the
project was renamed and no behaviour changed). The AutoMQ audit report is regenerated with this code.

## 0.2.0a1 (main, 2026-09-30; not a new experiment)

Fixes from the code review of 2026-09-29. The pre-registered results (tag `prereg-v1`, release v0.1.0) were
produced by the code at the tag and are not changed; RESULTS.md discloses the corrections and the post-hoc
baseline re-run. Running the grid on this code would be a new, v0.2 experiment.

Behaviour changes
- HPA, runbook and predictive size demand from the replica count the utilisation was measured on
  (`golden["replicas"]`), as Kubernetes does; the v0.1 code multiplied a stale utilisation by the current count.
- `cem_plan` ranks and de-duplicates on the effective first action (`first_mask`), so the gate is handed actions
  it will accept; the learned agents and the oracle pass the current mask.
- The gate's catalog is the single source of the planning mask (`Gate.limits`, `Gate.mask`); `allowed_mask`
  takes those limits, the fleet enforces only physical limits, and the actuator's budget window comes from the
  catalog. A catalog can tighten the simulator's limits, never loosen them.
- The approver can require degraded region health (`Approver(require_degraded_health=True)`); the registered
  approver stays the default.
- Training windows exclude the frame right after a behaviour-policy replica reset (its telemetry was measured
  on the old count).
- The first frame of a day is a zero-length measurement (no phantom backlog); the main and isotropic arms
  use their analytic unit source variance for calibration. These change the generated data's hash; the v0.1
  data is reproducible from the tag.

Records and metrics
- Decision records keep every candidate's verdict and whether a request was filed; an approved failover the
  gate then refuses is recorded and counted (`refused_approvals`), and the harness exposes pending requests
  and the gate's limits to agents (`Context.pending`, `Context.limits`).
- `shed_frac` reports the shed level that was billed; `churn` counts scale reversals within three steps;
  `plan_ms` is documented as batch time per environment; probes record the latent's scale (`z_std`).
- STERNWATCH run ids include the latent width; envelopes carry the gate's policy id and the candidate list; the
  audit replay applies the new catalog's action budget over the run's own history.
- MLflow decision traces (spans) are written for one evaluation day in eight, as the README always said.

Report and infrastructure
- `report.py` applies the registered direction of each secondary test and the Holm adjustment to the verdicts,
  refuses duplicate rows, prints exploratory cells, planning latency (from the tables) and the planner-optimism
  flag, and stamps UTC times.
- `aggregate` validates data hash, commit and schemas of every shard before writing and refuses an existing
  output directory; `grid.yml` requires HEAD to be the pre-registration tag's commit.
- Observation levels are written with a partition overwrite and both tables carry the data hash, which the
  loaders compare; rebuilding a lake in place is safe.
- `scripts/posthoc_baselines.py` re-runs the reference agents with the corrections; `docs/DESIGN_SPEC.md`
  commits the pre-build spec; stale docstrings and the MLflow `K` parameter corrected. 56 tests.

## 0.1.0 (2026-09-29)

Pre-registered build and results (tag `prereg-v1`, commit c90c402; grid run 36489642311).
