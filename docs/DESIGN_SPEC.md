# NOISEFLOOR design specification (v1.0, written before the build)

This is the document PREREGISTRATION.md and the scripts call "the spec". It was written on 2026-09-28 before any code existed and is committed unchanged apart from this note and the removal of local paths. The pilots and checks in PREREGISTRATION.md section 6 changed some of these settings; PREREGISTRATION.md, not this file, is the registered design. Section 14 sets the rules the pilots followed (the headroom gate and the one permitted change after a flat pilot).

**Pitch:** a governed SRE autoscaling agent that plans inside a JEPA latent world model (LeJEPA SIGReg; no decoder, no EMA teacher), trained CPU-only, with a pre-registered test of whether it still plans well when 80-95% of telemetry variance is unpredictable noise.

**Rules.** CPU only: `noisefloor/__init__.py` sets `CUDA_VISIBLE_DEVICES=""` before importing torch 2.14.0+cpu; `torch.set_num_threads(2)`; a test fails if CUDA initialises. No LLM anywhere. Repository path without spaces (delta-rs #2425: paths with spaces break reads). Python 3.11.

## 1. Simulator (`sim/`, vectorised numpy)

5-min steps, 288-step episodes (one day), μ=20 req/s per replica (S=50 ms), λ̄=150 req/s. True state s=(λ, q, r, p, c, h, ℓ95, e): load, queue, ready/pending replicas, cache warmth, region health, p95 latency, error rate. Actuator state u (r, p, shed, cooldowns, lockout, budget used) is known exactly to every controller.

- **Load** λ_t=λ̄·b_t·(1+Σ_j A_j e^{−(t−t_j)/6}·1[t≥t_j])·(1+0.05η_t), b_t=1+0.5 sin(2πt/288−π/2). Families (equal mix): diurnal; flash crowd (Poisson 3/day, A_j~U(1,2)); ramp (×(1+0.8t/288)); incident (h=0.4 from t~U[48,240] until failover or 36 steps). OOD (never trained): 24-step ×2.5 launch plateau; double spike.
- **Capacity** κ=r·μ·(0.7+0.3c)·h; λˢ=λ(1−s) after shedding; ρ=λˢ/κ.
- **Queue** q_{t+1}=max(0, q_t+300(λˢ−κ)).
- **Latency** ℓ95=S_eff+3W_q+q/κ, W_q=ρ̃^{√(2(r+1))}/((1−ρ̃)λˢ) (Sakasegawa + Little; ×3 = exponential p95/mean), ρ̃=min(ρ,0.98), S_eff=1/(μ(0.7+0.3c)h). Check: r=10, ρ=0.9 → W_q≈34 ms, matching Erlang C.
- **Errors** e=0.05(1−h)+0.2σ(40(ρ−1.05)).
- **Replicas** ready 2 steps after scale-up; scale-down immediate; 2≤r≤40. **Cache** c←0.97c; cache_warm c←c+0.5(1−c); failover h←1, c←0.2.
- **Step cost** c_t=0.05r+1.0·1[ℓ95>250 ms ∨ e>1%]+2.0s+0.1·1[warm]+20·1[failover]; J=Σc_t. A violation-step equals 20 replica-steps, and a static-peak baseline checks that over-provisioning isn't optimal. `prices.yaml` gives a secondary $/day lens.

|Action|Effect class|Approval|Gate constraint|
|---|---|---|---|
|noop|none|no|–|
|scale_up_2, scale_up_6|reversible|no|r+p+Δ≤40|
|scale_down_2|reversible|no|r−2≥2|
|cache_warm|reversible|no|≤1 per 6 steps|
|shed_10|compensable (auto restore_traffic after 3 steps)|no|total shed ≤0.3|
|failover|irreversible|**yes**|12-step lockout, ≤1/day|

Budget: ≤6 non-noop actions per rolling 12 steps. The scripted approver grants failover only if clean errors have been >2% for 2 steps, with 1-step latency. Meanwhile the agent runs its best reversible candidate.

## 2. Observations and distractors

u_t=Aφ(s_t)+β_d Qξ_t; o_t=softplus(u_t)+0.05ε_t, z-scored with training stats.
- φ∈ℝ¹²: normalised λ, q, log1p q, r, p, c, h, ℓ95, e, ρ, sin/cos time of day.
- A∈ℝ^{128×12} is a fixed Gaussian matrix.
- Q is a fixed random 128×128 orthogonal matrix, so distractors span every direction and cannot be removed by channel masking or linear projection.
- β_d sets the mean channel share Var(β_dQξ)/Var(u)=d∈{0, 0.5, 0.8, 0.95}.

- **Main arm (unpredictable):** ξ∈ℝ¹²⁸ comes from an independent RNG: 50% Student-t(ν=3), 25% Bernoulli(0.05)×Exp(1) bursts, 25% AR(1) with ρ≤0.2. Acceptance: lag-1 |autocorr|≤0.2 and |corr(ξ,s)|<0.02.
- **Predictable arm (falsification, d=0.8):** 50% AR(1) with ρ∈[0.95,0.99], 25% resetting random walks, 25% random-period sawtooth.
- **Golden channels** (clean ρ, ℓ95, e) never enter o_t. Only rule baselines and the approver read them, a declared handicap against the learned agents.
- **Data:** the behaviour policy is clean HPA (target 0.6) plus 30% random gate-allowed actions, r₀~U{2..40}, and 2%/step replica resets. Trajectories are generated once (200 train and 50 val episodes; 57.6k transitions). Each is rendered at every level and arm, so levels differ only in observation noise (~30 MB each).

## 3. World models

|Module|Map|Params|
|---|---|---|
|Encoder E|one frame 128→256→256→z∈ℝ¹⁶, LayerNorm+GELU|≈104k|
|Predictor P|[z_{t−2}, z_{t−1}, z_t, u_t, onehot a_t] (61)→128→128→Δz|≈26k|
|Cost head C|[sg(z), u, a]→128→ĉ, Huber|≈4k|
|Inverse dynamics g|[z_t, z_{t+1}]→128→7, cross-entropy (PLDM-style anchor)|≈5k|

The encoder sees a single frame, because stacked-frame targets overlap the input and make i.i.d. distractors predictable. History lives in the predictor, so E(o_{t+k}) always carries fresh noise. Rollout: ẑ_{t+k}=ẑ_{t+k−1}+P(ẑ_{t+k−3..t+k−1}, u, a), with K=4.

- **JEPA-SIGReg (≈139k):**
  - Loss: L=(1/K)Σ_k‖ẑ_{t+k}−E(o_{t+k})‖²+λ·SIGReg(Z)+β·L_ID+L_C, with λ=0.1 (LeWM, untuned) and β=0.1. No EMA, no stop-gradient.
  - SIGReg(Z)=(1/M)Σ_m EP(Za_m), where EP(x)=N∫_{−5}^{5}|φ̂_x(t)−e^{−t²/2}|²e^{−t²/2}dt and φ̂_x(t)=(1/N)Σ_j e^{itx_j}.
  - 17-point trapezoid. M=256 unit directions, redrawn each step and seeded by the global step. Z is all ≈1.8k frame embeddings in the batch.
  - The window follows the paper's code listing (the text says e^{−t²/σ²}, σ=1). The code is written from the equations because the lejepa repo is CC BY-NC.
- **Recon-WM (≈242k):** trunk + decoder 16→256→256→128; L=‖D(z_t)−o_t‖²+(1/K)Σ_k‖D(ẑ_{t+k})−o_{t+k}‖²+β·L_ID+L_C. Same planner.
- **JEPA λ=0:** the collapse control.
- **JEPA-EMA:** stop-gradient targets from an EMA encoder (τ=0.99), no SIGReg. This pits SIGReg against the classic anti-collapse method, not only a strawman.
- **Cost-only (value-equivalent):** Huber loss on rolled-out costs, with the gradient flowing into E and P.
- **Shared by all models:** C sees detached z on both real and imagined latents; L_ID and L_C are identical everywhere.
- **Training:** AdamW lr 1e-3, wd 1e-4, gradient clip 1.0, batch of 256 windows × 7 frames, **1,200 fixed steps**, 2 threads, no per-model tuning.
  - Estimates: JEPA ≈1.5 min (≈1.4 GFLOP/step), Recon ≈2.2 min. Both are measured in hour 2.
  - A 20-min wall-clock kill invalidates a run; steps are never truncated.

## 4. Planner

- **CEM settings:** categorical CEM over 7 actions. H=8 (40 min, longer than the 10-min warm-up), 128 samples, 16 elites, 3 iterations, update p←0.3p+0.7p_elite, shifted warm start.
- **Cost:** Σ_{h<8}0.97^h Ĉ(ẑ_h,u_h,a_h).
- **Rollouts:** the actuator state is rolled forward exactly, so gate masks apply inside rollouts. Failover takes effect after the approval delay.
- **Output:** the top-3 distinct first actions.
- **CPU cost:** smaller than DINO-WM/LeWM (300 samples, 30 elites) so it fits the CPU. That is 24 batched predictor calls ≈0.18 GFLOP per decision, ≈10-15 ms batched over 16 envs. One eval run (4,608 decisions) takes ≈1 min.

**Oracle MPC (a reference, not an upper bound):** the same CEM on 4 copies of the true simulator from the true state. The diurnal/ramp mean is known; spikes and incidents are sampled from the prior. ≈5 ms per decision.

## 5. Agent loop, governance, audit

```
z=E(o_{t-2..t}); for a in CEM(z,u).top3: v=gate.check(a,u,budget)
  allow→execute; deny→next; needs_approval→approver.request(a), next reversible
else noop; emit TraceEnvelope
```

- **Gate:** `configs/gate_v1.json` uses the governed-agent-orchestrator catalog schema (effect, requires_approval, compensation, constraints) with its check_constraints semantics. An irreversible entry without requires_approval fails at load. Every agent, HPA included, uses the same gate and action set.
- **Envelope:** `TraceEnvelope(run_id="{model}-{variant}-s{seed}-{arm}-d{d}-e{ep}", seq=t, type="decision", producer="noisefloor/agent@{sha}", policy_id="gate-v1", data={proposed, executed, effect_class, args, verdict, reason, approval_id, predicted_cost, elite_spread, mlflow_run_id, delta_version})` → STERNWATCH MemoryBus → WatchLedger (idempotent on (run_id, seq)).
- **Surprise fallback** (hold and page when latent error exceeds the training p99 for 3 steps): off in the science grid and used only as an ablation. It never hands control to a golden-channel controller.

## 6. Baselines and ablations

All baselines share the gate and actions.
- **Tuned HPA:**
  - desired=ceil(r·ρ/target); skip if |ρ/target−1|≤0.1.
  - Scale-down takes the max recommendation over the last W steps (W=1 is the 300 s Kubernetes default).
  - The result maps to the nearest scale action.
  - target∈{0.5,0.6,0.7} × W∈{1,3,6}, tuned for J on training episodes.
- **HPA+runbook (0% reference):** tuned HPA, plus shed_10 when ρ>1.1, plus a failover request when e>2% for 2 steps.
- **Predictive:** training-mean time-of-day profile plus a Holt trend on the last 6 clean λ, forecasting 2 steps ahead.
- **Static peak:** r=ceil(p99 training load/(0.7μ)).
- **Oracle MPC:** the 100% reference.
- **Learned controls:** λ=0, EMA, cost-only.

Ablations (all at d=0.8):
- **Tier 2, trained:**
  - Predictable arm: JEPA and Recon, 3 seeds.
  - z-dim 32 for JEPA and Recon, 2 seeds. This asks whether spare SIGReg dimensions absorb distractors.
- **Tier 2, eval-only (2 seeds):**
  - H=4.
  - Goal-embedding cost ‖ẑ_H−z_goal‖², where z_goal is the mean embedding of the lowest-cost decile.
  - MPPI weights exp(τ(φ_i−max φ)).
  - Governance off, which prices the gate in J and $/day.
  - Fallback on.
- **Tier 3:** z-dim 8; λ∈{0.03,0.3}; cost gradient into the encoder; β=0.

## 7. Evaluation protocol

`PREREGISTRATION.md` (hypotheses, metrics, frozen `sim.yaml`) is committed and tagged `prereg-v1` before the grid runs.
- **Seeds:** 5 training seeds at d∈{0, 0.8} (tested levels), 3 at d∈{0.5, 0.95} (descriptive), 3 for controls. Train, val and eval episode seeds are disjoint.
- **Episodes:** 16 held-out per level (4 per family). All agents get common random numbers (same loads, incidents and distractor draws), so every comparison is paired. Plus 8 OOD episodes at d=0.8.
- **Hypotheses:**
  - **H1 (primary):** at d=0.8, ΔJ=J_JEPA−J_Recon<0.
  - **Sanity:** at d=0, the two are equivalent within ±3% (TOST, 90% CI inside the margin).
  - **H2:** J_JEPA<J_runbook at d=0.8.
  - **H3:** erank(λ=0)≤3/16 while SIGReg ≥10/16.
  - **Secondary (Holm-corrected):**
    - JEPA vs Recon at d=0.5 and 0.95.
    - JEPA vs EMA and vs cost-only.
    - Predictable-arm difference-in-differences: JEPA's edge should shrink.
- **Statistics:** paired hierarchical bootstrap (seeds, then episodes; 10k resamples), 95% percentile CIs, IQM beside the mean. A win needs both the bootstrap CI and the seed-level paired t-interval (df=4) to exclude 0.
- **Metrics:** J; gap closed G=(J_runbook−J)/(J_runbook−J_oracle); SLO-violation minutes; replica-hours; shed %; failovers; denials; approvals; churn; decision latency p50/p99.
- **Diagnostics:**
  - Ridge-probe R² for z→state and z→ξ.
  - Effective rank exp(H(σ/Σσ)).
  - k-step error divided by latent variance.
  - Planned vs realised cost.
  - First-action agreement with the oracle on 500 states.
  - Planning metrics decide the result; the LeWM reproduction found that one-step accuracy doesn't predict planning success.
- `RESULTS.md` is published whatever the outcome.

## 8. Data and tracking

Pins:
- `deltalake[pandas,pyarrow]==1.6.6`
- `mlflow-skinny==3.16.1` + sqlalchemy + alembic (checked in hour 1; full mlflow locally for the UI)
- torch 2.14.0+cpu, pyarrow 25.0.1, pandas 3.0.6, duckdb 1.5.6

|Delta table|Partition|Columns|
|---|---|---|
|`lake/trajectories`|split|episode_id, family, t, true_state list<f32>[8], actuator[6], action int8, cost, reset, data_seed|
|`lake/observations`|arm, d|episode_id, t, obs list<f32>[128], xi16[16]|
|`lake/eval_steps`|model, d|run_id, variant, seed, episode_id, t, proposed, executed, verdict, reason, predicted_cost, realized_cost, plan_ms, envelope_seq|
|`lake/results`|arm|run_id, model, variant, train_seed, d, episode_id, family, split, J, violation_min, replica_hours, shed_frac, failovers, denials, approvals, churn, git_sha, data_version|

- **Writes:** `write_deltalake(path, df, mode="append", partition_by=[...])`.
- **Versioning:** each training run logs `DeltaTable(path).version()`, and evaluation reloads `DeltaTable(path, version=v)`.
- **Leakage guard:** the training loader allowlists {obs, action, actuator, cost}. true_state and xi16 are for the oracle, probes and evaluation only.
- **CI shards:** the aggregate job reads each job's tables with `to_pyarrow_table()` and appends them. Partition folders are never copied without the log.

**MLflow:** one sqlite store per process (`sqlite:///mlruns/{job}-{proc}.db`), which avoids the "database is locked" error (#6013). Set `MLFLOW_DISABLE_TELEMETRY=true` and `PYTHONUTF8=1`.
- **`noisefloor-train`:** one run per (model, arm, d, seed).
  - Params: config hash, SHA, Delta path+version, λ, β, K, M.
  - Metrics every 50 steps: loss terms, SIGReg, erank, latent std.
  - Final probes.
  - Artifacts: state_dict, config via `log_dict`, and the encoder via `mlflow.pytorch.log_model(..., input_example=np.zeros((2,128),np.float32))`.
- **`noisefloor-eval`:** one run per (model, variant, seed, arm, d), tagged with train_run_id.
- **Traces** on 1 in 8 eval episodes: AGENT `decision` → CHAIN `encode`, CHAIN `cem` (iterations, elite spread), GUARDRAIL `gate`, TOOL `act`, TOOL `emit`. Call `mlflow.flush_trace_async_logging()` at exit.
- **Aggregation:** stores are kept as CI artifacts and never merged; aggregation reads Delta only.

## 9. CPU budget (process-minutes at 2 threads; estimates re-measured in hour 2)

|Block|Runs|Min|
|---|---|---|
|Data generation + rendering|–|3|
|Main JEPA (seeds 5/5/3/3), train 1.5 + eval 1.0|16|40|
|Main Recon, train 2.2 + eval 1.0|16|51|
|OOD eval, d=0.8|10|5|
|λ=0 (d=0, 0.8), EMA, cost-only (d=0.8), 3 seeds|12|30|
|Rule baselines, HPA tuning, oracle|–|10|
|**Tier 1**||**≈139 (2.3 h)**|
|Tier 2: predictable (6), z-dim 32 (4), eval-only (10)|20|38|
|**Tier 1+2**||**≈177 (2.95 h)**|

- **Where the grid runs:** only on CI. 8 matrix jobs run 2 processes each on a 4-vCPU runner: ≈12 min compute plus ≈5 min setup.
- **Laptop:** tests, 200-step smoke runs, and the hour-2 benchmark and pilot (≈15 min). One process at a time, under 1.5 GB.
- **Cut order:** Tier 3 → eval-only → z-dim → predictable arm → training steps 1,200→800 for all models.
- **Never cut:** the oracle, the baselines, 5 seeds at the tested levels, the CIs.

## 10. Repo layout

```
noisefloor/ README.md PREREGISTRATION.md RESULTS.md pyproject.toml ([audit] extra = sternwatch)
  configs/ sim.yaml grid.yaml prices.yaml gate_v1.json gate_v2.json
  noisefloor/ __init__.py (CPU guard)
    sim/{fleet,loads,telemetry}.py   data/{collect,lake,loader}.py
    models/{encoder,predictor,sigreg,jepa,recon,ema,heads,train}.py
    plan/{cem,oracle}.py   agents/{hpa,runbook,predictive,static,wm_agent}.py
    govern/{gate,approver}.py   audit/envelopes.py   eval/{run,probes,stats,report}.py
  tests/ reports/ .github/workflows/{ci,grid,audit}.yml
```

## 11. Tests

1. Sim: q≥0, replica bounds, cost rises with r, 2-step readiness, failover resets h and c, seed determinism.
2. Telemetry: measured share within ±0.02 of d; main-arm autocorrelation ≤0.2; ξ independent of s; identical true-state hash across levels.
3. SIGReg is ≈0 on N(0,I) and large on rank-1 and constant inputs; float64 gradcheck.
4. CPU-only / CUDA-uninitialised; parameter counts within ±10%.
5. The loader rejects true_state and xi16.
6. CEM solves a toy categorical problem and respects masks.
7. Gate: denials, the irreversible-without-approval load error, budget, shed compensation.
8. HPA matches hand-computed Kubernetes cases (tolerance, stabilisation).
9. Envelope schema; WatchLedger idempotency.
10. Delta write/append/`version=0` round-trip.
11. Bootstrap coverage on synthetic data; TOST.
12. End-to-end smoke: 200 steps + 2 episodes.

## 12. CI

- **`ci.yml`** (push/PR, under 10 min):
  1. Set up Python 3.11.
  2. `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` as its own step, because the PyPI Linux wheel pulls in CUDA.
  3. `pip install -e .[dev]`.
  4. ruff, pytest, smoke run.
- **`grid.yml`** (workflow_dispatch, tier input):
  1. A data job uploads the lake.
  2. 8 matrix jobs run the grid.
  3. An aggregate job merges the Delta shards, bootstraps, and renders `reports/headline.png`, the table and `RESULTS.md`. Outputs are stamped with the SHA, prereg tag and command. The owner commits them.
- **`audit.yml`** (manual): STERNWATCH's AutoMQ compose (Docker on the runner only) receives 3 episodes. PolicyEcho replays them under `gate_v2.json` (≤24 replicas, no failover) and lists the decisions that would flip.

## 13. Headline result

**Chart:**
- x-axis: distractor share of variance (0/50/80/95%).
- y-axis: % of the runbook→oracle gap closed.
- Lines with 95% CI bands: JEPA-SIGReg, Recon-WM, JEPA λ=0, predictive and tuned HPA. The last two stay flat because they read clean metrics.
- Caption: "At 80% noise, JEPA closes X% [CI] of the gap vs Y% [CI] for reconstruction (5 seeds × 16 paired days, commit abc123)."
- Reply image: probe R² (state, distractors) vs d.
- If JEPA loses, the title says so.

**Table (d=0.8):**
- Rows: static, tuned HPA, runbook, predictive, Recon, λ=0, EMA, cost-only, JEPA, oracle.
- Columns: J [95% CI], gap closed, SLO-violation min/day, replica-h/day, ΔJ vs Recon [CI].

## 14. First two hours, and what could go wrong

**Hour 1:** build the sim, HPA/runbook, oracle and gate. Headroom gate: the oracle must be ≥25% below runbook J at d=0 on validation seeds. If not, tune warm-up and spike rate on validation seeds only, then freeze.

**Hour 2:**
1. SIGReg and gate tests.
2. Benchmark 100 training steps and 100 decisions.
3. Probe-only pilot: JEPA vs autoencoder R² at d=0.95, on a separate data seed, before any planning metric is seen.
4. Set `grid.yaml` from the measured timings.
5. Tag `prereg-v1`.

What could go wrong, and the fallback:
- **No headroom** → tune the sim before prereg and disclose it.
- **No separation in the pilot** → only distractor dimensionality may change, before prereg and judged on probes. After the tag, a null result is published as is.
- **SIGReg instability** → tests, erank every 50 steps, and gradient clipping fixed in advance.
- **Planner exploits model error** → if realised/planned cost exceeds 1.5, add an elite-spread penalty to all learned models equally and log it as a deviation.
- **Budget overrun** → follow the cut order.
- **Windows tooling** → no-space paths, per-process stores, `PYTHONUTF8=1` (MLflow #26200).
- **JEPA loses** → publish anyway.

## 15. What this does not claim

- It is not Kubernetes: actions are discrete, queueing latency is approximate, and prices are assumed.
- Distractors are state-independent by construction. A win shows the mechanism works where that premise holds, not that real dashboards look like this.
- No claim about JEPA vs reconstruction in general, on pixels or at scale.
- It is not RL: the data is offline and comes from one behaviour-policy family.
- Approvals are simulated, the gate is not a safety proof, and the oracle is not optimal.
- 3-seed cells are descriptive only.

## 16. References (verified in research)

**JEPA and world models**
- LeJEPA: https://arxiv.org/abs/2511.08544 (code https://github.com/rbalestr-lab/lejepa is CC BY-NC and is not copied)
- LeWorldModel: https://arxiv.org/abs/2603.19312, https://github.com/lucas-maes/le-wm
- LeWM reproduction: https://arxiv.org/abs/2608.10145
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- PLDM: https://arxiv.org/abs/2502.14819
- TD-MPC2: https://arxiv.org/abs/2310.16828
- DINO-WM: https://arxiv.org/html/2411.04983
- DreamerV3: https://arxiv.org/html/2301.04104v2

**Distractors**
- MuDreamer: https://export.arxiv.org/api/query?id_list=2405.15083
- DreamerPro: https://export.arxiv.org/api/query?id_list=2110.14565
- Distracting Control Suite: https://export.arxiv.org/api/query?id_list=2101.02722

**Planning**
- MPPI: https://ar5iv.labs.arxiv.org/html/1509.01149
- PlaNet CEM: https://ar5iv.labs.arxiv.org/html/1811.04551

**Autoscaling and queueing**
- Kubernetes HPA: https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
- M/M/c: https://en.wikipedia.org/wiki/M/M/c_queue
- Sakasegawa: https://github.com/VividCortex/approx-queueing-theory
- SRE SLOs: https://sre.google/sre-book/service-level-objectives/

**Tooling**
- delta-rs writing: https://raw.githubusercontent.com/delta-io/delta-rs/python-v1.6.6/docs/usage/writing/index.md
- delta-rs #2425: https://github.com/delta-io/delta-rs/issues/2425
- MLflow changelog: https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md
- MLflow tracing: https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/
- MLflow #6013: https://github.com/mlflow/mlflow/issues/6013
- MLflow #26200: https://github.com/mlflow/mlflow/issues/26200
- PyTorch CPU wheels: https://download.pytorch.org/whl/cpu/torch/