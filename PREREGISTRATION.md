# Pre-registration: NOISEFLOOR v1

Registered 2026-09-29 as git tag `prereg-v1`. The tag fixes the code, `configs/sim.json` (hash
`c45352c24e54`), `configs/grid.json` (hash `191e0d9b7f67`) and `configs/gate_v1.json` (hash
`85c4f52f422a`). The confirmatory grid runs after the tag, in GitHub Actions (`grid.yml`), on the
tagged code. `RESULTS.md` is published whatever it shows.

## 1. Question

An SRE agent sees 128 telemetry channels. A share *d* of their variance is distractor noise that is
independent of the system's state and cannot be predicted from the past. The agent plans by
imagining the future in a learned latent space and choosing actions whose imagined future is
cheapest. Does a JEPA-style world model with SIGReg, trained to predict its own future embeddings,
plan better as *d* grows than the same world model trained by reconstructing its input?

## 2. Frozen design (what the tag contains)

| Part | Setting |
|---|---|
| World | Simulated service: 5-minute steps, 288-step days, four load families (diurnal, flash crowd, ramp, regional incident), M/M/c latency, 30 s client timeout, SLO 250 ms p95 and 1% errors. Cost per step: 0.05 per replica, 5 per SLO-violation step, 2 x shed fraction, 0.1 per cache warm, 20 per failover. |
| Telemetry | u = A phi(s) + beta Q xi, o = softplus(u) + 0.05 eps, z-scored with training statistics. 16 distractor sources mixed into all 128 channels through orthonormal columns; beta is calibrated so the distractor share of mean channel variance is exactly d. Main arm: Student-t, burst and weak-AR sources (unpredictable). |
| Data | 200 training and 50 validation days from one behaviour policy: HPA on clean utilisation with a per-day target drawn from U(0.4, 1.3), 30% random gate-allowed actions, 2% per-step replica resets, r0 ~ U{2..40}. Data seed 20260929. Stored in Delta Lake; training reads an allow-list of columns only. |
| Models | Encoder 128 -> 256 -> 256 -> 16 (LayerNorm, GELU), one frame. Predictor on the last three latents, the actuator state and the action. Violation-risk head conditioned on rollout depth. Inverse-dynamics head (beta = 0.1). Variants: `jepa` (prediction + SIGReg, lambda = 0.1), `recon` (autoencoder; predictor fitted to its detached latents), `lambda0`, `ema` (tau = 0.99, stop-gradient targets), `costonly` (risk loss shapes encoder and predictor). One recipe for all: AdamW 1e-3, weight decay 1e-4, clip 1.0, 1,200 steps, batch 256 windows of 11 frames, K = 8 rollout steps, 2 CPU threads. |
| Planner | Categorical CEM, horizon 8, 128 samples, 16 elites, 3 iterations. Cost = exact known cost of the rolled-out actuator + 5 x predicted violation probability, discounted 0.97. Gate masks apply inside rollouts; failover is delayed one step for approval. |
| Governance | Every agent, rule baselines included, passes the same gate (`gate_v1.json`): replica bounds, cache-warm cooldown, shed cap with automatic restore, failover irreversible and approval-only, 6 actions per 12 steps. |
| Baselines | Static peak provisioning; Kubernetes HPA and HPA + runbook, each tuned (target x window) on training-seed days; predictive (time-of-day profile + Holt trend). All read clean channels, a declared handicap in their favour. Oracle MPC: the same CEM on the true simulator from the true state (reference, not a bound). |
| Evaluation | 32 held-out days per cell (8 per family), evaluation seed 5150, ids 100000+; every agent sees the same loads, incidents and distractor draws (paired). 16 out-of-distribution days (launch plateau, double spike). Start at 10 replicas. |

## 3. Hypotheses

Main arm, 16-dimensional latent, 8 training seeds at d in {0, 0.8}.

- **H1 (primary).** At d = 0.8, J_JEPA - J_Recon < 0.
- **Sanity.** At d = 0, JEPA and Recon are equivalent within +/- 3% of J (TOST: the 90% interval of the
  relative difference lies inside +/- 3%). If this fails, H1 is still reported, with that caveat.
- **H2.** At d = 0.8, J_JEPA - J_runbook < 0 (JEPA on noisy telemetry beats the tuned runbook on clean
  metrics).
- **H3.** Effective rank of the latent (of 16): every `lambda0` model <= 3, every `jepa` model >= 10.

Secondary family (Holm-corrected, 5 seeds each; predicted direction in brackets):

1. JEPA vs Recon at d = 0.5 [JEPA lower J]
2. JEPA vs Recon at d = 0.95 [JEPA lower J]
3. JEPA vs JEPA-EMA at d = 0.8 [no prediction]
4. JEPA vs cost-only at d = 0.8 [JEPA lower J]
5. Predictable-arm difference in differences, (J_JEPA - J_Recon)_predictable - (J_JEPA - J_Recon)_main
   at d = 0.8 [positive: JEPA's edge shrinks when the distractors become predictable, since a
   predictive objective then has reason to encode them]
6. JEPA vs Recon on the isotropic arm (128 sources) at d = 0.8 [JEPA lower J]

Exploratory, reported without tests: latent size 8 and 32, planning horizon 4, OOD days, ridge probes
(state and distractors), k-step latent error, realised over planned cost.

## 4. Analysis

- Per cell, a (seeds x days) matrix of J. Candidate and reference are paired by seed index and day.
- **Win rule.** The 95% hierarchical bootstrap interval (resample seeds, then days; 10,000 draws) and
  the seed-level t-interval (df = seeds - 1) must both exclude zero. "Reversed" means both exclude
  zero on the other side; anything else is "not supported".
- Gap closed = (J_runbook - J) / (J_runbook - J_oracle), with a bootstrap interval over the same
  resamples. Means and interquartile means are both reported.
- 3-seed cells are descriptive only.

## 5. Deviation policy

Anything changed after the tag is listed in `RESULTS.md` under "Deviations", with the reason, and the
tagged analysis is reported first. If the planner turns out to exploit model error on the grid
(realised over planned cost above 1.5), that is reported, not fixed.

## 6. What was seen before registration

Everything below used the separate pilot data seed 777001, and planning was checked only on 16
validation days (ids 9000-9015, the days of the headroom check). No trained model was run on an
evaluation day. Three smoke runs touched evaluation days 100000-100001 with 20- or 200-step models
(J 741, 1,467, 374). Every change below applies to all learned models, and each was prompted by a
failure of the design, not by which model it favoured. The files are in `reports/pilot/`.

**Headroom (hour 1).** On the validation days, the tuned runbook scored J = 295.6 and the oracle
220.9, a 25.3% gap. The spec required at least 25%.

**Probe pilots at d = 0.95** (ridge R^2 of the latent onto the true state and onto the distractors):

| round | change before the round | JEPA state / distractor | Recon state / distractor |
|---|---|---|---|
| 1 | first build: 128 distractor sources; SIGReg pooled over every frame | 0.175 / 0.072 | 0.303 / 0.035 |
| 2 | 16 distractor sources (the spec's one allowed change after a flat pilot) | 0.248 / 0.652 | 0.271 / 0.757 |
| 3 | two implementation fixes (below), 16 sources | 0.367 / 0.628 | 0.319 / 0.789 |
| 3 | the same, 128 sources | 0.303 / 0.064 | 0.213 / 0.038 |

The fixes before round 3:

- **SIGReg scaling.** Round 1 multiplied the Epps-Pulley statistic by every frame in the batch
  (about 1,800). The MIT-licensed LeWorldModel code computes it per time step over the batch. Round 1
  therefore ran SIGReg far stronger than lambda = 0.1 means in the reference, so it was aligned.
- **Reconstruction baseline.** Its predictor was trained only through the decoder, so imagined
  latents drifted to 16,000 x the latent variance within 4 steps. It is now fitted to the
  autoencoder's own detached latents, exactly as JEPA's predictor is fitted to JEPA's latents.

With both fixes, JEPA leads at both distractor dimensionalities. The 16-source world was kept as
primary: a few unrelated processes leaking into every metric is the more realistic case, and the
harder one for a reconstruction bottleneck. The 128-source world is secondary test 6, so the choice
cannot decide the result.

**Planning checks** on the validation days, J (realised over planned 8-step cost):

| check | design | JEPA d=0 | JEPA d=0.8 | Recon d=0 | Recon d=0.8 |
|---|---|---|---|---|---|
| 1 | Huber cost head, K = 4, behaviour target fixed at 0.6 | 327 (1.6) | 835 (2.9) | 255 (1.3) | 1,110 (3.4) |
| 2 | violation-risk head + exact known cost | 403 (1.8) | 611 (2.2) | 229 (1.0) | 840 (2.6) |
| 3 | risk head conditioned on rollout depth, K = 8 | 370 (1.6) | 846 (2.7) | 261 (1.1) | 841 (2.5) |
| 4 | behaviour HPA target per day from U(0.4, 1.3) | 263 (1.3) | 278 (1.0) | 258 (1.1) | 353 (1.4) |

Each change fixed a failure the diagnostics exposed:

- **Check 1.** The Huber cost head acted like a median: it predicted 3.2-4.1 for violation steps that
  cost 6.2. The step cost was split into its exactly known part and a learned violation probability.
- **Check 2.** Imagined latents regress toward typical states, so 4-step imagined risk averaged
  1.2-1.8% against a true 2.8%. The risk head was given the rollout depth, and rollouts were trained
  to the planning horizon.
- **Check 3.** Planner variants with the d = 0.8 JEPA model:
  - horizon 2: 1,446
  - horizon 4: 906
  - a weaker search: 567
  - a latent filter: 767
  - 3 x violation weight: 425

  The planner drove the fleet to 2-3 replicas. Under a fixed-target HPA, few replicas almost always
  meant low load, so the data never showed the cost of running hot. The behaviour policy became a
  mix of cautious and reckless operators.

**After check 4.** Two final changes, with no further planning check:

- beta is now calibrated on the measured variance of day-long distractor streams. The predictable
  arm's random walks had pushed its share to about 0.82 instead of 0.80.
- Seeds and evaluation days were raised (5/3/3 seeds and 16 days in the spec became 8/5/5/3 and 32)
  after `reports/benchmark.json` showed the grid costs about 4 CPU-hours.

**Cut from the design spec.** The governance-off ablation (the simulator enforces the gate's
constraints) and the goal-embedding, MPPI and surprise-fallback ablations.
