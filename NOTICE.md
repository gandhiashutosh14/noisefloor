# Notices

NOISEFLOOR is MIT-licensed (see `LICENSE`). It reimplements published methods; this file records
where each idea comes from and what, if anything, was reused.

| Component | Source | What was reused |
|---|---|---|
| SIGReg (`noisefloor/models/sigreg.py`) | Balestriero and LeCun, *LeJEPA*, arXiv 2511.08544 | The method, implemented from the paper's equations. The LeJEPA reference code (github.com/rbalestr-lab/lejepa) is CC BY-NC and was not copied or consulted for code. |
| SIGReg numerics: per-time-step statistic, [-3, 3] grid, 17 knots, window | Maes et al., *LeWorldModel*, arXiv 2603.19312; github.com/lucas-maes/le-wm (MIT, Copyright (c) 2026 Lucas Maes) | Numerical choices matched to `module.SIGReg` so that lambda = 0.1 means what it means there; the code here is written independently. |
| World-model recipe: encoder + action-conditioned predictor, prediction loss + SIGReg, no EMA or stop-gradient, CEM planning | LeWorldModel (as above) | The recipe, scaled down to CPU (MLPs instead of a ViT, 16-dimensional latent). |
| Inverse-dynamics anchor | Sobal et al., *PLDM*, arXiv 2502.14819 | The idea of an inverse-dynamics term. |
| HPA baseline | Kubernetes Horizontal Pod Autoscaler algorithm (kubernetes.io docs) | The published formula, tolerance and stabilisation window. |
| Latency model | Sakasegawa approximation for M/M/c, with Little's law | Textbook formulas. |
| Gate catalog schema and constraint semantics | `governed-agent-orchestrator` (same author, MIT) | The schema and `check_constraints` rules, reimplemented. |
| Audit envelopes | `tracewake` (same author, MIT) | Used as an optional dependency (`pip install -e .[audit]`). |
