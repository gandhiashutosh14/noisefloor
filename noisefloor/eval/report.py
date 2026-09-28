"""Turn the aggregated Delta tables into RESULTS.md, reports/results.json and the two charts.

Everything here is computed from ``lake/results`` (one row per model x seed x episode) and
``lake/train_runs`` (one row per trained model); nothing is read from MLflow. The hypotheses and
their tests are the ones fixed in PREREGISTRATION.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from deltalake import DeltaTable

from .stats import hierarchical_bootstrap, holm, iqm, tost, win

LEVELS_MAIN = [0.0, 0.5, 0.8, 0.95]
ROW_ORDER = ["static", "hpa", "runbook", "predictive", "recon", "lambda0", "ema", "costonly", "jepa", "oracle"]
LABELS = {"static": "Static peak", "hpa": "Tuned HPA", "runbook": "HPA + runbook", "predictive": "Predictive",
          "recon": "Recon-WM", "lambda0": "JEPA, lambda = 0", "ema": "JEPA-EMA", "costonly": "Cost-only",
          "jepa": "JEPA-SIGReg", "oracle": "Oracle MPC"}


def _load(lake: Path):
    res = DeltaTable(str(lake / "results")).to_pandas()
    runs = DeltaTable(str(lake / "train_runs")).to_pandas() if (lake / "train_runs" / "_delta_log").exists() else pd.DataFrame()
    return res, runs


def matrix(res: pd.DataFrame, model: str, arm: str, d: float, variant: str = "main", z: int = 16,
           seeds: Optional[List[int]] = None) -> np.ndarray:
    """(seeds, episodes) J matrix for a learned model, episodes sorted by id."""
    q = res[(res.model == model) & (res.arm == arm) & (np.isclose(res.d, d)) & (res.variant == variant) & (res.z_dim == z)]
    if seeds is not None:
        q = q[q.train_seed.isin(seeds)]
    if q.empty:
        return np.zeros((0, 0))
    return q.pivot_table(index="train_seed", columns="episode_id", values="J").sort_index().to_numpy()


def baseline(res: pd.DataFrame, model: str, variant: str = "main") -> np.ndarray:
    q = res[(res.model == model) & (res.arm == "clean") & (res.variant == variant)].sort_values("episode_id")
    return q.J.to_numpy()


def gap_closed(J: np.ndarray, Jrb: np.ndarray, Jor: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    J = np.atleast_2d(J)
    S, E = J.shape
    rng = np.random.default_rng(seed)
    si, ei = rng.integers(0, S, (n_boot, S)), rng.integers(0, E, (n_boot, E))
    m = J[si[:, :, None], ei[:, None, :]].mean(axis=(1, 2))
    rb, orc = Jrb[ei].mean(1), Jor[ei].mean(1)
    g = (rb - m) / (rb - orc)
    point = (Jrb.mean() - J.mean()) / (Jrb.mean() - Jor.mean())
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi)}


def _fmt(b: Dict[str, float], pct: bool = False) -> str:
    if pct:
        return f"{100 * b['mean']:.0f}% [{100 * b['lo']:.0f}, {100 * b['hi']:.0f}]"
    return f"{b['mean']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}]"


def _per_day(res: pd.DataFrame, model: str, arm: str, d: float, col: str, variant: str = "main") -> float:
    q = res[(res.model == model) & (res.arm == arm) & (res.variant == variant)]
    if arm != "clean":
        q = q[np.isclose(q.d, d) & (q.z_dim == 16)]
    return float(q[col].mean()) if len(q) else float("nan")


def build_report(lake: Path, out: Path, prereg_tag: str = "prereg-v1", command: str = "") -> Dict:
    res, runs = _load(lake)
    Jrb, Jor = baseline(res, "runbook"), baseline(res, "oracle")
    rep: Dict = {"generated": time.strftime("%Y-%m-%d"), "prereg_tag": prereg_tag, "command": command,
                 "git_sha": sorted(set(res.git_sha))[-1] if "git_sha" in res else "", "data_hash": sorted(set(res.data_hash)),
                 "episodes": int(len(Jrb)), "table_d80": [], "hypotheses": {}, "curves": {}, "probes": {}}

    # ------------------------------------------------------------------ table at d = 0.8
    for m in ROW_ORDER:
        if m in ("static", "hpa", "runbook", "predictive", "oracle"):
            J = baseline(res, m)[None, :]
            arm = "clean"
        else:
            J = matrix(res, m, "main", 0.8)
            arm = "main"
        if J.size == 0:
            continue
        row = {"model": m, "label": LABELS[m], "seeds": int(J.shape[0]), "J": hierarchical_bootstrap(J),
               "J_iqm": iqm(J), "gap_closed": gap_closed(J, Jrb, Jor),
               "slo_violation_min_per_day": _per_day(res, m, arm, 0.8, "violation_min"),
               "replica_hours_per_day": _per_day(res, m, arm, 0.8, "replica_hours"),
               "failovers_per_day": _per_day(res, m, arm, 0.8, "failovers"),
               "denials_per_day": _per_day(res, m, arm, 0.8, "denials")}
        Jr = matrix(res, "recon", "main", 0.8)
        if m not in ("recon",) and arm == "main" and Jr.size:
            k = min(len(J), len(Jr))
            row["dJ_vs_recon"] = hierarchical_bootstrap(J[:k] - Jr[:k])
        rep["table_d80"].append(row)

    # ------------------------------------------------------------------ pre-registered hypotheses
    H = rep["hypotheses"]
    Jj8, Jr8 = matrix(res, "jepa", "main", 0.8), matrix(res, "recon", "main", 0.8)
    if Jj8.size and Jr8.size:
        H["H1_jepa_beats_recon_d80"] = win(Jj8 - Jr8)
    Jj0, Jr0 = matrix(res, "jepa", "main", 0.0), matrix(res, "recon", "main", 0.0)
    if Jj0.size and Jr0.size:
        H["sanity_equivalent_d0"] = tost((Jj0 - Jr0) / Jr0, margin=0.03)
    if Jj8.size:
        H["H2_jepa_beats_runbook_d80"] = win(Jj8 - Jrb[None, :])
    if len(runs):
        er = runs[(runs.z_dim == 16) & (runs.arm == "main")]
        e0 = er[er.variant == "lambda0"].erank
        e1 = er[er.variant == "jepa"].erank
        H["H3_collapse"] = {"erank_lambda0": [float(x) for x in e0], "erank_sigreg": [float(x) for x in e1],
                            "holds": bool(len(e0) and len(e1) and e0.max() <= 3.0 and e1.min() >= 10.0)}
    secondary = {}
    for d in (0.5, 0.95):
        a, b = matrix(res, "jepa", "main", d), matrix(res, "recon", "main", d)
        if a.size and b.size:
            k = min(len(a), len(b))
            secondary[f"jepa_vs_recon_d{int(d * 100)}"] = a[:k] - b[:k]
    for other in ("ema", "costonly"):
        b = matrix(res, other, "main", 0.8)
        if Jj8.size and b.size:
            k = min(len(Jj8), len(b))
            secondary[f"jepa_vs_{other}_d80"] = Jj8[:k] - b[:k]
    for arm in ("predictable", "isotropic"):
        a, b = matrix(res, "jepa", arm, 0.8), matrix(res, "recon", arm, 0.8)
        if a.size and b.size and Jj8.size:
            k = min(len(a), len(b), len(Jj8), len(Jr8))
            if arm == "predictable":
                secondary["did_predictable_minus_main"] = (a[:k] - b[:k]) - (Jj8[:k] - Jr8[:k])
            else:
                secondary["jepa_vs_recon_isotropic_d80"] = a[:k] - b[:k]
    names = list(secondary)
    tests = {n: win(secondary[n]) for n in names}
    adj = holm([tests[n]["bootstrap"]["p_two_sided"] for n in names]) if names else []
    for n, p in zip(names, adj):
        tests[n]["p_holm"] = p
    H["secondary"] = tests

    # ------------------------------------------------------------------ curves over d
    for m in ("jepa", "recon", "lambda0"):
        pts = []
        for d in LEVELS_MAIN:
            J = matrix(res, m, "main", d)
            if J.size:
                pts.append({"d": d, "seeds": int(J.shape[0]), "gap": gap_closed(J, Jrb, Jor), "J": hierarchical_bootstrap(J)})
        rep["curves"][m] = pts
    for m in ("hpa", "predictive"):
        g = gap_closed(baseline(res, m)[None, :], Jrb, Jor)
        rep["curves"][m] = [{"d": d, "seeds": 1, "gap": g} for d in LEVELS_MAIN]
    if len(runs):
        for m in ("jepa", "recon", "lambda0", "ema", "costonly"):
            q = runs[(runs.variant == m) & (runs.z_dim == 16) & (runs.arm == "main")]
            rep["probes"][m] = [{"d": float(d), "state_r2": float(g.probe_state_r2.mean()), "xi_r2": float(g.probe_xi_r2.mean()),
                                 "erank": float(g.erank.mean()), "kstep4": float(g.kstep_err_4.mean()), "n": int(len(g))}
                                for d, g in q.groupby("d")]
        rep["probes_other"] = [{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in r.items()
                                if k in ("run", "variant", "arm", "d", "z_dim", "probe_state_r2", "probe_xi_r2", "erank", "kstep_err_4")}
                               for r in runs[(runs.arm != "main") | (runs.z_dim != 16)].to_dict("records")]
    ood = {}
    for m in ("jepa", "recon"):
        J = matrix(res, m, "main", 0.8, variant="ood")
        if J.size:
            ood[m] = {"J": hierarchical_bootstrap(J), "gap": gap_closed(J, baseline(res, "runbook", "ood"), baseline(res, "oracle", "ood"))}
    for m in ("runbook", "oracle", "hpa"):
        b = baseline(res, m, "ood")
        if b.size:
            ood[m] = {"J": hierarchical_bootstrap(b[None, :])}
    rep["ood"] = ood
    h4 = matrix(res, "jepa", "main", 0.8, variant="h4")
    if h4.size:
        rep["horizon4"] = {"J": hierarchical_bootstrap(h4), "vs_h8": hierarchical_bootstrap(h4 - Jj8[:len(h4)])}
    rop = res[(res.arm == "main") & np.isclose(res.d, 0.8) & (res.variant == "main") & (res.z_dim == 16)]
    rep["realised_over_planned"] = {m: float(g.realised_over_planned.mean()) for m, g in rop.groupby("model")}

    out.mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    _charts(rep, out / "reports")
    (out / "RESULTS.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


def _charts(rep: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"jepa": "#1f77b4", "recon": "#d62728", "lambda0": "#9467bd", "hpa": "#7f7f7f", "predictive": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for m, pts in rep["curves"].items():
        if not pts:
            continue
        x = [100 * p["d"] for p in pts]
        y = [100 * p["gap"]["mean"] for p in pts]
        lo = [100 * p["gap"]["lo"] for p in pts]
        hi = [100 * p["gap"]["hi"] for p in pts]
        style = "--" if m in ("hpa", "predictive") else "-"
        ax.plot(x, y, style, marker="o" if style == "-" else None, color=colors[m], label=LABELS.get(m, m))
        if style == "-":
            ax.fill_between(x, lo, hi, color=colors[m], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(100, color="black", lw=0.8, ls=":")
    ax.set_xlabel("distractor share of telemetry variance (%)")
    ax.set_ylabel("runbook-to-oracle gap closed (%)")
    ax.set_xticks([0, 50, 80, 95])
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Planning in a learned latent space as telemetry noise grows", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "headline.png")
    plt.close(fig)
    if rep.get("probes"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=150)
        for m in ("jepa", "recon"):
            pts = rep["probes"].get(m) or []
            if pts:
                axes[0].plot([100 * p["d"] for p in pts], [p["state_r2"] for p in pts], "-o", color=colors[m], label=LABELS[m])
                axes[1].plot([100 * p["d"] for p in pts], [p["xi_r2"] for p in pts], "-o", color=colors[m], label=LABELS[m])
        axes[0].set_title("ridge probe R^2: true state", fontsize=9)
        axes[1].set_title("ridge probe R^2: distractors", fontsize=9)
        for a in axes:
            a.set_xlabel("distractor share (%)")
            a.set_xticks([0, 50, 80, 95])
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "probes.png")
        plt.close(fig)


def render_markdown(rep: Dict) -> str:
    H = rep["hypotheses"]
    L = ["# NOISEFLOOR results", "",
         f"Generated {rep['generated']} from commit `{rep['git_sha']}` under pre-registration tag `{rep['prereg_tag']}`; "
         f"data hash {', '.join(rep['data_hash'])}; {rep['episodes']} paired evaluation days per cell.", "",
         f"Command: `{rep['command']}`", "", "![headline](reports/headline.png)", "",
         "## Table at d = 0.8 (80% of telemetry variance is distractor)", "",
         "| model | seeds | J [95% CI] | IQM | gap closed | SLO-violation min/day | replica-h/day | dJ vs Recon [95% CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rep["table_d80"]:
        dj = _fmt(r["dJ_vs_recon"]) if "dJ_vs_recon" in r else "-"
        L.append(f"| {r['label']} | {r['seeds']} | {_fmt(r['J'])} | {r['J_iqm']:.1f} | {_fmt(r['gap_closed'], pct=True)} | "
                 f"{r['slo_violation_min_per_day']:.0f} | {r['replica_hours_per_day']:.0f} | {dj} |")
    L += ["", "Rule baselines and the oracle read clean utilisation, latency and error channels (a declared handicap in "
          "their favour) and do not depend on d. Gap closed = (J_runbook - J) / (J_runbook - J_oracle).", "",
          "## Pre-registered hypotheses", ""]

    def verdict(w):
        return "supported" if w["win"] else ("reversed" if w["loss"] else "not supported (interval includes 0)")

    if "H1_jepa_beats_recon_d80" in H:
        w = H["H1_jepa_beats_recon_d80"]
        L.append(f"- **H1 (primary)**, J_JEPA - J_Recon < 0 at d = 0.8: {_fmt(w['bootstrap'])} (bootstrap), seed t-interval "
                 f"[{w['seed_t']['lo']:.1f}, {w['seed_t']['hi']:.1f}]: **{verdict(w)}**.")
    if "sanity_equivalent_d0" in H:
        t = H["sanity_equivalent_d0"]
        L.append(f"- **Sanity**, equivalence within 3% at d = 0: 90% interval of the relative difference "
                 f"[{100 * t['interval90'][0]:.1f}%, {100 * t['interval90'][1]:.1f}%]: "
                 f"**{'equivalent' if t['equivalent'] else 'not shown equivalent'}**.")
    if "H2_jepa_beats_runbook_d80" in H:
        w = H["H2_jepa_beats_runbook_d80"]
        L.append(f"- **H2**, J_JEPA - J_runbook < 0 at d = 0.8: {_fmt(w['bootstrap'])}: **{verdict(w)}**.")
    if "H3_collapse" in H:
        h = H["H3_collapse"]
        L.append(f"- **H3**, effective rank of lambda = 0 <= 3 and of SIGReg >= 10 (of 16): lambda = 0 "
                 f"{[round(x, 1) for x in h['erank_lambda0']]}, SIGReg {[round(x, 1) for x in h['erank_sigreg']]}: "
                 f"**{'holds' if h['holds'] else 'does not hold'}**.")
    if H.get("secondary"):
        L += ["", "Secondary comparisons (Holm-adjusted across the family; 3-seed cells are descriptive):", "",
              "| comparison (candidate - reference) | mean [95% CI] | seed t-interval | Holm p | verdict |", "|---|---|---|---|---|"]
        for n, w in H["secondary"].items():
            L.append(f"| {n} | {_fmt(w['bootstrap'])} | [{w['seed_t']['lo']:.1f}, {w['seed_t']['hi']:.1f}] | "
                     f"{w.get('p_holm', float('nan')):.3f} | {verdict(w)} |")
    if rep.get("probes"):
        L += ["", "## What the latent codes contain", "", "![probes](reports/probes.png)", "",
              "| model | d | state R^2 | distractor R^2 | effective rank | 4-step error / latent variance | models |", "|---|---|---|---|---|---|---|"]
        for m, pts in rep["probes"].items():
            for p in pts:
                L.append(f"| {LABELS.get(m, m)} | {p['d']:.2f} | {p['state_r2']:.3f} | {p['xi_r2']:.3f} | {p['erank']:.1f} | {p['kstep4']:.2f} | {p['n']} |")
    if rep.get("ood"):
        L += ["", "## Unseen load shapes (launch plateau, double spike) at d = 0.8", ""]
        for m, v in rep["ood"].items():
            g = f", gap closed {_fmt(v['gap'], pct=True)}" if "gap" in v else ""
            L.append(f"- {LABELS.get(m, m)}: J {_fmt(v['J'])}{g}")
    if rep.get("horizon4"):
        L += ["", f"Planning horizon 4 instead of 8 (JEPA, d = 0.8): J {_fmt(rep['horizon4']['J'])}, "
              f"difference {_fmt(rep['horizon4']['vs_h8'])}."]
    if rep.get("realised_over_planned"):
        L += ["", "Realised over planned 8-step cost at d = 0.8 (above 1 means the planner was optimistic): " +
              ", ".join(f"{LABELS.get(m, m)} {v:.2f}" for m, v in rep["realised_over_planned"].items()) + "."]
    return "\n".join(L) + "\n"
