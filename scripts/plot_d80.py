"""Bar chart of every model's daily cost at d = 0.8 with 95% intervals, from reports/results.json.
Presentation only; the numbers are the ones in RESULTS.md.

    python scripts/plot_d80.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rep = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    rows = sorted(rep["table_d80"], key=lambda r: r["J"]["mean"])
    learned = {"recon", "lambda0", "ema", "costonly", "jepa"}
    colors = ["#1f77b4" if r["model"] == "jepa" else "#d62728" if r["model"] == "recon" else
              "#9ecae1" if r["model"] in learned else "#bdbdbd" for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    y = range(len(rows))
    means = [r["J"]["mean"] for r in rows]
    err = [[r["J"]["mean"] - r["J"]["lo"] for r in rows], [r["J"]["hi"] - r["J"]["mean"] for r in rows]]
    ax.barh(list(y), means, xerr=err, color=colors, capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r['label']}  ({r['seeds']} seed{'s' if r['seeds'] > 1 else ''})" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("daily cost J (lower is better), mean and 95% CI over seeds x 32 paired days")
    ax.set_title("80% of telemetry variance is noise: every agent, same days, same gate", fontsize=10)
    for i, r in enumerate(rows):
        ax.text(r["J"]["hi"] + 5, i, f"{r['J']['mean']:.0f}", va="center", fontsize=7)
    ax.text(0.99, 0.02, "grey: reads clean metrics or the true simulator", transform=ax.transAxes, ha="right", fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "d80_models.png")


if __name__ == "__main__":
    main()
