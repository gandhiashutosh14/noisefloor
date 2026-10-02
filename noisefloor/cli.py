"""Command line: data, baselines, one cell, one grid shard, aggregate, report, smoke.

    noisefloor data      --lake lake
    noisefloor baselines --lake lake
    noisefloor cell      --variant jepa --arm main --d 0.8 --seed 0 [--z 16] [--evals main,ood]
    noisefloor grid      --tier 1 --shard 0 --of 8 --lake lake-0 --runs runs-0
    noisefloor aggregate --shards "artifacts/*/lake" --out lake-all
    noisefloor report    --lake lake-all
    noisefloor audit     [--bootstrap localhost:9092] (3 episodes -> STERNWATCH log -> ledger -> replay under gate-v2)
    noisefloor smoke     (200 training steps, 2 evaluation days)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import noisefloor  # noqa: F401  (CPU guard runs first)

from . import tracking
from .config import ROOT, sim_config
from .govern.gate import Gate


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_data(a) -> None:
    from .grid import build_lake
    info = build_lake(Path(a.lake), sim_config())
    _log(json.dumps({k: v for k, v in info.items() if k != "levels"}) + f" levels={list(info['levels'])}")


def cmd_baselines(a) -> None:
    from .grid import run_baselines
    tracking.configure(a.job, a.proc)
    run_baselines(Path(a.lake), sim_config(), Gate.load("gate_v1.json"), log=_log, n_eval=a.n_eval)


def cmd_cell(a) -> None:
    from .grid import Cell, eval_cell, train_cell
    tracking.configure(a.job, a.proc)
    cell = Cell(a.variant, a.arm, a.d, a.seed, a.z, tuple(a.evals.split(",")))
    lake, runs = Path(a.lake), Path(a.runs)
    train_cell(cell, lake, runs, steps=a.steps, log=_log)
    eval_cell(cell, lake, runs, sim_config(), Gate.load("gate_v1.json"), log=_log, n_eval=a.n_eval)


def cmd_grid(a) -> None:
    from .grid import build_lake, eval_cell, grid_cells, run_baselines, shard, train_cell
    tracking.configure(a.job, a.shard)
    cfg, gate = sim_config(), Gate.load("gate_v1.json")
    lake, runs = Path(a.lake), Path(a.runs)
    if not (lake / "meta" / "data.json").exists():
        info = build_lake(lake, cfg)
        _log(f"data {info['data_hash']} built in {info['seconds']}s")
    cells = shard(grid_cells(a.tier), a.shard, a.of)
    _log(f"shard {a.shard}/{a.of}: {len(cells)} cells: {[c.name for c in cells]}")
    t0 = time.time()
    if a.shard == 0 and not a.skip_baselines:
        run_baselines(lake, cfg, gate, log=_log)
    for c in cells:
        if a.resume and (runs / c.name / "model.pt").exists():
            _log(f"skip training {c.name} (model exists)")
        else:
            train_cell(c, lake, runs, log=_log)
        eval_cell(c, lake, runs, cfg, gate, log=_log)
    _log(f"shard done in {(time.time() - t0) / 60:.1f} min")


def cmd_aggregate(a) -> None:
    """Merge the shards' Delta tables. Every shard is checked first (same data hash, same commit, same
    table schemas) and nothing is written unless all pass; the output directory must not exist."""
    from deltalake import DeltaTable, write_deltalake
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; aggregate writes a fresh lake (re-running would append duplicates)")
    shards = [Path(p) for p in sorted(glob.glob(a.shards))]
    if not shards:
        raise SystemExit(f"no shards match {a.shards}")
    hashes, shas, schemas = set(), set(), {}
    for lake in shards:
        meta = json.loads((lake / "meta" / "data.json").read_text(encoding="utf-8"))
        hashes.add(meta["data_hash"])
        for table in ("results", "train_runs", "eval_steps"):
            if (lake / table / "_delta_log").exists():
                dt = DeltaTable(str(lake / table))
                schema = dt.schema().to_arrow()
                if schemas.setdefault(table, schema) != schema:
                    raise SystemExit(f"{lake}/{table} has a different schema from the first shard's")
                if table == "results":
                    shas |= set(dt.to_pyarrow_table(columns=["git_sha"]).column("git_sha").unique().to_pylist())
    if len(hashes) != 1:
        raise SystemExit(f"shards saw different data: {sorted(hashes)}")
    if len(shas) != 1:
        raise SystemExit(f"shards ran on different commits: {sorted(shas)}")
    for lake in shards:
        for table in ("results", "train_runs", "eval_steps"):
            if (lake / table / "_delta_log").exists():
                tbl = DeltaTable(str(lake / table)).to_pyarrow_table()
                write_deltalake(str(out / table), tbl, mode="append", partition_by=["arm"] if table == "results" else None)
        tuning = lake / "meta" / "hpa_tuning.json"
        if tuning.exists():
            (out / "meta").mkdir(parents=True, exist_ok=True)
            (out / "meta" / "hpa_tuning.json").write_text(tuning.read_text(encoding="utf-8"), encoding="utf-8")
        _log(f"merged {lake}")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "data.json").write_text(json.dumps({"data_hash": hashes.pop(), "git_sha": shas.pop()}), encoding="utf-8")


def cmd_report(a) -> None:
    from .eval.report import build_report
    build_report(Path(a.lake), Path(a.out), prereg_tag=a.prereg, command=" ".join(sys.argv))


def cmd_audit(a) -> None:
    """Train (or load) a JEPA agent, run a few evaluation days, and audit them through STERNWATCH."""
    from .agents.wm_agent import WMAgent
    from .audit.envelopes import EnvelopeRecorder
    from .audit.replay import run_audit
    from .eval.run import eval_set, load_world, run_agent
    from .grid import Cell, build_lake, train_cell
    from .models.train import load as load_model
    os.environ.setdefault("NOISEFLOOR_MLFLOW", "0")
    cfg, gate = sim_config(), Gate.load("gate_v1.json")
    lake, runs = Path(a.lake), Path(a.runs)
    if not (lake / "meta" / "data.json").exists():
        build_lake(lake, cfg, levels=[("main", 0.8)])
    cell = Cell("jepa", "main", 0.8, 0)
    model_dir = Path(a.model) if a.model else runs / cell.name
    if not (model_dir / "model.pt").exists():
        train_cell(cell, lake, runs, steps=a.steps, log=_log)
        model_dir = runs / cell.name
    model = load_model(model_dir)
    agent = WMAgent(model, cfg, seed=0)
    _, ids = eval_set(cfg, n=a.episodes)
    rec = EnvelopeRecorder("jepa", "audit", 0, "main", 0.8, ids, agent=agent)
    run_agent(agent, cfg, gate, world=load_world(lake, cell.level), n=a.episodes, record=rec)
    if a.bootstrap:
        from sternwatch.bus import KafkaBus
        bus, describe = KafkaBus(a.bootstrap, ready_timeout_s=180.0), f"Kafka-compatible broker at {a.bootstrap}"
    else:
        from sternwatch.bus import MemoryBus
        bus, describe = MemoryBus(), "in-memory log"
    out = run_audit(rec, bus, Path(a.out), Path(a.out).with_suffix(".json"), topic=a.topic, describe=describe)
    _log(json.dumps({k: v for k, v in out.items() if k != "examples"}))
    if not out["idempotent"] or out["gaps"]:
        raise SystemExit("audit failed: ledger not idempotent or has gaps")


def cmd_smoke(a) -> None:
    import shutil

    from .grid import Cell, build_lake, eval_cell, train_cell
    os.environ.setdefault("NOISEFLOOR_MLFLOW", "0")
    base = Path(a.dir)
    shutil.rmtree(base, ignore_errors=True)
    cfg, gate = sim_config(), Gate.load("gate_v1.json")
    info = build_lake(base / "lake", cfg, levels=[("main", 0.8)])
    _log(f"data {info['data_hash']} in {info['seconds']}s")
    cell = Cell("jepa", "main", 0.8, 0, 16, ("main",))
    train_cell(cell, base / "lake", base / "runs", steps=a.steps, log=_log)
    rows = eval_cell(cell, base / "lake", base / "runs", cfg, gate, log=_log, n_eval=2)
    assert len(rows) == 2 and all(r["J"] > 0 for r in rows)
    _log("smoke ok")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="noisefloor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lake", default=str(ROOT / "lake"))
    common.add_argument("--runs", default=str(ROOT / "runs"))
    common.add_argument("--job", default="local")
    common.add_argument("--proc", type=int, default=0)
    common.add_argument("--n-eval", type=int, default=None, help="evaluation days (default: configs/grid.json)")
    sub.add_parser("data", parents=[common]).set_defaults(fn=cmd_data)
    sub.add_parser("baselines", parents=[common]).set_defaults(fn=cmd_baselines)
    c = sub.add_parser("cell", parents=[common])
    c.add_argument("--variant", required=True)
    c.add_argument("--arm", default="main")
    c.add_argument("--d", type=float, default=0.8)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--z", type=int, default=16)
    c.add_argument("--steps", type=int, default=1200)
    c.add_argument("--evals", default="main")
    c.set_defaults(fn=cmd_cell)
    g = sub.add_parser("grid", parents=[common])
    g.add_argument("--tier", type=int, default=1)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--of", type=int, default=1)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--skip-baselines", action="store_true")
    g.set_defaults(fn=cmd_grid)
    ag = sub.add_parser("aggregate")
    ag.add_argument("--shards", required=True)
    ag.add_argument("--out", required=True)
    ag.set_defaults(fn=cmd_aggregate)
    r = sub.add_parser("report", parents=[common])
    r.add_argument("--out", default=str(ROOT))
    r.add_argument("--prereg", default="prereg-v1")
    r.set_defaults(fn=cmd_report)
    au = sub.add_parser("audit", parents=[common])
    au.add_argument("--bootstrap", default=None)
    au.add_argument("--topic", default="noisefloor.decisions")
    au.add_argument("--model", default=None)
    au.add_argument("--episodes", type=int, default=3)
    au.add_argument("--steps", type=int, default=1200)
    au.add_argument("--out", default=str(ROOT / "reports" / "audit.md"))
    au.set_defaults(fn=cmd_audit)
    s = sub.add_parser("smoke")
    s.add_argument("--dir", default=str(ROOT / ".smoke"))
    s.add_argument("--steps", type=int, default=200)
    s.set_defaults(fn=cmd_smoke)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
