"""Pre-registration diagnostics: train one model on the pilot data seed, save it under .checks/, and
measure what its latent contains (ridge probes) and how well its violation-risk head ranks and
calibrates on real and imagined latents. Usage: python scripts/diagnose.py VARIANT D [TAG]"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from noisefloor.config import sim_config
from noisefloor.data.collect import build_world, collect, level_key, render, world_meta
from noisefloor.data.lake import load_probe_arrays, load_training_arrays, write_observations, write_trajectories
from noisefloor.eval.probes import kstep_error, probe
from noisefloor.models.train import CONTEXT, K, TrainConfig, Windows, save, train

variant, d = sys.argv[1], float(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "v2"
cfg = sim_config()
key = level_key("main", d)
SCR = ROOT / ".checks"
mdir = SCR / "diag_models" / f"{tag}_{variant}_{key}"
lake = SCR / "diag_lakes" / f"{tag}_{variant}_{key}"
shutil.rmtree(lake, ignore_errors=True)
tr = collect(cfg, seed=777001)
write_trajectories(lake / "trajectories", tr, 777001)
world = build_world(tr, cfg, "main", d)
obs, xi = render(tr, cfg, world)
write_observations(lake / "observations", key, tr.episode_id, obs, xi)
out = train(load_training_arrays(lake, key), TrainConfig(variant=variant, seed=0))
model = out["model"]
save(model, TrainConfig(variant=variant), mdir, extra={"world": world_meta(world)})
fit, test = load_probe_arrays(lake, key, splits=("train",)), load_probe_arrays(lake, key, splits=("val",))
val = load_training_arrays(lake, key, splits=("val",))
diag = {**probe(model, fit, test), **kstep_error(model, val)}
# risk head calibration: real latents and 4-step imagined latents
win = Windows(val)
idx = np.random.default_rng(0).integers(0, len(win), 4096)
o, u, a, v = win.batch(idx)
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    B = o.shape[0]
    z = model.encoder(o.reshape(B * o.shape[1], -1)).reshape(B, o.shape[1], -1)
    zh = model.rollout(z[:, :CONTEXT], u[:, CONTEXT - 1:CONTEXT - 1 + K], a[:, CONTEXT - 1:CONTEXT - 1 + K])
    p_real = torch.sigmoid(model.risk(z, u, a, 0)).numpy().ravel()
    p_imag = torch.sigmoid(model.risk(zh, u[:, CONTEXT:], a[:, CONTEXT:], torch.arange(1, K + 1))).numpy()
y_real = v.numpy().ravel(); y_imag = v[:, CONTEXT:].numpy()
diag["risk_auc_real"] = roc_auc_score(y_real, p_real)
diag["risk_brier_real"] = float(((p_real - y_real) ** 2).mean())
diag["risk_mean_p_real"] = float(p_real.mean()); diag["viol_rate"] = float(y_real.mean())
for k in range(K):
    diag[f"risk_auc_imag_k{k+1}"] = roc_auc_score(y_imag[:, k], p_imag[:, k])
diag["risk_mean_p_imag_k4"] = float(p_imag[:, 3].mean()); diag["risk_mean_p_imag_k8"] = float(p_imag[:, 7].mean())
print(json.dumps({"variant": variant, "d": d, **{k: round(float(x), 3) for k, x in diag.items()}}), flush=True)
(mdir / "diag.json").write_text(json.dumps(diag, indent=2))
shutil.rmtree(lake, ignore_errors=True)
