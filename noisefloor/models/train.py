"""Training for every world-model variant, with one fixed recipe (no per-model tuning).

Each batch is 256 windows of 11 consecutive frames (t-2 .. t+8). The encoder embeds every frame; the
predictor starts from the three context latents (t-2, t-1, t) and rolls K = 8 steps forward (the
planning horizon) using
the logged actuator states and actions. Windows that contain a behaviour-policy replica reset are
skipped, because the reset is not caused by any action the model can see.

Variants (loss terms; L_ID and L_C are identical everywhere):
  jepa     (1/K) sum_k ||z_hat_{t+k} - E(o_{t+k})||^2 + lambda * SIGReg(latents per time step)  lambda = 0.1
  lambda0  the same with lambda = 0 (collapse control)
  ema      targets from an EMA copy of the encoder (tau = 0.99) with stop-gradient, no SIGReg
  recon    ||D(z) - o||^2 on every real frame trains the encoder (an autoencoder); the predictor is fitted
           to the autoencoder's own latents, (1/K) sum_k ||z_hat_{t+k} - sg(z_{t+k})||^2, rolled out from
           detached context latents, so the representation comes from reconstruction alone and the
           dynamics are trained exactly as JEPA's are
  costonly the risk loss on real and rolled-out latents with gradients into E and P, no latent loss
  L_ID     beta * cross-entropy of the inverse-dynamics head on consecutive real latents   beta = 0.1
  L_C      cross-entropy of the violation-risk head on detached real and imagined latents
           (costonly: not detached, so the gradient shapes E and P: a value-equivalent model)
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .nets import WorldModel, count_params
from .sigreg import effective_rank, sigreg

VARIANTS = ("jepa", "lambda0", "ema", "recon", "costonly")
CONTEXT = 3
K = 8
WINDOW = CONTEXT + K


@dataclass
class TrainConfig:
    variant: str = "jepa"
    z_dim: int = 16
    lam: float = 0.1
    beta: float = 0.1
    slices: int = 256
    ema_tau: float = 0.99
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    batch: int = 256
    steps: int = 1200
    threads: int = 2
    seed: int = 0
    log_every: int = 50
    kill_minutes: float = 20.0

    def resolved_lambda(self) -> float:
        return {"jepa": self.lam, "lambda0": 0.0}.get(self.variant, 0.0)


class Windows:
    """Index of valid (episode, t) window starts over arrays shaped (episodes, T, ...)."""

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.obs = torch.from_numpy(np.ascontiguousarray(arrays["obs"], dtype=np.float32))
        self.u = torch.from_numpy(np.ascontiguousarray(arrays["actuator"], dtype=np.float32))
        self.a = torch.from_numpy(np.ascontiguousarray(arrays["action"]).astype(np.int64))
        self.c = torch.from_numpy(np.ascontiguousarray(arrays["violation"], dtype=np.float32))
        reset = np.asarray(arrays["reset"], dtype=bool)
        N, T = reset.shape
        starts = []
        for s in range(0, T - WINDOW + 1):
            ok = ~reset[:, s:s + WINDOW - 1].any(axis=1)
            starts.extend((e, s) for e in np.flatnonzero(ok))
        self.starts = np.array(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, idx: np.ndarray):
        e = torch.from_numpy(self.starts[idx, 0])
        s = torch.from_numpy(self.starts[idx, 1])
        off = s[:, None] + torch.arange(WINDOW)[None, :]
        ee = e[:, None].expand_as(off)
        return self.obs[ee, off], self.u[ee, off], self.a[ee, off], self.c[ee, off]


def losses(model: WorldModel, target_enc: Optional[torch.nn.Module], cfg: TrainConfig, o, u, a, c,
           gen: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    B = o.shape[0]
    v = cfg.variant
    z = model.encoder(o.reshape(B * WINDOW, -1)).reshape(B, WINDOW, -1)
    ctx = z[:, :CONTEXT].detach() if v == "recon" else z[:, :CONTEXT]
    z_hat = model.rollout(ctx, u[:, CONTEXT - 1:CONTEXT - 1 + K], a[:, CONTEXT - 1:CONTEXT - 1 + K])
    out: Dict[str, torch.Tensor] = {}
    if v in ("jepa", "lambda0"):
        out["pred"] = F.mse_loss(z_hat, z[:, CONTEXT:])
    elif v == "ema":
        with torch.no_grad():
            zt = target_enc(o[:, CONTEXT:].reshape(B * K, -1)).reshape(B, K, -1)
        out["pred"] = F.mse_loss(z_hat, zt)
    elif v == "recon":
        out["recon"] = F.mse_loss(model.decoder(z), o)
        out["pred"] = F.mse_loss(z_hat, z[:, CONTEXT:].detach())
    lam = cfg.resolved_lambda()
    sr = sigreg(z.transpose(0, 1), num_slices=cfg.slices, generator=gen)
    out["sigreg"] = sr if lam > 0 else sr.detach()
    logits = model.inverse(z[:, :-1].reshape(-1, z.shape[-1]), z[:, 1:].reshape(-1, z.shape[-1]))
    out["inverse"] = F.cross_entropy(logits, a[:, :-1].reshape(-1))
    zr = z if v == "costonly" else z.detach()
    zi = z_hat if v == "costonly" else z_hat.detach()
    depth = torch.arange(1, K + 1)
    real = F.binary_cross_entropy_with_logits(model.risk(zr, u, a, 0), c)
    imag = F.binary_cross_entropy_with_logits(model.risk(zi, u[:, CONTEXT:], a[:, CONTEXT:], depth), c[:, CONTEXT:])
    out["cost"] = real + imag
    total = out["cost"] + cfg.beta * out["inverse"]
    if "pred" in out:
        total = total + out["pred"]
    if lam > 0:
        total = total + lam * out["sigreg"]
    if v == "recon":
        total = total + out["recon"]
    out["total"] = total
    out["_z"] = z.detach()
    return out


def train(arrays: Dict[str, np.ndarray], cfg: TrainConfig, logger=None) -> Dict:
    """Train one model. ``logger(step, metrics)`` receives scalar metrics every ``log_every`` steps."""
    if cfg.variant not in VARIANTS:
        raise ValueError(f"unknown variant {cfg.variant!r}")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    win = Windows(arrays)
    model = WorldModel("recon" if cfg.variant == "recon" else cfg.variant, cfg.z_dim)
    target_enc = copy.deepcopy(model.encoder).requires_grad_(False) if cfg.variant == "ema" else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator()
    history = []
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        gen.manual_seed(cfg.seed * 1_000_003 + step)       # SIGReg directions reseeded every step
        o, u, a, c = win.batch(rng.integers(0, len(win), cfg.batch))
        out = losses(model, target_enc, cfg, o, u, a, c, gen)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        if target_enc is not None:
            with torch.no_grad():
                for pt, po in zip(target_enc.parameters(), model.encoder.parameters()):
                    pt.mul_(cfg.ema_tau).add_(po, alpha=1 - cfg.ema_tau)
        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            zf = out["_z"].reshape(-1, out["_z"].shape[-1])
            m = {k: float(v.detach()) for k, v in out.items() if not k.startswith("_")}
            m.update(erank=effective_rank(zf), z_std=float(zf.std(0).mean()), grad_norm=float(gnorm),
                     elapsed_s=time.perf_counter() - t0)
            history.append({"step": step, **m})
            if logger:
                logger(step, m)
        if (time.perf_counter() - t0) / 60.0 > cfg.kill_minutes:
            raise RuntimeError(f"wall-clock kill after {step} steps: run invalid")
        if not math.isfinite(float(out["total"].detach())):
            raise RuntimeError(f"non-finite loss at step {step}")
    return {"model": model, "history": history, "train_seconds": time.perf_counter() - t0,
            "windows": len(win), "params": count_params(model), "config": asdict(cfg)}


def save(model: WorldModel, cfg: TrainConfig, path: Path, extra: Optional[Dict] = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps({"train": asdict(cfg), **(extra or {})}, indent=2), encoding="utf-8")


def load(path: Path) -> WorldModel:
    meta = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tc = meta["train"]
    model = WorldModel("recon" if tc["variant"] == "recon" else tc["variant"], tc["z_dim"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model
