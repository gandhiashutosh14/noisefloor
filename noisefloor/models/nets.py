"""Network modules shared by every world model.

  Encoder E        one telemetry frame 128 -> 256 -> 256 -> z (16), LayerNorm + GELU
  Predictor P      [z_{t-2}, z_{t-1}, z_t, u_t, onehot a_t] (61) -> 128 -> 128 -> delta z
  RiskHead R       [z, u, onehot a, onehot k] (38) -> 128 -> logit of an SLO violation during the step,
                   where k = 0 for an encoded frame and k = 1..8 for a latent imagined k steps ahead
  InverseDynamics  [z_t, z_{t+1}] (32) -> 128 -> logits over the 7 actions
  Decoder D        z -> 256 -> 256 -> 128 (reconstruction baseline only)

The step cost splits into a known part (replicas, shed traffic, warms, failovers: exact functions of
the actuator state and the action) and the violation penalty, which depends on the hidden load and
health. Only the hidden part is learned: expected cost = known + 5 * sigmoid(R(z, u, a)). A regression
head on the whole cost trained with a robust loss behaves like a median and predicts rare, expensive
violations as cheap, which a planner then exploits.

The risk head is told how deep in an imagined rollout its latent is. Deterministic rollouts regress
toward typical states, and a single risk function evaluated on them underestimates the violation risk
of the future they stand for (the risk is nonlinear, so risk(E[z]) < E[risk(z)]). Conditioning on the
depth lets each depth be calibrated separately against what actually happened k steps later.

The encoder sees a single frame on purpose: with stacked frames the prediction target would share
input with the context and i.i.d. distractors would become partly predictable. History lives in the
predictor, which reads the last three latents.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

OBS_DIM = 128
ACT_DIM = 6
N_ACTIONS = 7
HISTORY = 3
MAX_DEPTH = 8


def mlp(sizes, norm: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, z_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.net = mlp([OBS_DIM, hidden, hidden, z_dim], norm=True)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.net(o)


class Predictor(nn.Module):
    def __init__(self, z_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.net = mlp([HISTORY * z_dim + ACT_DIM + N_ACTIONS, hidden, hidden, z_dim])

    def forward(self, zhist: torch.Tensor, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """zhist (B, 3, z), u (B, 6), a (B,) long -> next latent z_{t+1} = z_t + delta."""
        x = torch.cat([zhist.flatten(1), u, F.one_hot(a, N_ACTIONS).to(zhist.dtype)], dim=1)
        return zhist[:, -1] + self.net(x)


class RiskHead(nn.Module):
    """Logit of P(SLO violation during the step | z, u, a, rollout depth k)."""

    def __init__(self, z_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.net = mlp([z_dim + ACT_DIM + N_ACTIONS + MAX_DEPTH + 1, hidden, 1])

    def forward(self, z: torch.Tensor, u: torch.Tensor, a: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """z (..., z_dim), u (..., 6), a (...) long, depth broadcastable to a's shape (long, 0..MAX_DEPTH)."""
        depth = torch.as_tensor(depth, dtype=torch.long).expand(a.shape)
        x = torch.cat([z, u, F.one_hot(a, N_ACTIONS).to(z.dtype), F.one_hot(depth, MAX_DEPTH + 1).to(z.dtype)], dim=-1)
        return self.net(x).squeeze(-1)


class InverseDynamics(nn.Module):
    def __init__(self, z_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.net = mlp([2 * z_dim, hidden, N_ACTIONS])

    def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z0, z1], dim=-1))


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.net = mlp([z_dim, hidden, hidden, OBS_DIM], norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorldModel(nn.Module):
    """Container for one trained model; ``variant`` records how it was trained."""

    def __init__(self, variant: str, z_dim: int = 16):
        super().__init__()
        self.variant, self.z_dim = variant, z_dim
        self.encoder = Encoder(z_dim)
        self.predictor = Predictor(z_dim)
        self.risk = RiskHead(z_dim)
        self.inverse = InverseDynamics(z_dim)
        self.decoder = Decoder(z_dim) if variant == "recon" else None

    def rollout(self, zhist: torch.Tensor, u_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """zhist (B, 3, z); u_seq (B, H, 6); a_seq (B, H) -> imagined latents (B, H, z) for steps 1..H."""
        hist = zhist
        out = []
        for k in range(a_seq.shape[1]):
            z_next = self.predictor(hist, u_seq[:, k], a_seq[:, k])
            out.append(z_next)
            hist = torch.cat([hist[:, 1:], z_next[:, None]], dim=1)
        return torch.stack(out, dim=1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
