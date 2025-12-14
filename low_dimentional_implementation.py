# ree_pytorch.py
# Minimal REE scaffold:
# - E1: deep recurrent predictor (long-horizon, multi-step)
# - E2: fast feedforward predictor (short-horizon, single-step)
# - Fuse into L-space latent z_t
# - E3: generate candidate latent trajectories, score coherence, select one
#
# Based on REE descriptions by Daniel De La Harpe Golden in:
# - Synthese manuscript (three constraints: fast prediction, deep temporal synthesis, constrained selection)
# - Defensive publication (E1/E2 fusion into temporally-displaced manifold; E3 trajectory selection)
#
# Author of this file: ChatGPT (implementation scaffold)
# 

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0) -> nn.Module:
    layers = []
    d = in_dim
    for _ in range(max(depth - 1, 0)):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


@dataclass
class REEConfig:
    state_dim: int
    action_dim: int
    latent_dim: int = 128
    e1_hidden_dim: int = 256
    e2_hidden_dim: int = 128
    horizon_K: int = 8          # E1 predicts steps 2..K
    num_candidates: int = 32    # E3 candidates per step
    traj_len: int = 8           # rollout length for E3 selection
    temperature: float = 1.0    # sampling temperature for candidate noise
    coherence_w: float = 1.0    # weight for coherence term
    value_w: float = 0.0        # optional value modulation
    mod_w: float = 0.0          # optional modulatory term


# -----------------------------
# Encoders / Decoders
# -----------------------------

class StateActionEncoder(nn.Module):
    """Encodes (s_t, a_t) into a shared latent embedding."""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, hidden_dim, latent_dim, depth=3)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class StateDecoder(nn.Module):
    """Decodes a latent into predicted next-state (or future-state)."""
    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(latent_dim, hidden_dim, state_dim, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ActionDecoder(nn.Module):
    """Decodes an action from a latent state (can implement delay compensation later)."""
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(latent_dim, hidden_dim, action_dim, depth=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -----------------------------
# E2: Fast feedforward predictor (single-step)
# -----------------------------

class E2FastPredictor(nn.Module):
    """
    E2: fast, low-latency single-step predictor.
    Given (s_t, a_t) -> latent u2 and predicted next-state.
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.enc = StateActionEncoder(cfg.state_dim, cfg.action_dim, cfg.latent_dim, cfg.e2_hidden_dim)
        self.dec_next = StateDecoder(cfg.latent_dim, cfg.state_dim, cfg.e2_hidden_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u2 = self.enc(s, a)
        s_next_hat = self.dec_next(u2)
        return u2, s_next_hat


# -----------------------------
# E1: Deep recurrent predictor (multi-step, long-horizon)
# -----------------------------

class E1DeepRecurrentPredictor(nn.Module):
    """
    E1: deep temporal synthesis / world-model-like recurrent predictor.
    Uses a GRU (Gated Recurrent Unit) for simplicity (swap for LSTM (Long Short-Term Memory) or Transformer later).
    Produces:
      - hidden state h_{t}
      - multi-step future state predictions (k=2..K)
      - latent summary u1
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.cfg = cfg
        self.in_enc = StateActionEncoder(cfg.state_dim, cfg.action_dim, cfg.latent_dim, cfg.e1_hidden_dim)
        self.rnn = nn.GRU(input_size=cfg.latent_dim, hidden_size=cfg.e1_hidden_dim, batch_first=True)
        self.to_u1 = nn.Linear(cfg.e1_hidden_dim, cfg.latent_dim)

        # Predict future states for steps 2..K from the recurrent hidden
        self.future_heads = nn.ModuleList([
            mlp(cfg.e1_hidden_dim, cfg.e1_hidden_dim, cfg.state_dim, depth=2)
            for _ in range(max(cfg.horizon_K - 1, 1))  # corresponds to k=2..K (K-1 heads)
        ])

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          s: [B, state_dim]
          a: [B, action_dim]
          h_prev: [1, B, hidden_dim] or None
        Returns:
          h: [1, B, hidden_dim]
          u1: [B, latent_dim]
          future_state_hats: [B, (K-1), state_dim] for k=2..K
        """
        B = s.size(0)
        x = self.in_enc(s, a).unsqueeze(1)  # [B, 1, latent_dim]
        out, h = self.rnn(x, h_prev)        # out: [B, 1, hidden_dim], h: [1, B, hidden_dim]
        ht = out[:, 0, :]                   # [B, hidden_dim]
        u1 = self.to_u1(ht)                 # [B, latent_dim]

        preds = []
        for head in self.future_heads:
            preds.append(head(ht))          # each [B, state_dim]
        future_state_hats = torch.stack(preds, dim=1)  # [B, (K-1), state_dim]
        return h, u1, future_state_hats


# -----------------------------
# Fusion: combine E1 and E2 into unified latent manifold z_t
# -----------------------------

class LatentFusion(nn.Module):
    """
    Implements a learned fusion operator F_fuse(u1, u2) -> z_t.
    You can replace with cross-attention, gating, etc.
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.gate = mlp(cfg.latent_dim * 2, cfg.latent_dim, cfg.latent_dim, depth=2)
        self.mix = mlp(cfg.latent_dim * 2, cfg.latent_dim, cfg.latent_dim, depth=2)

    def forward(self, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u1, u2], dim=-1)
        g = torch.sigmoid(self.gate(x))
        m = self.mix(x)
        z = g * u1 + (1.0 - g) * u2 + 0.1 * m
        return z


# -----------------------------
# E3: Trajectory selection
# -----------------------------

class TrajectoryProposal(nn.Module):
    """
    Proposes candidate latent trajectories from current z_t.
    Here: simple latent dynamics + Gaussian noise proposals.
    Replace with diffusion sampling / MCTS / learned rollout policy later.
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.cfg = cfg
        self.dyn = mlp(cfg.latent_dim, cfg.latent_dim, cfg.latent_dim, depth=3)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Returns candidates:
          candidates: [B, N, T, latent_dim]
        """
        B = z0.size(0)
        N = self.cfg.num_candidates
        T = self.cfg.traj_len

        z = z0[:, None, None, :].expand(B, N, 1, self.cfg.latent_dim).contiguous()
        traj = [z[:, :, 0, :]]

        for _ in range(1, T):
            prev = traj[-1]
            step = self.dyn(prev)
            noise = torch.randn_like(step) * (self.cfg.temperature / max(T, 1) ** 0.5)
            nxt = prev + step + noise
            traj.append(nxt)

        candidates = torch.stack(traj, dim=2)  # [B, N, T, latent_dim]
        return candidates


class CoherenceScorer(nn.Module):
    """
    Scores a candidate trajectory for:
      - internal smoothness (temporal coherence)
      - compatibility with E2 short-horizon grounding
      - optional modulatory terms (value, ethics, safety) as in M(gamma)
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.cfg = cfg
        self.value_head = mlp(cfg.latent_dim, cfg.latent_dim, 1, depth=2)  # optional V(gamma)
        self.mod_head = mlp(cfg.latent_dim, cfg.latent_dim, 1, depth=2)    # optional M(gamma)

    def forward(self, candidates: torch.Tensor, z_e2_anchor: torch.Tensor) -> torch.Tensor:
        """
        Args:
          candidates: [B, N, T, D]
          z_e2_anchor: [B, D] (fast grounded latent from E2)
        Returns:
          score: [B, N]
        """
        B, N, T, D = candidates.shape

        # 1) Temporal smoothness: penalize big jumps
        diffs = candidates[:, :, 1:, :] - candidates[:, :, :-1, :]         # [B, N, T-1, D]
        smooth_pen = diffs.pow(2).mean(dim=(-1, -2))                       # [B, N]

        # 2) Grounding: keep early trajectory close to E2 anchor (short-horizon coupling)
        # Compare first step (or mean of first few steps)
        early = candidates[:, :, 0, :]                                     # [B, N, D]
        anchor = z_e2_anchor[:, None, :].expand(B, N, D)
        anchor_pen = (early - anchor).pow(2).mean(dim=-1)                  # [B, N]

        # 3) Optional value and modulation (learned; start at 0 weight unless you set cfg.value_w/mod_w)
        # Use final state as trajectory summary
        zT = candidates[:, :, -1, :]                                       # [B, N, D]
        V = self.value_head(zT).squeeze(-1)                                # [B, N]
        M = self.mod_head(zT).squeeze(-1)                                  # [B, N]

        # Convert penalties to coherence "score" (higher is better)
        coherence = -(smooth_pen + anchor_pen)

        score = self.cfg.coherence_w * coherence + self.cfg.value_w * V + self.cfg.mod_w * M
        return score


class E3Selector(nn.Module):
    """Generate candidate trajectories, score them, and select one."""
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.cfg = cfg
        self.proposer = TrajectoryProposal(cfg)
        self.scorer = CoherenceScorer(cfg)

    def forward(self, z_t: torch.Tensor, z_e2_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
          selected_traj: [B, T, D]
          selected_z: [B, D] (often last point, or you can use a specific offset for delay compensation)
          scores: [B, N]
          idx: [B]
        """
        candidates = self.proposer(z_t)                    # [B, N, T, D]
        scores = self.scorer(candidates, z_e2_anchor)      # [B, N]
        idx = torch.argmax(scores, dim=1)                  # [B]

        B, N, T, D = candidates.shape
        sel = candidates[torch.arange(B), idx]             # [B, T, D]
        selected_z = sel[:, -1, :]                         # [B, D] (choose other point if needed)

        return {
            "selected_traj": sel,
            "selected_z": selected_z,
            "scores": scores,
            "idx": idx,
        }


# -----------------------------
# REE: Full model
# -----------------------------

class REE(nn.Module):
    """
    Full REE:
      - E2 for grounded short-horizon prediction
      - E1 for long-horizon multi-step prediction + recurrent state
      - Fuse u1/u2 into z_t (L-space point)
      - E3 selects a trajectory; selected latent is decoded into action + (optionally) predicted state
    """
    def __init__(self, cfg: REEConfig):
        super().__init__()
        self.cfg = cfg
        self.e2 = E2FastPredictor(cfg)
        self.e1 = E1DeepRecurrentPredictor(cfg)
        self.fuse = LatentFusion(cfg)
        self.e3 = E3Selector(cfg)

        self.state_dec = StateDecoder(cfg.latent_dim, cfg.state_dim, cfg.latent_dim)
        self.action_dec = ActionDecoder(cfg.latent_dim, cfg.action_dim, cfg.latent_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.cfg.e1_hidden_dim, device=device)

    def forward(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # E2
        u2, s1_hat_e2 = self.e2(s_t, a_t)

        # E1
        h, u1, future_hats_e1 = self.e1(s_t, a_t, h_prev=h_prev)

        # Fuse into L-space point
        z_t = self.fuse(u1, u2)

        # E3 trajectory selection (coherence-biased)
        sel = self.e3(z_t, z_e2_anchor=u2)
        z_sel = sel["selected_z"]

        # Decode action and predicted next state from selected latent
        a_hat = self.action_dec(z_sel)
        s_hat = self.state_dec(z_sel)

        return {
            "h": h,
            "u1": u1,
            "u2": u2,
            "z_t": z_t,
            "e2_s1_hat": s1_hat_e2,
            "e1_future_hats": future_hats_e1,   # k=2..K
            "selected_traj": sel["selected_traj"],
            "selected_z": z_sel,
            "scores": sel["scores"],
            "selected_idx": sel["idx"],
            "a_hat": a_hat,
            "s_hat": s_hat,
        }


# -----------------------------
# Losses and a simple training step
# -----------------------------

def ree_loss(
    out: Dict[str, torch.Tensor],
    s_tp1: torch.Tensor,
    s_future: Optional[torch.Tensor] = None,
    w_e2: float = 1.0,
    w_e1: float = 0.5,
    w_fuse_align: float = 0.1,
    w_sel_pred: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Args:
      out: model outputs from REE.forward
      s_tp1: ground truth next state [B, state_dim]
      s_future: optional ground truth future states for k=2..K [B, (K-1), state_dim]
    """
    losses = {}

    # E2 one-step prediction loss
    losses["e2"] = F.mse_loss(out["e2_s1_hat"], s_tp1)

    # E1 multi-step loss (if provided)
    if s_future is not None:
        losses["e1"] = F.mse_loss(out["e1_future_hats"], s_future)
    else:
        losses["e1"] = torch.zeros((), device=s_tp1.device)

    # Fusion alignment encourages u1 and u2 to be mutually informative (softly)
    losses["fuse_align"] = F.mse_loss(out["u1"], out["u2"])

    # Selected latent prediction loss (selected trajectory should remain grounded)
    losses["sel_pred"] = F.mse_loss(out["s_hat"], s_tp1)

    total = w_e2 * losses["e2"] + w_e1 * losses["e1"] + w_fuse_align * losses["fuse_align"] + w_sel_pred * losses["sel_pred"]
    losses["total"] = total
    return losses


@torch.no_grad()
def offline_replay_stub(buffer: torch.Tensor) -> None:
    """
    Placeholder for offline consolidation modes:
      - replay
      - denoising
      - expansion
      - consolidation
    In practice you'd:
      - sample stored trajectories
      - train with different lr / objectives
      - add noise then denoise (autoencoding / consistency)
      - do recombination for expansion
    """
    _ = buffer


# -----------------------------
# Minimal usage example
# -----------------------------

def demo_step(device: str = "cpu") -> None:
    torch.manual_seed(0)
    dev = torch.device(device)

    cfg = REEConfig(state_dim=16, action_dim=4, latent_dim=64, horizon_K=6, num_candidates=16, traj_len=6)
    model = REE(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 32
    s_t = torch.randn(B, cfg.state_dim, device=dev)
    a_t = torch.randn(B, cfg.action_dim, device=dev)

    # Fake dynamics target for demo: next state is a noisy linear function
    W = torch.randn(cfg.action_dim, cfg.state_dim, device=dev) * 0.1
    s_tp1 = s_t + a_t @ W + 0.01 * torch.randn_like(s_t)

    # Fake future for k=2..K (optional)
    s_future = torch.stack([s_tp1 + (k * 0.01) * torch.randn_like(s_tp1) for k in range(1, cfg.horizon_K)], dim=1)

    h0 = model.init_hidden(B, dev)
    out = model(s_t, a_t, h_prev=h0)
    losses = ree_loss(out, s_tp1=s_tp1, s_future=s_future)

    opt.zero_grad(set_to_none=True)
    losses["total"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    print({k: float(v.detach().cpu()) for k, v in losses.items()})


if __name__ == "__main__":
    demo_step("cpu")
