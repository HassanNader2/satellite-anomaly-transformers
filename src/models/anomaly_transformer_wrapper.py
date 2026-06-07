"""
Anomaly Transformer wrapper for OPS-SAT-AD reconstruction-based anomaly detection.

Training: minimax association discrepancy criterion (xu2022, solver.py).
  loss1 = MSE - k * KL(series, prior_normalized)   [minimize series divergence]
  loss2 = MSE + k * KL(prior_normalized, series)    [maximize prior divergence]
  Both losses backpropagated in one optimizer step per batch.

Anomaly score (OPS-SAT-AD): plain masked MSE over non-padded positions.
  The AT composite score (softmax(AssDis) * MSE) degrades on heavily-padded segments
  and is not used on OPS-SAT-AD. The composite score is restored in run_secondary.py
  for SMAP/MSL where padding is absent.

Input to model: (B, T, 1)
Output from model: (B, T, 1) reconstruction + series/prior attention lists
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_AT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "Anomaly-Transformer"))
if _AT_PATH not in sys.path:
    sys.path.insert(0, _AT_PATH)

from model.AnomalyTransformer import AnomalyTransformer


def _kl_loss(p, q):
    return torch.mean(torch.sum(p * (torch.log(p + 1e-4) - torch.log(q + 1e-4)), dim=-1), dim=1)


class AnomalyTransformerWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        p = config["model_params"]
        self.k = p.get("k", 3)
        self.temperature = p.get("temperature", 50)
        self.win_size = p["win_size"]

        self.model = AnomalyTransformer(
            win_size=p["win_size"],
            enc_in=p["enc_in"],
            c_out=p["c_out"],
            d_model=p.get("d_model", 64),
            n_heads=p.get("n_heads", 4),
            e_layers=p.get("e_layers", 2),
            d_ff=p.get("d_ff", 128),
            dropout=p.get("dropout", 0.0),
            activation=p.get("activation", "gelu"),
            output_attention=True,
        )
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x, mask=None):
        """
        x: (B, T) or (B, T, 1)
        Returns: reconstruction (B, T, 1), series list, prior list
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction, series, prior, _ = self.model(x)
        return reconstruction, series, prior

    def compute_train_loss(self, x, mask=None):
        """
        Returns loss1, loss2 for minimax training (association discrepancy).
        Caller does: combined_loss = loss1 + loss2; combined_loss.backward(); optimizer.step()
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        reconstruction, series, prior, _ = self.model(x)

        # Reconstruction loss: masked mean over non-padded positions
        rec_loss_per_step = self.criterion(reconstruction, x)  # (B, T, 1)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            rec_loss = (rec_loss_per_step * m).sum() / m.sum().clamp(min=1)
        else:
            rec_loss = rec_loss_per_step.mean()

        # Association discrepancy
        series_loss = torch.zeros(1, device=x.device)
        prior_loss = torch.zeros(1, device=x.device)
        for u in range(len(prior)):
            prior_sum = torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
            prior_sum = prior_sum.clamp(min=1e-8)
            prior_norm = prior[u] / prior_sum.repeat(1, 1, 1, self.win_size)
            series_loss = series_loss + (
                _kl_loss(series[u], prior_norm.detach()).mean() +
                _kl_loss(prior_norm.detach(), series[u]).mean()
            )
            prior_loss = prior_loss + (
                _kl_loss(prior_norm, series[u].detach()).mean() +
                _kl_loss(series[u].detach(), prior_norm).mean()
            )
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        del series, prior

        loss1 = rec_loss - self.k * series_loss
        loss2 = rec_loss + self.k * prior_loss
        return loss1, loss2

    def compute_warmup_loss(self, x, mask=None):
        """Reconstruction-only loss for warmup epochs (gradients enabled, no minimax)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction, _, _, _ = self.model(x)
        rec = self.criterion(reconstruction, x)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (rec * m).sum() / m.sum()
        return rec.mean()

    def compute_val_loss(self, x, mask=None):
        """MSE-only validation loss (no minimax — stable metric for early stopping)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x, mask)
        rec = self.criterion(reconstruction, x)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (rec * m).sum() / m.sum()
        return rec.mean()

    @torch.no_grad()
    def anomaly_score(self, x, mask=None):
        """
        Returns per-segment anomaly score as a 1-D tensor of shape (B,).
        Score = masked mean over timesteps of: softmax(-(series_kl + prior_kl)) * mse * temperature
        """
        self.eval()  # Use running stats for BatchNorm, disable dropout
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        reconstruction, series, prior, _ = self.model(x)
        B, T, _ = x.shape

        rec_loss = self.criterion(reconstruction, x).squeeze(-1)  # (B, T)

        del series, prior

        # Segment-level score: masked mean MSE over real (non-padded) positions.
        # The AT paper's composite score (softmax(AssDis) * MSE) and the raw association
        # discrepancy both degrade on OPS-SAT-AD's heavily-padded short segments — the
        # attention mechanism lacks sufficient real signal to produce reliable discrepancy
        # patterns at segment level. Plain masked MSE is consistent with PatchTST and
        # iTransformer and produces well-ordered scores from the AT's trained reconstruction.
        if mask is not None:
            m = mask.float()
            scores = (rec_loss * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            scores = rec_loss.mean(dim=1)

        # Clip outliers for numerical stability
        if scores.numel() > 1:
            p999 = torch.quantile(scores, 0.999).item()
            scores = scores.clamp(max=min(p999, 1e6))

        return scores  # (B,)
