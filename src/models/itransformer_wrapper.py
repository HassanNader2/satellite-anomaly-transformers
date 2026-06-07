"""
iTransformer wrapper for reconstruction-based anomaly detection.

Inverted architecture: variates are tokens, not timesteps.
On OPS-SAT-AD (enc_in=1): degenerate case — 1 variate token, cross-channel
attention nullified. Result documented as architecture-format fit finding.
On SMAP (55ch) / MSL (27ch): full operation.

Internal normalization disabled (use_norm=False) — conflicts with per-segment
z-score already applied in data_loader.py.

Forward signature: model(x_enc, x_mark_enc, x_dec, x_mark_dec)
  - x_enc: (B, seq_len, enc_in)
  - All mark and decoder inputs: None
  - output_attention=False → returns (B, pred_len, enc_in) directly

Anomaly score: masked MSE per segment.
"""

import os
import sys
import types
import torch
import torch.nn as nn

_IT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "iTransformer")
)
if _IT_PATH not in sys.path:
    sys.path.insert(0, _IT_PATH)

from model.iTransformer import Model


def _build_configs(p: dict):
    return types.SimpleNamespace(
        seq_len=p["seq_len"],
        pred_len=p["seq_len"],         # reconstruction: predict full sequence
        enc_in=p["enc_in"],
        d_model=p.get("d_model", 64),
        n_heads=p.get("n_heads", 4),
        e_layers=p.get("e_layers", 2),
        d_ff=p.get("d_ff", 128),
        dropout=p.get("dropout", 0.1),
        activation=p.get("activation", "gelu"),
        output_attention=False,        # simplify output — no attention tuple
        use_norm=False,                # MUST be False — we pre-normalize per segment
        embed=p.get("embed", "timeF"),
        freq=p.get("freq", "h"),
        factor=p.get("factor", 1),
        class_strategy=p.get("class_strategy", "projection"),
    )


class iTransformerWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        p = config["model_params"]
        self.T = p["seq_len"]
        cfg = _build_configs(p)
        self.model = Model(cfg)
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x, mask=None):
        """
        x: (B, T) or (B, T, 1)
        Returns: reconstruction (B, T, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)   # (B, T, 1)
        reconstruction = self.model(x, None, None, None)
        return reconstruction

    def _masked_mse(self, reconstruction, x, mask):
        loss = self.criterion(reconstruction, x)  # (B, T, 1)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (loss * m).sum() / m.sum().clamp(min=1)
        return loss.mean()

    def compute_train_loss(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction = self.model(x, None, None, None)
        return self._masked_mse(reconstruction, x, mask)

    def compute_val_loss(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            reconstruction = self.model(x, None, None, None)
        return self._masked_mse(reconstruction, x, mask)

    @torch.no_grad()
    def anomaly_score(self, x, mask=None):
        """
        Returns per-segment anomaly score as a 1-D tensor of shape (B,).
        Score = masked mean MSE per segment. Scores are clipped to
        min(99.9th percentile, 1e6) to guard against numerical explosions
        on datasets with many near-zero-variance channels (e.g. SMAP).
        """
        self.eval()  # Use running stats for BatchNorm, disable dropout
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction = self.model(x, None, None, None)
        loss = self.criterion(reconstruction, x).squeeze(-1)  # (B, T)
        if mask is not None:
            m = mask.float()
            scores = (loss * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            scores = loss.mean(dim=1)
        # clip outlier scores before returning
        if scores.numel() > 1:
            p999 = torch.quantile(scores, 0.999).item()
            scores = scores.clamp(max=min(p999, 1e6))
        return scores  # (B,)
