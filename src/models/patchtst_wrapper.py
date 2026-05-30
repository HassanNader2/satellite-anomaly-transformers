"""
PatchTST wrapper for reconstruction-based anomaly detection.

Adapted from forecasting to reconstruction by setting pred_len = seq_len.
RevIN disabled (revin=False) — conflicts with per-segment z-score normalization
already applied in data_loader.py.

Input to model: (B, seq_len, n_vars) — internally permuted to (B, n_vars, seq_len)
Output from model: (B, pred_len, n_vars) = (B, T, 1) for OPS-SAT-AD

Anomaly score: masked MSE per segment.
"""

import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn

_PATCHTST_SUPERVISED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "PatchTST", "PatchTST_supervised")
)
# Add to sys.path so PatchTST's internal imports (layers.*) resolve correctly
if _PATCHTST_SUPERVISED not in sys.path:
    sys.path.insert(0, _PATCHTST_SUPERVISED)

# Load PatchTST Model directly by path to avoid collision with our local `models/` package
_patchtst_spec = importlib.util.spec_from_file_location(
    "patchtst_model", os.path.join(_PATCHTST_SUPERVISED, "models", "PatchTST.py")
)
_patchtst_mod = importlib.util.module_from_spec(_patchtst_spec)
_patchtst_spec.loader.exec_module(_patchtst_mod)
Model = _patchtst_mod.Model


def _build_configs(p: dict):
    cfg = types.SimpleNamespace(
        enc_in=p["enc_in"],
        seq_len=p["seq_len"],
        pred_len=p["seq_len"],        # reconstruction: predict full sequence
        e_layers=p.get("e_layers", 2),
        n_heads=p.get("n_heads", 4),
        d_model=p.get("d_model", 128),
        d_ff=p.get("d_ff", 256),
        dropout=p.get("dropout", 0.1),
        fc_dropout=p.get("fc_dropout", 0.1),
        head_dropout=p.get("head_dropout", 0.0),
        individual=p.get("individual", False),
        patch_len=p.get("patch_len", 16),
        stride=p.get("stride", 8),
        padding_patch=p.get("padding_patch", "end"),
        revin=False,                  # MUST be False — we pre-normalize per segment
        affine=p.get("affine", False),
        subtract_last=p.get("subtract_last", False),
        decomposition=p.get("decomposition", False),
        kernel_size=p.get("kernel_size", 25),
    )
    return cfg


class PatchTSTWrapper(nn.Module):
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
        reconstruction = self.model(x)  # (B, pred_len=T, 1)
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
        reconstruction = self.model(x)
        loss = self._masked_mse(reconstruction, x, mask)
        return loss

    def compute_val_loss(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            reconstruction = self.model(x)
        return self._masked_mse(reconstruction, x, mask)

    @torch.no_grad()
    def anomaly_score(self, x, mask=None):
        """
        Returns per-segment anomaly score as a 1-D tensor of shape (B,).
        Score = masked mean MSE per segment.
        """
        self.eval()  # ensure BatchNorm uses running stats, not batch stats
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction = self.model(x)
        loss = self.criterion(reconstruction, x).squeeze(-1)  # (B, T)
        if mask is not None:
            m = mask.float()
            scores = (loss * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            scores = loss.mean(dim=1)
        return scores  # (B,)
