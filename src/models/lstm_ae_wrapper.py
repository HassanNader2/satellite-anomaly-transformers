"""
LSTM autoencoder wrapper for reconstruction-based anomaly detection.

Architecture: encoder-decoder LSTM with context-vector bottleneck.
  Encoder: 2-layer LSTM processes (B, T, 1) -> context vector (h_n, c_n)
  Decoder: 2-layer LSTM initialized with encoder state, zero input, reconstructs sequence
  Output:  linear projection from hidden_size to 1

Anomaly score: masked mean MSE per segment, identical to PatchTST/iTransformer scoring.
Same preprocessing pipeline (min_len=16, T=512, zero-padding, per-segment z-score).
"""

import torch
import torch.nn as nn


class LSTMAEWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        p = config["model_params"]
        self.T = p["seq_len"]
        hidden_size = p.get("hidden_size", 128)
        num_layers = p.get("num_layers", 2)
        # nn.LSTM requires dropout=0 when num_layers==1
        dropout = p.get("dropout", 0.1) if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss(reduction="none")

    def _reconstruct(self, x, mask=None):
        """
        x: (B, T, 1) — already unsqueezed.
        mask: (B, T) boolean — encoder processes only real (non-padded) timesteps.
        Returns reconstruction (B, T, 1).
        """
        B, T, _ = x.shape
        if mask is not None:
            lengths = mask.sum(dim=1).long().cpu().clamp(min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, c_n) = self.encoder(packed)
        else:
            _, (h_n, c_n) = self.encoder(x)
        dec_input = torch.zeros(B, T, 1, device=x.device)
        dec_out, _ = self.decoder(dec_input, (h_n, c_n))
        return self.output_proj(dec_out)

    def _masked_mse(self, reconstruction, x, mask):
        loss = self.criterion(reconstruction, x)  # (B, T, 1)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (loss * m).sum() / m.sum().clamp(min=1)
        return loss.mean()

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self._reconstruct(x, mask)

    def compute_train_loss(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction = self._reconstruct(x, mask)
        return self._masked_mse(reconstruction, x, mask)

    def compute_val_loss(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        with torch.no_grad():
            reconstruction = self._reconstruct(x, mask)
        return self._masked_mse(reconstruction, x, mask)

    @torch.no_grad()
    def anomaly_score(self, x, mask=None):
        """
        Returns per-segment anomaly score as a 1-D tensor of shape (B,).
        Score = masked mean MSE per segment (same as PatchTST and iTransformer).
        Clipped to guard against numerical explosions.
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        reconstruction = self._reconstruct(x, mask)
        loss = self.criterion(reconstruction, x).squeeze(-1)  # (B, T)
        if mask is not None:
            m = mask.float()
            scores = (loss * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            scores = loss.mean(dim=1)
        # Clip outliers for numerical stability
        if scores.numel() > 1:
            p999 = torch.quantile(scores, 0.999).item()
            scores = scores.clamp(max=min(p999, 1e6))
        return scores  # (B,)
