"""
SMAP and MSL data loader for secondary validation (Stage 8).

Format:
  SMAP: 54 unique channel files, each (T, 25) — enc_in=25
  MSL:  27 channel files, each (T, 55) — enc_in=55
  Labels: labeled_anomalies.csv — anomaly_sequences are 0-indexed test-set ranges.

Design:
  - Normalise using per-feature mean/std computed from training data.
  - Training: non-overlapping windows of size W (stride=W). No padding.
  - Scoring: stride=1 windows over test data → one score per timestep.
  - No attention mask needed (all windows are full — no padding).
  - Threshold tuning: first val_frac of TEST data (with labels).
  - Final evaluation: remaining (1 - val_frac) of TEST data.

Usage:
  from data_loader_smap_msl import load_smap_msl

  channels = load_smap_msl(
      base_dir = 'papers/satellite-anomaly/data/smap',
      spacecraft = 'SMAP',
      win_size = 100,
      val_frac = 0.20,
  )
  # channels: list of dicts, one per unique channel
  # Each dict: {
  #   'chan_id':        str,
  #   'enc_in':         int,
  #   'train_dataset':  SlidingWindowDataset,   # non-overlapping
  #   'val_scores_np':  np.ndarray (T_val,),    # for threshold tuning (scoring mode)
  #   'test_scores_np': np.ndarray (T_test,),   # for final evaluation
  #   'val_labels':     np.ndarray (T_val,) int,
  #   'test_labels':    np.ndarray (T_test,) int,
  #   'val_data':       np.ndarray (T_val, enc_in),   # normalised
  #   'test_data':      np.ndarray (T_test, enc_in),  # normalised
  # }
  # (val_scores_np and test_scores_np are filled in by the experiment runner.)
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# P-2 appears twice in labeled_anomalies.csv with different anomaly windows.
# The duplicate is the second occurrence — skip it.
_SKIP_DUPLICATES = {"P-2"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Non-overlapping sliding window dataset for training.
    x: (W, enc_in) float32
    No mask — all windows are full.
    """

    def __init__(self, data: np.ndarray, win_size: int):
        """
        data: (T, enc_in) normalised float32
        win_size: window length W
        """
        self.data     = data.astype(np.float32)
        self.win_size = win_size
        self.n_windows = len(data) // win_size

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.win_size
        x = self.data[start : start + self.win_size]
        return torch.from_numpy(x)   # (W, enc_in)


# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------

def _build_labels(anomaly_sequences_str: str, n_timesteps: int) -> np.ndarray:
    """
    Convert 'anomaly_sequences' string from CSV to a binary label array.
    Labels are 1-indexed in the CSV (anomaly_sequences are [start, end) ranges).
    """
    labels = np.zeros(n_timesteps, dtype=np.int32)
    seqs   = ast.literal_eval(anomaly_sequences_str)
    for start, end in seqs:
        labels[start:end] = 1
    return labels


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_smap_msl(base_dir: str, spacecraft: str, win_size: int = 100,
                  val_frac: float = 0.20) -> list:
    """
    Load all channels for a given spacecraft ('SMAP' or 'MSL').

    Returns a list of channel dicts (see module docstring).
    """
    labels_csv = pd.read_csv(os.path.join(base_dir, "labeled_anomalies.csv"))
    labels_csv = labels_csv[labels_csv["spacecraft"] == spacecraft].reset_index(drop=True)

    train_dir = os.path.join(base_dir, "train")
    test_dir  = os.path.join(base_dir, "test")

    seen = set()
    channels = []

    for _, row in labels_csv.iterrows():
        chan_id = row["chan_id"]

        # Skip duplicate entries (P-2 appears twice)
        if chan_id in seen and chan_id in _SKIP_DUPLICATES:
            continue
        seen.add(chan_id)

        train_path = os.path.join(train_dir, f"{chan_id}.npy")
        test_path  = os.path.join(test_dir,  f"{chan_id}.npy")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"  [skip] {chan_id} — file missing")
            continue

        train_raw = np.load(train_path).astype(np.float64)  # (T_train, enc_in)
        test_raw  = np.load(test_path).astype(np.float64)   # (T_test,  enc_in)

        enc_in = train_raw.shape[1]

        # Normalise: per-feature mean/std from training data
        mean = train_raw.mean(axis=0, keepdims=True)
        std  = train_raw.std(axis=0, keepdims=True)
        std  = np.where(std == 0, 1.0, std)    # avoid division by zero on constant channels

        train_norm = ((train_raw - mean) / std).astype(np.float32)
        test_norm  = ((test_raw  - mean) / std).astype(np.float32)

        # Build full test labels
        full_labels = _build_labels(row["anomaly_sequences"], len(test_norm))

        # Split test into val (first val_frac) and eval (rest)
        val_end = int(len(test_norm) * val_frac)
        # Ensure val_end falls on a window boundary for consistent scoring
        val_end = max(win_size, (val_end // win_size) * win_size)

        val_data   = test_norm[:val_end]
        test_data  = test_norm[val_end:]
        val_labels = full_labels[:val_end]
        test_labels = full_labels[val_end:]

        train_ds = SlidingWindowDataset(train_norm, win_size)

        channels.append({
            "chan_id":       chan_id,
            "enc_in":        enc_in,
            "train_dataset": train_ds,
            "val_data":      val_data,    # (T_val, enc_in) normalised
            "test_data":     test_data,   # (T_test_eval, enc_in) normalised
            "val_labels":    val_labels,
            "test_labels":   test_labels,
            # Placeholders — filled by experiment runner after scoring
            "val_scores":    None,
            "test_scores":   None,
        })

    print(f"  Loaded {len(channels)} {spacecraft} channels "
          f"(enc_in={channels[0]['enc_in'] if channels else '?'}, "
          f"win_size={win_size})")
    return channels


# ---------------------------------------------------------------------------
# Sliding window scorer helper (used by experiment runner)
# ---------------------------------------------------------------------------

def score_continuous(model_wrapper, data: np.ndarray, win_size: int,
                     batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """
    Compute per-timestep reconstruction MSE on a continuous time series.

    Slides a window of size win_size with stride=1 over data.
    Score for timestep t is the mean reconstruction MSE of the window
    ending at t (i.e. window data[t - win_size + 1 : t + 1]).

    First win_size - 1 timesteps receive the score of the first window.

    data: (T, enc_in) normalised numpy array
    Returns: scores (T,) numpy array
    """
    model_wrapper.eval()
    T, enc_in = data.shape
    n_windows = T - win_size + 1

    all_scores = []

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end   = min(start + batch_size, n_windows)
            batch = np.stack([data[i : i + win_size] for i in range(start, end)])
            x     = torch.from_numpy(batch).float().to(device)  # (B, W, enc_in)

            # Get reconstruction
            wrapper_key = getattr(model_wrapper, "_wrapper_key", None)
            if wrapper_key == "at":
                recon, _, _ = model_wrapper(x)
            elif wrapper_key == "itransformer":
                recon = model_wrapper.model(x, None, None, None)
                if isinstance(recon, (list, tuple)):
                    recon = recon[0]
            else:
                recon = model_wrapper.model(x)

            # MSE per window: mean over timesteps and features → (B,)
            mse = ((recon - x) ** 2).mean(dim=(1, 2))
            all_scores.append(mse.cpu().numpy())

    scores_per_window = np.concatenate(all_scores)  # (n_windows,)

    # Pad first win_size - 1 timesteps with the first window's score
    scores = np.empty(T)
    scores[:win_size - 1] = scores_per_window[0]
    scores[win_size - 1:] = scores_per_window

    return scores
