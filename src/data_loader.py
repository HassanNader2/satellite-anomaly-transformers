"""
OPS-SAT-AD Dataset loader.

Preprocessing (locked 2026-05-14):
  - Filter segments shorter than min_len (default 16) — removes 66 segments (3.1%)
  - Per-segment z-score normalization (mean=0, std=1) before padding
  - Zero-pad segments in [min_len, T] to length T=512; boolean mask tracks real vs padded positions
  - Center-crop segments longer than T to length T — affects 69 segments (3.3%)
  - Stratified 80/20 train/val split from the official training set
  - Official test set used as-is

Input file: segments.csv columns: channel, timestamp, value, label, sampling, anomaly, segment, train
  - `anomaly` column: 1 = anomalous segment, 0 = nominal
  - `train` column: 1 = training set, 0 = test set
  - `segment` column: integer ID grouping timesteps belonging to the same segment
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class OPSSATDataset(Dataset):
    def __init__(self, segments, masks, labels):
        """
        segments: float32 tensor of shape (N, T)
        masks:    bool tensor of shape (N, T) — True = real timestep, False = padding
        labels:   int64 tensor of shape (N,)  — 1 = anomalous, 0 = nominal
        """
        self.segments = segments
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.masks[idx], self.labels[idx]


def load_opssat(
    csv_path: str,
    T: int = 512,
    min_len: int = 16,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Load and preprocess OPS-SAT-AD segments.csv.

    Returns a dict with keys: train, val, test — each an OPSSATDataset.
    Also returns split_counts with exact segment and anomaly counts per split.
    """
    df = pd.read_csv(csv_path)

    # Build segment-level table: one row per segment
    seg_table = (
        df.groupby("segment")
        .agg(
            values=("value", list),
            anomaly=("anomaly", "first"),
            is_train=("train", "first"),
        )
        .reset_index()
    )

    seg_table["length"] = seg_table["values"].apply(len)

    # Filter: remove segments shorter than min_len
    before = len(seg_table)
    seg_table = seg_table[seg_table["length"] >= min_len].reset_index(drop=True)
    filtered = before - len(seg_table)

    # Split into official train and test sets
    train_all = seg_table[seg_table["is_train"] == 1].reset_index(drop=True)
    test_df = seg_table[seg_table["is_train"] == 0].reset_index(drop=True)

    # Stratified val split from training set
    train_idx, val_idx = train_test_split(
        range(len(train_all)),
        test_size=val_frac,
        stratify=train_all["anomaly"].values,
        random_state=seed,
    )
    train_df = train_all.iloc[train_idx].reset_index(drop=True)
    val_df = train_all.iloc[val_idx].reset_index(drop=True)

    def preprocess(df_split):
        segs, masks, labels = [], [], []
        for _, row in df_split.iterrows():
            raw = np.array(row["values"], dtype=np.float32)

            # Center-crop segments longer than T
            if len(raw) > T:
                start = (len(raw) - T) // 2
                raw = raw[start : start + T]

            # Per-segment z-score normalization (before padding)
            std = raw.std()
            if std > 0:
                raw = (raw - raw.mean()) / std
            else:
                raw = raw - raw.mean()

            # Zero-pad to length T
            pad_len = T - len(raw)
            mask = np.ones(T, dtype=bool)
            if pad_len > 0:
                raw = np.concatenate([raw, np.zeros(pad_len, dtype=np.float32)])
                mask[T - pad_len :] = False

            segs.append(raw)
            masks.append(mask)
            labels.append(int(row["anomaly"]))

        return (
            torch.tensor(np.stack(segs), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.bool),
            torch.tensor(labels, dtype=torch.long),
        )

    train_segs, train_masks, train_labels = preprocess(train_df)
    val_segs, val_masks, val_labels = preprocess(val_df)
    test_segs, test_masks, test_labels = preprocess(test_df)

    split_counts = {
        "filtered_segments": filtered,
        "train_total": len(train_labels),
        "train_anomalous": int(train_labels.sum()),
        "train_nominal": int((train_labels == 0).sum()),
        "val_total": len(val_labels),
        "val_anomalous": int(val_labels.sum()),
        "val_nominal": int((val_labels == 0).sum()),
        "test_total": len(test_labels),
        "test_anomalous": int(test_labels.sum()),
        "test_nominal": int((test_labels == 0).sum()),
    }

    return {
        "train": OPSSATDataset(train_segs, train_masks, train_labels),
        "val": OPSSATDataset(val_segs, val_masks, val_labels),
        "test": OPSSATDataset(test_segs, test_masks, test_labels),
        "split_counts": split_counts,
    }
