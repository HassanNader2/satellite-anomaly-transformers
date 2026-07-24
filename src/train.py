"""
Shared training loop for all three model wrappers.

Training regime: unsupervised reconstruction. All training segments used
(nominal + anomalous) — models have no access to labels during training.

Anomaly Transformer: combined-loss surrogate per batch.
  combined_loss = loss1 + loss2; combined_loss.backward(); optimizer.step()
  Gradient-equivalent to the sequential two-backward form with a single optimizer.step().
  Neither form implements the true alternating-minimax of Xu et al. (two optimizer.step() calls).

PatchTST / iTransformer: standard single backward per batch.

Early stopping: patience on validation MSE (MSE-only for all models — stable metric).
"""

import os
import gc
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader


def train(wrapper, train_dataset, val_dataset, config, run_id, logger=None, checkpoints_dir=None):
    """
    Train a model wrapper.

    Returns path to the best checkpoint file.
    """
    if logger is None:
        logger = logging.getLogger("train")

    t = config["training"]
    epochs = t.get("epochs", 10)
    batch_size = t.get("batch_size", 32)
    lr = t.get("lr", 1e-4)
    patience = t.get("patience", 5)
    seed = t.get("seed", 42)
    warmup_epochs = t.get("warmup_epochs", 0)  # epochs of reconstruction-only before minimax

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    wrapper = wrapper.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    weight_decay = t.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr, weight_decay=weight_decay)
    # ReduceLROnPlateau on validation loss — identical rule across all runs.
    # Pairs with early stopping: halve LR after 5 stagnant epochs, floor at 1e-6.
    # threshold/threshold_mode are PyTorch defaults, set explicitly for reproducibility:
    # a relative improvement of >0.01% of best is required to count as progress.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-6,
    )

    # Early-stopping improvement criterion (identical rule across all runs):
    # an epoch resets patience only if val loss drops by more than this relative
    # fraction of the running best (min_delta = 1e-4 relative = 0.01%).
    es_rel_min_delta = 1e-4

    model_type = config.get("model", "unknown")
    is_at = model_type == "anomaly-transformer"
    is_lstm = model_type == "lstm-ae"

    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, f"{run_id}-best.pt")

    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    stop_reason = "max_epochs"

    for epoch in range(1, epochs + 1):
        wrapper.train()
        train_loss_sum = 0.0
        n_batches = 0

        for segs, masks, _ in train_loader:
            segs = segs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if is_at:
                if epoch <= warmup_epochs:
                    # warmup: reconstruction only, no minimax
                    loss = wrapper.compute_warmup_loss(segs, masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss_sum += loss.item()
                    del loss
                else:
                    loss1, loss2 = wrapper.compute_train_loss(segs, masks)
                    combined_loss = loss1 + loss2
                    combined_loss.backward()
                    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss_sum += combined_loss.item() / 2
                    del loss1, loss2, combined_loss
            else:
                loss = wrapper.compute_train_loss(segs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item()
                del loss

            del segs, masks
            if n_batches % 10 == 0:
                gc.collect()
            n_batches += 1

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # Validation — MSE only for all models
        wrapper.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for segs, masks, _ in val_loader:
                segs = segs.to(device)
                masks = masks.to(device)
                val_loss = wrapper.compute_val_loss(segs, masks)
                val_loss_sum += val_loss.item()
                n_val += 1

        avg_val_loss = val_loss_sum / max(n_val, 1)

        scheduler.step(avg_val_loss)

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:03d}/{epochs} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | lr={cur_lr:.2e}")
        for handler in logger.handlers:
            handler.flush()

        # Significant-improvement test against the OLD best (before any update below).
        # Patience resets only on a relative drop > min_delta (0.01% of best).
        significant = avg_val_loss < best_val_loss * (1.0 - es_rel_min_delta)

        # Always keep the absolute lowest-val-loss checkpoint, even if the drop
        # is too small to reset patience.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": wrapper.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, best_ckpt_path)
            logger.info(f"  Checkpoint saved (val_loss={best_val_loss:.6f})")

        if significant:
            patience_count = 0
        else:
            patience_count += 1
            logger.info(f"  No significant improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                stop_reason = "early_stop"
                break

    logger.info(f"Training complete. Best val_loss={best_val_loss:.6f} at epoch {best_epoch} ({stop_reason}). Checkpoint: {best_ckpt_path}")
    return best_ckpt_path, best_epoch, stop_reason
