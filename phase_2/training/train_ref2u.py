# phase_2/training/train_ref2u.py
"""
Phase 2 training: Ref -> TCN -> u(t) with secondary path in the loss (Option D).

For each scenario:
  - Load X (ref windows), err_t (aligned error samples), h_s (secondary path)
  - Train the TCN so that residual r(t) = err(t) + (u * h_s)(t) is minimized

Key detail:
  Convolution couples time samples. If you shuffle individual samples you break physics.
  לכן we train in contiguous blocks with overlap of (K-1) samples (K=len(h_s)).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from phase_2.models.tcn_ref2u import build_tcn_ref2u, TCNConfig
from phase_2.utils.dataset_builder import build_training_example, list_scenarios


# -----------------------
# Configuration
# -----------------------
@dataclass
class TrainConfig:
    fs: int = 16000
    window_length: int = 200

    train_dir: str = "data/train"
    val_dir: str = "data/val"

    # Block training params
    block_len: int = 4096          # number of time steps per block loss
    batch_windows: int = 512       # batch size for running model on windows inside a block

    # Training params
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Early stopping
    patience: int = 3

    # Saving
    save_path: str = "phase_2/models/tcn_ref2u.keras"


# -----------------------
# Helpers
# -----------------------
def _make_conv_kernel(h_s: np.ndarray) -> tf.Tensor:
    """
    Build TF conv1d kernel for y = conv(u, h_s) with SAME length.
    tf.nn.conv1d uses cross-correlation, so we reverse h_s to get convolution.
    Kernel shape: (K, in_ch=1, out_ch=1)
    """
    h = tf.convert_to_tensor(h_s.astype(np.float32))
    h = tf.reverse(h, axis=[0])
    k = tf.reshape(h, (-1, 1, 1))
    return k


def _predict_u_in_batches(model: tf.keras.Model, X: np.ndarray, batch_windows: int) -> tf.Tensor:
    """
    Compute u = model(X) for many windows, in mini-batches, in TF (so gradients flow).
    Returns shape (T, 1).
    """
    T = X.shape[0]
    outputs = []

    for s in range(0, T, batch_windows):
        xb = tf.convert_to_tensor(X[s:s + batch_windows], dtype=tf.float32)
        ub = model(xb, training=True)  # (B,1)
        outputs.append(ub)

    return tf.concat(outputs, axis=0)  # (T,1)


@tf.function
def _train_block_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X_ext: tf.Tensor,          # (T_ext, L, 1)
    err_ext: tf.Tensor,        # (T_ext, 1)
    kernel: tf.Tensor,         # (K,1,1)
    center_start: tf.Tensor,   # scalar int
    center_len: tf.Tensor,     # scalar int
) -> tf.Tensor:
    """
    One gradient step on a contiguous time block using overlap:
      - predict u for the extended region
      - convolve u with h_s (SAME)
      - compute residual only on the center region (valid part)
    """
    with tf.GradientTape() as tape:
        u_ext = model(X_ext, training=True)  # (T_ext,1)
        u_ext_3d = tf.reshape(u_ext, (1, -1, 1))  # (1, T_ext, 1)

        # y_ctrl_ext = conv(u_ext, h_s) with SAME length
        y_ctrl_ext = tf.nn.conv1d(u_ext_3d, kernel, stride=1, padding="SAME")
        y_ctrl_ext = tf.reshape(y_ctrl_ext, (-1, 1))  # (T_ext,1)

        # Slice the "center" region (the block we care about)
        cs = center_start
        ce = center_start + center_len

        err_c = err_ext[cs:ce]              # (block_len,1)
        y_c = y_ctrl_ext[cs:ce]             # (block_len,1)
        residual = err_c + y_c              # (block_len,1)

        loss = tf.reduce_mean(tf.square(residual))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def _eval_scenario_loss(
    model: tf.keras.Model,
    X: np.ndarray,
    err_t: np.ndarray,
    h_s: np.ndarray,
    batch_windows: int,
) -> float:
    """
    Evaluate full-sequence loss (MSE of residual) for one scenario.
    No gradients.
    """
    kernel = _make_conv_kernel(h_s)

    # predict u for full scenario in batches (no training)
    u_list = []
    T = X.shape[0]
    for s in range(0, T, batch_windows):
        xb = tf.convert_to_tensor(X[s:s + batch_windows], dtype=tf.float32)
        ub = model(xb, training=False)
        u_list.append(ub)
    u = tf.concat(u_list, axis=0)  # (T,1)

    u_3d = tf.reshape(u, (1, -1, 1))
    y = tf.nn.conv1d(u_3d, kernel, stride=1, padding="SAME")
    y = tf.reshape(y, (-1, 1))  # (T,1)

    err_tf = tf.convert_to_tensor(err_t, dtype=tf.float32)
    residual = err_tf + y

    loss = tf.reduce_mean(tf.square(residual))
    return float(loss.numpy())


def _load_all(directory: str, fs: int, window_length: int) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all scenarios from a directory.
    Returns list of (tag, X, err_t, h_s)
    """
    prefixes = list_scenarios(directory)
    items = []

    for p in prefixes:
        tag = os.path.basename(p)
        X, err_t, h_s = build_training_example(p, fs=fs, window_length=window_length)
        items.append((tag, X, err_t, h_s))

    return items


# -----------------------
# Main training
# -----------------------
def main():
    cfg = TrainConfig()

    # Build model
    model_cfg = TCNConfig(
        sequence_length=cfg.window_length,
        initial_lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    model = build_tcn_ref2u(model_cfg)

    optimizer = model.optimizer  # AdamW with schedule from model config

    # Load data
    train_items = _load_all(cfg.train_dir, cfg.fs, cfg.window_length)
    val_items = _load_all(cfg.val_dir, cfg.fs, cfg.window_length) if os.path.isdir(cfg.val_dir) else []

    print(f"Loaded {len(train_items)} train scenarios from {cfg.train_dir}")
    print(f"Loaded {len(val_items)} val scenarios from {cfg.val_dir}" if val_items else "No val scenarios found.")
    print()

    best_val = float("inf")
    bad_epochs = 0

    # Ensure save dir exists
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_losses = []

        # --- TRAIN ---
        for tag, X, err_t, h_s in train_items:
            K = len(h_s)
            overlap = K - 1
            kernel = _make_conv_kernel(h_s)

            T = X.shape[0]
            # Walk through scenario in contiguous blocks
            for start in range(0, T, cfg.block_len):
                end = min(start + cfg.block_len, T)
                center_len = end - start

                # Extended region includes past overlap so convolution is correct
                ext_start = max(0, start - overlap)
                ext_end = min(T, end + overlap)

                X_ext_np = X[ext_start:ext_end]
                err_ext_np = err_t[ext_start:ext_end]

                # Convert to TF
                X_ext = tf.convert_to_tensor(X_ext_np, dtype=tf.float32)
                err_ext = tf.convert_to_tensor(err_ext_np, dtype=tf.float32)

                # Center indices relative to ext
                center_start = tf.constant(start - ext_start, dtype=tf.int32)
                center_len_tf = tf.constant(center_len, dtype=tf.int32)

                loss = _train_block_step(
                    model=model,
                    optimizer=optimizer,
                    X_ext=X_ext,
                    err_ext=err_ext,
                    kernel=kernel,
                    center_start=center_start,
                    center_len=center_len_tf,
                )

                train_losses.append(float(loss.numpy()))

        train_loss_epoch = float(np.mean(train_losses)) if train_losses else float("nan")

        # --- VALIDATE ---
        if val_items:
            val_losses = []
            for tag, Xv, errv, hsv in val_items:
                lv = _eval_scenario_loss(model, Xv, errv, hsv, cfg.batch_windows)
                val_losses.append(lv)
            val_loss_epoch = float(np.mean(val_losses))
        else:
            val_loss_epoch = train_loss_epoch

        print(f"[Epoch {epoch}/{cfg.epochs}] train_loss={train_loss_epoch:.6e}  val_loss={val_loss_epoch:.6e}")

        # --- EARLY STOP + SAVE BEST ---
        if val_loss_epoch < best_val - 1e-8:
            best_val = val_loss_epoch
            bad_epochs = 0
            model.save(cfg.save_path)
            print(f"  ✓ Saved best model to: {cfg.save_path}")
        else:
            bad_epochs += 1
            print(f"  (no improvement) patience={bad_epochs}/{cfg.patience}")
            if bad_epochs >= cfg.patience:
                print("Early stopping triggered.")
                break

    print("\nTraining done.")
    print(f"Best val loss: {best_val:.6e}")
    print(f"Best model path: {cfg.save_path}")


if __name__ == "__main__":
    # Run from project root:
    #   python -m phase_2.training.train_ref2u
    main()
