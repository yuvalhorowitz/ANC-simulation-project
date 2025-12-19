# phase_2/training/train_ref2u.py
"""
Phase 2 Training (Ref -> TCN -> u) with engine-band weighted loss.

Enhancement over baseline Phase 2:
- Loss prioritizes engine band (50–400 Hz)
- Keeps broadband term for stability
- Fully causal, FxLMS-equivalent
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from scipy.signal import firwin

from phase_2.models.tcn_ref2u import build_tcn_ref2u, TCNConfig
from phase_2.utils.dataset_builder import build_training_example, list_scenarios


# =========================
# Configuration
# =========================
@dataclass
class TrainConfig:
    fs: int = 16000
    window_length: int = 200

    train_dir: str = "data/train"
    val_dir: str = "data/val"

    # Block training
    block_len: int = 4096

    # Training params
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3

    # Engine-band emphasis
    engine_cutoff_hz: int = 400
    engine_fir_taps: int = 129
    alpha_engine: float = 0.8   # weight on engine-band loss

    # Saving
    save_path: str = "phase_2/models/tcn_ref2u.keras"


CFG = TrainConfig()


# =========================
# Engine-band FIR filter
# =========================
def build_engine_lpf(fs: int, cutoff: int, taps: int) -> tf.Tensor:
    """
    Fixed low-pass FIR for engine band (<= cutoff).
    Returned shape: (taps, 1, 1) for tf.nn.conv1d
    """
    h = firwin(taps, cutoff, fs=fs)
    h = h / np.sum(h)
    h = tf.constant(h[::-1], dtype=tf.float32)  # reverse for conv
    return tf.reshape(h, (-1, 1, 1))


ENGINE_LPF = build_engine_lpf(
    fs=CFG.fs,
    cutoff=CFG.engine_cutoff_hz,
    taps=CFG.engine_fir_taps,
)


# =========================
# Secondary-path convolution
# =========================
def make_secondary_kernel(h_s: np.ndarray) -> tf.Tensor:
    """
    Kernel for tf.nn.conv1d (reversed for convolution).
    """
    h = tf.constant(h_s[::-1], dtype=tf.float32)
    return tf.reshape(h, (-1, 1, 1))


# =========================
# One training step (block)
# =========================
@tf.function
def train_block_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X_ext: tf.Tensor,
    err_ext: tf.Tensor,
    sec_kernel: tf.Tensor,
    center_start: tf.Tensor,
    center_len: tf.Tensor,
) -> tf.Tensor:

    with tf.GradientTape() as tape:
        # Predict control
        u_ext = model(X_ext, training=True)               # (T,1)
        u_ext_3d = tf.reshape(u_ext, (1, -1, 1))

        # Apply secondary path
        y_ext = tf.nn.conv1d(u_ext_3d, sec_kernel, 1, "SAME")
        y_ext = tf.reshape(y_ext, (-1, 1))

        # Residual at error mic
        r_ext = err_ext + y_ext

        # Center region (valid samples)
        cs = center_start
        ce = center_start + center_len
        r = r_ext[cs:ce]

        # ---- Loss terms ----
        # Full-band
        loss_full = tf.reduce_mean(tf.square(r))

        # Engine-band
        r_3d = tf.reshape(r, (1, -1, 1))
        r_engine = tf.nn.conv1d(r_3d, ENGINE_LPF, 1, "SAME")
        r_engine = tf.reshape(r_engine, (-1, 1))
        loss_engine = tf.reduce_mean(tf.square(r_engine))

        # Combined weighted loss
        loss = (
            CFG.alpha_engine * loss_engine
            + (1.0 - CFG.alpha_engine) * loss_full
        )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# =========================
# Validation (no gradients)
# =========================
def evaluate_scenario(
    model: tf.keras.Model,
    X: np.ndarray,
    err_t: np.ndarray,
    h_s: np.ndarray,
) -> float:

    sec_kernel = make_secondary_kernel(h_s)

    u = model.predict(X, batch_size=512, verbose=0).flatten()
    y = tf.nn.conv1d(
        tf.reshape(u, (1, -1, 1)),
        sec_kernel,
        1,
        "SAME",
    )
    y = tf.reshape(y, (-1,))

    r = err_t.flatten()[: len(y)] + y.numpy()
    return float(np.mean(r ** 2))


# =========================
# Main training loop
# =========================
def main():
    # Build model
    model_cfg = TCNConfig(
        sequence_length=CFG.window_length,
        initial_lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    model = build_tcn_ref2u(model_cfg)
    optimizer = model.optimizer

    # Load data
    train_items = [
        (os.path.basename(p), *build_training_example(p, CFG.fs, CFG.window_length))
        for p in list_scenarios(CFG.train_dir)
    ]
    val_items = [
        (os.path.basename(p), *build_training_example(p, CFG.fs, CFG.window_length))
        for p in list_scenarios(CFG.val_dir)
    ]

    print(f"Loaded {len(train_items)} train scenarios")
    print(f"Loaded {len(val_items)} val scenarios\n")

    best_val = np.inf
    bad_epochs = 0
    os.makedirs(os.path.dirname(CFG.save_path), exist_ok=True)

    for epoch in range(1, CFG.epochs + 1):
        losses = []

        for _, X, err_t, h_s in train_items:
            sec_kernel = make_secondary_kernel(h_s)
            T = X.shape[0]
            overlap = len(h_s) - 1

            for start in range(0, T, CFG.block_len):
                end = min(start + CFG.block_len, T)
                ext_start = max(0, start - overlap)
                ext_end = min(T, end + overlap)

                X_ext = tf.constant(X[ext_start:ext_end])
                err_ext = tf.constant(err_t[ext_start:ext_end])

                loss = train_block_step(
                    model,
                    optimizer,
                    X_ext,
                    err_ext,
                    sec_kernel,
                    tf.constant(start - ext_start),
                    tf.constant(end - start),
                )
                losses.append(loss.numpy())

        train_loss = float(np.mean(losses))

        # Validation
        val_losses = [
            evaluate_scenario(model, Xv, errv, hsv)
            for _, Xv, errv, hsv in val_items
        ]
        val_loss = float(np.mean(val_losses))

        print(
            f"[Epoch {epoch}/{CFG.epochs}] "
            f"train_loss={train_loss:.6e}  val_loss={val_loss:.6e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            model.save(CFG.save_path)
            print(f"  ✓ Saved best model → {CFG.save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= CFG.patience:
                print("Early stopping triggered.")
                break

    print("\nTraining done.")
    print(f"Best val loss: {best_val:.6e}")


if __name__ == "__main__":
    # Run from project root:
    # python -m phase_2.training.train_ref2u
    main()
