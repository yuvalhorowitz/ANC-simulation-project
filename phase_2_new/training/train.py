# phase_2_new/training/train.py
"""
Main training script for Phase 2 New experiments.

Usage:
    python -m phase_2_new.training.train --config baseline
    python -m phase_2_new.training.train --config exp_1b_wider
    python -m phase_2_new.training.train --config exp_1a_deeper
"""

from __future__ import annotations

import os
import argparse
import json
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from phase_2_new.training.config import (
    ExperimentConfig,
    get_baseline_config,
    get_wider_config,
    get_deeper_config,
    get_weighted_loss_config,
    get_longer_context_config,
    get_exp_4a_constrained,
    get_exp_4b_amp_penalty,
    get_exp_4c_freq_weighted,
    get_exp_4d_combined,
    get_exp_4e_combined_wider,
)
from phase_2_new.models.model_factory import create_model, print_model_info
from phase_2_new.utils.dataset_builder import build_training_example, list_scenarios
from phase_2_new.training.loss_functions import get_loss_function


# Mapping of config names to functions
CONFIG_MAP = {
    "baseline": get_baseline_config,
    "exp_00_baseline": get_baseline_config,
    "exp_1b_wider": get_wider_config,
    "exp_1a_deeper": get_deeper_config,
    "exp_3b_weighted": get_weighted_loss_config,
    "exp_2a_long_context": get_longer_context_config,
    # Experiment 4: Model architecture fixes
    "exp_4a_constrained": get_exp_4a_constrained,
    "exp_4b_amp_penalty": get_exp_4b_amp_penalty,
    "exp_4c_freq_weighted": get_exp_4c_freq_weighted,
    "exp_4d_combined": get_exp_4d_combined,
    "exp_4e_combined_wider": get_exp_4e_combined_wider,
}


def make_secondary_kernel(h_s: np.ndarray) -> tf.Tensor:
    """Build TF conv1d kernel for secondary path (reversed for convolution)."""
    h = tf.constant(h_s[::-1], dtype=tf.float32)
    return tf.reshape(h, (-1, 1, 1))


def train_block_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X_ext: tf.Tensor,
    err_ext: tf.Tensor,
    sec_kernel: tf.Tensor,
    center_start: int,
    center_len: int,
    loss_fn,
    loss_params: dict,
) -> Tuple[tf.Tensor, dict]:
    """
    One gradient step on a contiguous time block.

    This is the core training loop that implements Phase 2's approach:
    - Predict control signal u(t)
    - Apply secondary path: y = conv(u, h_s)
    - Compute residual: r = err + y
    - Minimize loss (configurable: MSE, amplitude penalty, frequency weighted, etc.)

    Args:
        model: TCN model
        optimizer: Optimizer
        X_ext: Extended input window
        err_ext: Extended error signal
        sec_kernel: Secondary path kernel
        center_start: Start index of valid region
        center_len: Length of valid region
        loss_fn: Loss function to use
        loss_params: Parameters for loss function

    Returns:
        loss: Total loss value
        loss_components: Dict of loss component values for logging
    """
    with tf.GradientTape() as tape:
        # Predict control
        u_ext = model(X_ext, training=True)  # (T_ext, 1)
        u_ext_3d = tf.reshape(u_ext, (1, -1, 1))

        # Apply secondary path
        y_ext = tf.nn.conv1d(u_ext_3d, sec_kernel, stride=1, padding="SAME")
        y_ext = tf.reshape(y_ext, (-1, 1))

        # Residual at error mic
        residual_ext = err_ext + y_ext

        # Extract center region (valid samples)
        cs = center_start
        ce = center_start + center_len
        residual = residual_ext[cs:ce]
        u = u_ext[cs:ce]  # Extract control signal for amplitude penalty

        # Compute loss using configurable loss function
        loss, loss_components = loss_fn(residual, u, **loss_params)

    # Update weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, loss_components


def evaluate_scenario(
    model: tf.keras.Model,
    X: np.ndarray,
    err_t: np.ndarray,
    h_s: np.ndarray,
    batch_size: int = 512,
) -> float:
    """Evaluate full-sequence loss for one scenario (no gradients)."""
    sec_kernel = make_secondary_kernel(h_s)

    # Predict in batches
    u = model.predict(X, batch_size=batch_size, verbose=0).flatten()

    # Apply secondary path
    u_3d = tf.reshape(u, (1, -1, 1))
    y = tf.nn.conv1d(u_3d, sec_kernel, stride=1, padding="SAME")
    y = tf.reshape(y, (-1,)).numpy()

    # Residual
    err = err_t.flatten()[:len(y)]
    residual = err + y

    return float(np.mean(residual ** 2))


def load_dataset(
    directory: str,
    fs: int,
    window_length: int,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Load all scenarios from a directory."""
    prefixes = list_scenarios(directory)
    items = []

    for p in prefixes:
        tag = os.path.basename(p)
        X, err_t, h_s = build_training_example(p, fs=fs, window_length=window_length)
        items.append((tag, X, err_t, h_s))

    return items


def train(cfg: ExperimentConfig):
    """Main training function."""

    print(f"\n{'='*60}")
    print(f"PHASE 2 NEW - Training")
    print(f"Experiment: {cfg.experiment_id}")
    print(f"Description: {cfg.description}")
    print(f"{'='*60}\n")

    # Create model
    model = create_model(cfg)
    print_model_info(model, cfg)

    # Get loss function
    loss_fn, loss_params = get_loss_function(cfg)
    print(f"\nLoss function: {cfg.training.loss_type}")
    if cfg.training.loss_type == "amplitude_penalty":
        print(f"  lambda_amp: {cfg.training.lambda_amp}")
    elif cfg.training.loss_type == "frequency_weighted":
        print(f"  alpha_engine: {cfg.training.alpha_engine}")
        print(f"  engine_cutoff_hz: {getattr(cfg.training, 'engine_cutoff_hz', 400)}")
    elif cfg.training.loss_type == "combined":
        print(f"  alpha_engine: {cfg.training.alpha_engine}")
        print(f"  lambda_amp: {cfg.training.lambda_amp}")
        print(f"  engine_cutoff_hz: {getattr(cfg.training, 'engine_cutoff_hz', 400)}")
    print()

    # Load data
    print(f"Loading data from {cfg.data.train_dir}...")
    train_items = load_dataset(
        cfg.data.train_dir,
        cfg.data.fs,
        cfg.model.sequence_length,
    )
    print(f"✓ Loaded {len(train_items)} training scenarios")

    if os.path.isdir(cfg.data.val_dir):
        print(f"Loading data from {cfg.data.val_dir}...")
        val_items = load_dataset(
            cfg.data.val_dir,
            cfg.data.fs,
            cfg.model.sequence_length,
        )
        print(f"✓ Loaded {len(val_items)} validation scenarios")
    else:
        print(f"⚠ No validation directory found at {cfg.data.val_dir}")
        val_items = []

    print()

    # Training loop
    optimizer = model.optimizer
    best_val_loss = float("inf")
    bad_epochs = 0

    # Ensure output directories exist
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    os.makedirs(cfg.results_path, exist_ok=True)

    # Track loss components across epochs
    epoch_metrics = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_losses = []
        train_components = {}  # Accumulate loss components

        # Train on all scenarios
        for tag, X, err_t, h_s in train_items:
            sec_kernel = make_secondary_kernel(h_s)
            T = X.shape[0]
            overlap = len(h_s) - 1

            # Block-based training (preserve convolution physics)
            for start in range(0, T, cfg.training.block_len):
                end = min(start + cfg.training.block_len, T)
                center_len = end - start

                # Extended region for correct convolution
                ext_start = max(0, start - overlap)
                ext_end = min(T, end + overlap)

                X_ext = tf.constant(X[ext_start:ext_end], dtype=tf.float32)
                err_ext = tf.constant(err_t[ext_start:ext_end], dtype=tf.float32)

                # Training step with configurable loss
                loss, loss_comp = train_block_step(
                    model=model,
                    optimizer=optimizer,
                    X_ext=X_ext,
                    err_ext=err_ext,
                    sec_kernel=sec_kernel,
                    center_start=start - ext_start,
                    center_len=center_len,
                    loss_fn=loss_fn,
                    loss_params=loss_params,
                )

                train_losses.append(float(loss.numpy()))

                # Accumulate loss components
                for key, value in loss_comp.items():
                    if key not in train_components:
                        train_components[key] = []
                    train_components[key].append(float(value.numpy()))

        train_loss_epoch = float(np.mean(train_losses))

        # Average loss components
        train_comp_avg = {k: float(np.mean(v)) for k, v in train_components.items()}

        # Validate
        if val_items:
            val_losses = []
            for tag, Xv, errv, hsv in val_items:
                lv = evaluate_scenario(model, Xv, errv, hsv, cfg.training.batch_windows)
                val_losses.append(lv)
            val_loss_epoch = float(np.mean(val_losses))
        else:
            val_loss_epoch = train_loss_epoch

        # Print epoch summary with loss components
        comp_str = " | ".join([f"{k}={v:.4e}" for k, v in train_comp_avg.items()])
        print(
            f"[Epoch {epoch}/{cfg.training.epochs}] "
            f"train={train_loss_epoch:.6e} | val={val_loss_epoch:.6e} | {comp_str}"
        )

        # Save metrics for this epoch
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "components": train_comp_avg,
        })

        # Save best model
        if val_loss_epoch < best_val_loss - 1e-8:
            best_val_loss = val_loss_epoch
            bad_epochs = 0
            model.save(cfg.model_path)
            print(f"  ✓ Saved best → {cfg.model_path}")
        else:
            bad_epochs += 1
            print(f"  (no improvement) patience={bad_epochs}/{cfg.training.patience}")
            if bad_epochs >= cfg.training.patience:
                print("  ✗ Early stopping triggered")
                break

    # Save config and results
    config_path = os.path.join(cfg.results_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "experiment_id": cfg.experiment_id,
            "description": cfg.description,
            "model": {
                "name": cfg.model.model_name,
                "filters": list(cfg.model.filters),
                "dilations": list(cfg.model.dilations),
                "sequence_length": cfg.model.sequence_length,
                "dense_units": cfg.model.dense_units,
            },
            "training": {
                "epochs": cfg.training.epochs,
                "learning_rate": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
                "loss_type": cfg.training.loss_type,
                "loss_params": {
                    "lambda_amp": getattr(cfg.training, 'lambda_amp', None),
                    "alpha_engine": getattr(cfg.training, 'alpha_engine', None),
                    "engine_cutoff_hz": getattr(cfg.training, 'engine_cutoff_hz', None),
                },
            },
            "final_val_loss": best_val_loss,
        }, f, indent=2)

    # Save training metrics
    metrics_path = os.path.join(cfg.results_path, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(epoch_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Model saved: {cfg.model_path}")
    print(f"Config saved: {config_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 New models")
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        choices=list(CONFIG_MAP.keys()),
        help="Experiment configuration to use"
    )
    args = parser.parse_args()

    # Get config
    cfg = CONFIG_MAP[args.config]()

    # Train
    train(cfg)


if __name__ == "__main__":
    main()
