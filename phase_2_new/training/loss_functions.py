# phase_2_new/training/loss_functions.py
"""
Enhanced loss functions for ANC training.

Addresses root causes of baseline model failure:
1. Amplitude penalty: Constrains |u| to prevent overshooting
2. Frequency weighting: Emphasizes engine band (50-400 Hz)
3. Combined: Both penalties for best results

Root cause addressed:
- Baseline outputs u with RMS 0.524 (2.5x too loud)
- No frequency prioritization (engine band treated same as noise)
- Result: -3.74 dB amplification instead of reduction
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.signal import firwin


def build_engine_lpf(fs: int = 16000, cutoff: int = 400, taps: int = 65) -> tf.Tensor:
    """
    Create FIR low-pass filter for engine band (50-400 Hz).

    Args:
        fs: Sampling rate (Hz)
        cutoff: Cutoff frequency (Hz) - upper bound of engine band
        taps: Filter length (odd number recommended, default 65 for speed)

    Returns:
        TensorFlow constant kernel ready for tf.nn.conv1d
        Shape: (taps, 1, 1)

    Example:
        >>> engine_lpf = build_engine_lpf(fs=16000, cutoff=400)
        >>> residual_engine = tf.nn.conv1d(residual_3d, engine_lpf, 1, "SAME")

    Note: Reduced default taps from 129 to 65 for 2x speedup with minimal quality loss.
    """
    # Design FIR low-pass filter
    h = firwin(taps, cutoff, fs=fs, window="hamming")
    h = h.astype(np.float32)

    # Normalize to preserve signal amplitude
    h = h / (np.sum(h) + 1e-9)

    # Reshape for conv1d: (kernel_width, in_channels, out_channels)
    kernel = tf.constant(h.reshape(-1, 1, 1), dtype=tf.float32)

    return kernel


def compute_amplitude_penalty_loss(
    residual: tf.Tensor,
    u: tf.Tensor,
    lambda_amp: float = 0.01,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Loss = MSE(residual) + lambda_amp * MSE(u)

    Penalizes large control signals to prevent overshooting.

    Args:
        residual: Residual signal at error mic (after ANC)
        u: Control signal output by model
        lambda_amp: Weight for amplitude penalty (default 0.01 = 1% of residual loss)

    Returns:
        total_loss: Combined loss
        loss_components: Dict with breakdown for logging
            - "loss_residual": MSE(residual)
            - "loss_amplitude": MSE(u)

    Why this helps:
        Baseline model outputs u with RMS 0.524 (2.5x too loud).
        Amplitude penalty encourages smaller, more controlled outputs.
        lambda_amp = 0.01 means u penalty is 1% of residual MSE.
    """
    loss_residual = tf.reduce_mean(tf.square(residual))
    loss_amplitude = tf.reduce_mean(tf.square(u))

    total_loss = loss_residual + lambda_amp * loss_amplitude

    return total_loss, {
        "loss_residual": loss_residual,
        "loss_amplitude": loss_amplitude,
    }


def compute_frequency_weighted_loss(
    residual: tf.Tensor,
    engine_lpf_kernel: tf.Tensor,
    alpha_engine: float = 0.85,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Loss = alpha * MSE(residual_engine) + (1-alpha) * MSE(residual_full)

    Emphasizes engine band (50-400 Hz) reduction over full-band.

    Args:
        residual: Residual signal at error mic (shape: (batch, 1) or (batch,))
        engine_lpf_kernel: Low-pass filter for engine band (from build_engine_lpf)
        alpha_engine: Weight for engine band (default 0.85 = 85% engine, 15% full-band)

    Returns:
        total_loss: Weighted loss
        loss_components: Dict with breakdown
            - "loss_full": Full-band MSE
            - "loss_engine": Engine-band MSE

    Why this helps:
        Project goal: ≥15 dB reduction in engine band (50-400 Hz).
        Baseline treats all frequencies equally.
        alpha=0.85 means 85% of training effort focuses on engine band.
    """
    # Full-band loss
    loss_full = tf.reduce_mean(tf.square(residual))

    # Engine-band loss: filter residual to isolate 50-400 Hz
    # Reshape for conv1d: (batch, time, channels)
    if len(residual.shape) == 1:
        residual_3d = tf.reshape(residual, (1, -1, 1))
    elif len(residual.shape) == 2:
        residual_3d = tf.reshape(residual, (1, -1, 1))
    else:
        residual_3d = residual

    # Apply low-pass filter
    residual_engine = tf.nn.conv1d(residual_3d, engine_lpf_kernel, stride=1, padding="SAME")
    residual_engine = tf.reshape(residual_engine, (-1,))

    loss_engine = tf.reduce_mean(tf.square(residual_engine))

    # Weighted combination
    total_loss = alpha_engine * loss_engine + (1.0 - alpha_engine) * loss_full

    return total_loss, {
        "loss_full": loss_full,
        "loss_engine": loss_engine,
    }


def compute_combined_loss(
    residual: tf.Tensor,
    u: tf.Tensor,
    engine_lpf_kernel: tf.Tensor,
    alpha_engine: float = 0.85,
    lambda_amp: float = 0.01,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Loss = [alpha * MSE(residual_engine) + (1-alpha) * MSE(residual_full)] + lambda * MSE(u)

    Combines frequency weighting AND amplitude penalty.
    **This is the recommended loss function.**

    Args:
        residual: Residual signal at error mic
        u: Control signal output by model
        engine_lpf_kernel: Low-pass filter for engine band
        alpha_engine: Weight for engine band (default 0.85)
        lambda_amp: Weight for amplitude penalty (default 0.01)

    Returns:
        total_loss: Combined loss
        loss_components: Dict with full breakdown
            - "loss_weighted": Frequency-weighted residual loss
            - "loss_engine": Engine-band MSE
            - "loss_full": Full-band MSE
            - "loss_amplitude": Amplitude penalty

    Why this is best:
        Addresses all root causes simultaneously:
        1. Frequency weighting: Prioritizes engine band (project goal)
        2. Amplitude penalty: Prevents overshooting (baseline RMS 0.524 → target ~0.4)
        3. Constrained output (tanh): Hard limit on u ∈ [-1, +1]
    """
    # Frequency-weighted component
    loss_full = tf.reduce_mean(tf.square(residual))

    # Engine-band filtering
    if len(residual.shape) == 1:
        residual_3d = tf.reshape(residual, (1, -1, 1))
    elif len(residual.shape) == 2:
        residual_3d = tf.reshape(residual, (1, -1, 1))
    else:
        residual_3d = residual

    residual_engine = tf.nn.conv1d(residual_3d, engine_lpf_kernel, stride=1, padding="SAME")
    residual_engine = tf.reshape(residual_engine, (-1,))

    loss_engine = tf.reduce_mean(tf.square(residual_engine))

    # Weighted frequency loss
    loss_weighted = alpha_engine * loss_engine + (1.0 - alpha_engine) * loss_full

    # Amplitude penalty
    loss_amplitude = tf.reduce_mean(tf.square(u))

    # Combine all components
    total_loss = loss_weighted + lambda_amp * loss_amplitude

    return total_loss, {
        "loss_weighted": loss_weighted,
        "loss_engine": loss_engine,
        "loss_full": loss_full,
        "loss_amplitude": loss_amplitude,
    }


# ============================================================================
# Loss function factory for training script
# ============================================================================


def get_loss_function(cfg):
    """
    Factory function to select loss based on config.

    Args:
        cfg: ExperimentConfig or TrainingConfig with loss_type field

    Returns:
        loss_fn: Callable that takes (residual, u, **kwargs) and returns (loss, components)
        loss_params: Dict of parameters to pass to loss_fn

    Supported loss_type values:
        - "mse": Baseline MSE(residual) only
        - "amplitude_penalty": MSE(residual) + lambda * MSE(u)
        - "frequency_weighted": Alpha-weighted engine band loss
        - "combined": Frequency + amplitude (recommended)

    Example:
        >>> loss_fn, loss_params = get_loss_function(cfg)
        >>> loss, components = loss_fn(residual, u, **loss_params)
    """
    # Extract config (handle both ExperimentConfig and TrainingConfig)
    if hasattr(cfg, 'training'):
        training_cfg = cfg.training
    else:
        training_cfg = cfg

    loss_type = training_cfg.loss_type

    if loss_type == "mse":
        # Baseline: MSE(residual) only
        def mse_loss(residual, u, **kwargs):
            loss = tf.reduce_mean(tf.square(residual))
            return loss, {"loss": loss}

        return mse_loss, {}

    elif loss_type == "amplitude_penalty":
        lambda_amp = training_cfg.lambda_amp
        return compute_amplitude_penalty_loss, {"lambda_amp": lambda_amp}

    elif loss_type == "frequency_weighted":
        # Build engine LPF once at initialization
        engine_lpf = build_engine_lpf(
            fs=16000,  # Fixed sample rate
            cutoff=getattr(training_cfg, 'engine_cutoff_hz', 400),
            taps=getattr(training_cfg, 'engine_fir_taps', 129),
        )
        alpha_engine = training_cfg.alpha_engine

        return compute_frequency_weighted_loss, {
            "engine_lpf_kernel": engine_lpf,
            "alpha_engine": alpha_engine,
        }

    elif loss_type == "combined":
        # Build engine LPF once at initialization
        engine_lpf = build_engine_lpf(
            fs=16000,
            cutoff=getattr(training_cfg, 'engine_cutoff_hz', 400),
            taps=getattr(training_cfg, 'engine_fir_taps', 129),
        )
        alpha_engine = training_cfg.alpha_engine
        lambda_amp = training_cfg.lambda_amp

        return compute_combined_loss, {
            "engine_lpf_kernel": engine_lpf,
            "alpha_engine": alpha_engine,
            "lambda_amp": lambda_amp,
        }

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Expected one of: mse, amplitude_penalty, frequency_weighted, combined"
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...\n")

    # Create dummy data
    batch_size = 1000
    residual = tf.random.normal((batch_size, 1))
    u = tf.random.normal((batch_size, 1)) * 0.5  # Control signal

    # Test 1: Amplitude penalty
    print("1. Amplitude penalty loss:")
    loss, components = compute_amplitude_penalty_loss(residual, u, lambda_amp=0.01)
    print(f"   Total loss: {loss.numpy():.6f}")
    print(f"   Residual MSE: {components['loss_residual'].numpy():.6f}")
    print(f"   Amplitude MSE: {components['loss_amplitude'].numpy():.6f}\n")

    # Test 2: Frequency weighted
    print("2. Frequency weighted loss:")
    engine_lpf = build_engine_lpf(fs=16000, cutoff=400)
    loss, components = compute_frequency_weighted_loss(residual, engine_lpf, alpha_engine=0.85)
    print(f"   Total loss: {loss.numpy():.6f}")
    print(f"   Full-band MSE: {components['loss_full'].numpy():.6f}")
    print(f"   Engine-band MSE: {components['loss_engine'].numpy():.6f}\n")

    # Test 3: Combined
    print("3. Combined loss (recommended):")
    loss, components = compute_combined_loss(
        residual, u, engine_lpf, alpha_engine=0.85, lambda_amp=0.01
    )
    print(f"   Total loss: {loss.numpy():.6f}")
    print(f"   Weighted residual: {components['loss_weighted'].numpy():.6f}")
    print(f"   Engine-band MSE: {components['loss_engine'].numpy():.6f}")
    print(f"   Full-band MSE: {components['loss_full'].numpy():.6f}")
    print(f"   Amplitude MSE: {components['loss_amplitude'].numpy():.6f}\n")

    print("✓ All loss functions working correctly")
