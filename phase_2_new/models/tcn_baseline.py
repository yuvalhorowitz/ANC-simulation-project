# phase_2_new/models/tcn_baseline.py
"""
Baseline TCN model for Phase 2 New - identical to phase_2 for comparison.

This serves as our reference point to ensure the new framework reproduces
the original ~3.6 dB engine band results.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from phase_2_new.training.config import ModelConfig


def build_tcn_baseline(cfg: ModelConfig) -> tf.keras.Model:
    """
    Build baseline TCN identical to phase_2/models/tcn_ref2u.py

    Architecture:
    - Input: (sequence_length, 1) reference mic window
    - Causal Conv1D blocks with dilations
    - BatchNorm after each conv
    - Flatten + Dense layers
    - Output: (1,) control signal u[t]

    Args:
        cfg: Model configuration

    Returns:
        Compiled Keras model
    """

    if len(cfg.filters) != len(cfg.dilations):
        raise ValueError(
            f"filters and dilations must have same length. "
            f"Got filters={len(cfg.filters)}, dilations={len(cfg.dilations)}"
        )

    # Input layer
    inputs = layers.Input(
        shape=(cfg.sequence_length, cfg.channels),
        name="ref_window",
    )

    # TCN blocks
    x = inputs
    for i, (n_filters, dilation) in enumerate(zip(cfg.filters, cfg.dilations)):
        x = layers.Conv1D(
            filters=n_filters,
            kernel_size=cfg.kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            name=f"tcn_conv_{i+1}_d{dilation}_f{n_filters}",
        )(x)

        if cfg.use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

        if cfg.dropout and cfg.dropout > 0.0:
            x = layers.Dropout(cfg.dropout, name=f"drop_{i+1}")(x)

    # Output layers
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    u = layers.Dense(1, name="u")(x)  # Control signal output

    # Build model
    model = models.Model(inputs=inputs, outputs=u, name=cfg.model_name)

    # Compile (loss is placeholder - real loss computed during training)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=cfg.initial_lr,
        decay_steps=cfg.decay_steps,
        decay_rate=cfg.decay_rate,
        staircase=True,
    )

    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay),
        loss="mse",
    )

    return model


if __name__ == "__main__":
    # Test model creation
    from phase_2_new.training.config import get_baseline_config

    cfg = get_baseline_config()
    model = build_tcn_baseline(cfg.model)
    model.summary()

    print(f"\nModel: {cfg.model.model_name}")
    print(f"Filters: {cfg.model.filters}")
    print(f"Dilations: {cfg.model.dilations}")
    print(f"Window length: {cfg.model.sequence_length}")
    print(f"Total params: {model.count_params():,}")
