# phase_2_new/models/tcn_wider.py
"""
Wider TCN model - Experiment 1B

Hypothesis: Current model lacks capacity. Increase filters to capture
more complex secondary path interactions.

Changes from baseline:
- Filters: [32,32,64] → [64,64,128,128]
- 4 layers instead of 3
- Dilations: [1,2,4,8] to match layer count
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from phase_2_new.training.config import ModelConfig


def build_tcn_wider(cfg: ModelConfig) -> tf.keras.Model:
    """
    Build wider TCN with increased capacity.

    Expected parameters (4-8x more than baseline):
    - Baseline: ~50k params
    - Wider: ~200-400k params

    Args:
        cfg: Model configuration (should use get_wider_config())

    Returns:
        Compiled Keras model
    """

    if len(cfg.filters) != len(cfg.dilations):
        raise ValueError(
            f"filters and dilations must have same length. "
            f"Got filters={len(cfg.filters)}, dilations={len(cfg.dilations)}"
        )

    # Input
    inputs = layers.Input(
        shape=(cfg.sequence_length, cfg.channels),
        name="ref_window",
    )

    # TCN blocks (wider)
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

    # Output (also wider dense layer)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units * 2, activation="relu", name="dense")(x)  # 2x wider
    u = layers.Dense(1, name="u")(x)

    # Build model
    model = models.Model(inputs=inputs, outputs=u, name=cfg.model_name)

    # Compile
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
    from phase_2_new.training.config import get_wider_config

    cfg = get_wider_config()
    model = build_tcn_wider(cfg.model)
    model.summary()

    print(f"\nModel: {cfg.model.model_name}")
    print(f"Filters: {cfg.model.filters}")
    print(f"Dilations: {cfg.model.dilations}")
    print(f"Total params: {model.count_params():,}")
