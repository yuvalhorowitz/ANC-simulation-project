# phase_2_new/models/tcn_deeper.py
"""
Deeper TCN model - Experiment 1A

Hypothesis: Need longer receptive field for low-frequency engine harmonics.
Extend dilations to capture dependencies across longer time spans.

Changes from baseline:
- Layers: 3 → 6
- Dilations: [1,2,4] → [1,2,4,8,16,32]
- Filters: [32,32,64] → [32,32,64,64,128,128]
- Receptive field: ~25 samples → ~200 samples
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from phase_2_new.training.config import ModelConfig


def build_tcn_deeper(cfg: ModelConfig) -> tf.keras.Model:
    """
    Build deeper TCN with extended receptive field.

    Receptive field calculation:
    RF = 1 + sum((kernel_size - 1) * dilation for each layer)
    For k=3, dilations=[1,2,4,8,16,32]:
    RF = 1 + (2*1 + 2*2 + 2*4 + 2*8 + 2*16 + 2*32) = 1 + 126 = 127 samples
    At 16kHz, 127 samples ≈ 8ms

    Args:
        cfg: Model configuration (should use get_deeper_config())

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

    # Deep TCN blocks
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

    # Output
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
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
    from phase_2_new.training.config import get_deeper_config

    cfg = get_deeper_config()
    model = build_tcn_deeper(cfg.model)
    model.summary()

    print(f"\nModel: {cfg.model.model_name}")
    print(f"Filters: {cfg.model.filters}")
    print(f"Dilations: {cfg.model.dilations}")
    print(f"Receptive field: ~{1 + sum((3-1) * d for d in cfg.model.dilations)} samples")
    print(f"Total params: {model.count_params():,}")
