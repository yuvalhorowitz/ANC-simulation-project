# phase2/models/tcn_ref2u.py
"""
Phase 2 ANC Controller (Option D)

Causal TCN that maps a reference microphone window x[t-L ... t-1]
to a single control sample u[t].

- Input:  (sequence_length, 1)
- Output: (1,)  -> control sample u[t]
- Causal conv1d (no future leakage)
- Dilations to expand receptive field
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay


@dataclass(frozen=True)
class TCNConfig:
    # Data / signal config
    sequence_length: int = 200  # L
    channels: int = 1

    # Network size
    kernel_size: int = 3
    filters: Sequence[int] = (32, 32, 64)         # per block
    dilations: Sequence[int] = (1, 2, 4)          # must match len(filters)

    # Regularization / stability
    use_batchnorm: bool = True
    dropout: float = 0.0                          # e.g., 0.05 if needed
    dense_units: int = 32

    # Optimizer defaults (training loop may override)
    initial_lr: float = 1e-3
    decay_steps: int = 4000
    decay_rate: float = 0.7
    weight_decay: float = 1e-5

    # Naming
    model_name: str = "tcn_ref2u"


def build_tcn_ref2u(cfg: Optional[TCNConfig] = None) -> tf.keras.Model:
    """
    Build a causal TCN controller for Phase 2.

    Returns a compiled Keras model:
      input  : (L, 1)
      output : (1,)
    Note: loss used here is a placeholder. In Phase 2 training, the real loss
    is computed at the error mic AFTER applying the secondary path.
    """
    if cfg is None:
        cfg = TCNConfig()

    if len(cfg.filters) != len(cfg.dilations):
        raise ValueError(
            f"filters and dilations must have same length. "
            f"Got filters={len(cfg.filters)} dilations={len(cfg.dilations)}"
        )

    inputs = layers.Input(
        shape=(cfg.sequence_length, cfg.channels),
        name="ref_window",
    )

    x = inputs
    for i, (filt, dil) in enumerate(zip(cfg.filters, cfg.dilations)):
        x = layers.Conv1D(
            filters=filt,
            kernel_size=cfg.kernel_size,
            padding="causal",
            dilation_rate=dil,
            activation="relu",
            name=f"tcn_conv_{i+1}_d{dil}_f{filt}",
        )(x)

        if cfg.use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

        if cfg.dropout and cfg.dropout > 0.0:
            x = layers.Dropout(cfg.dropout, name=f"drop_{i+1}")(x)

    # Flatten temporal features and output single control sample
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    u = layers.Dense(1, name="u")(x)  # u[t] (single sample)

    model = models.Model(inputs=inputs, outputs=u, name=cfg.model_name)

    # Compile with optimizer schedule (loss here is placeholder)
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


def print_model_summary(sequence_length: int = 200) -> None:
    """Quick helper to sanity-check the architecture."""
    cfg = TCNConfig(sequence_length=sequence_length)
    model = build_tcn_ref2u(cfg)
    model.summary()


if __name__ == "__main__":
    # Run: python -m phase2.models.tcn_ref2u
    print_model_summary(sequence_length=200)
