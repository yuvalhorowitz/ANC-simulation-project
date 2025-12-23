# phase_2_new/models/tcn_constrained_wider.py
"""
Wider Constrained TCN model - Enhanced capacity with bounded output.

**Key changes from tcn_constrained.py:**
- More filters: [64, 64, 128, 128] instead of [32, 32, 64]
- More layers: 4 TCN blocks instead of 3
- Extended dilations: [1, 2, 4, 8] instead of [1, 2, 4]
- Larger dense: 64 units instead of 32
- Same tanh output constraint: u ∈ [-1, +1]

**Use case:**
For complex acoustic environments where more model capacity helps.
Default config still uses constrained (not wider) to stay lean.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from phase_2_new.training.config import ModelConfig


def build_tcn_constrained_wider(cfg: ModelConfig) -> tf.keras.Model:
    """
    Build wider constrained TCN with bounded output activation.

    Architecture:
    - Input: (sequence_length, 1) reference mic window
    - 4 Causal Conv1D blocks with dilations [1, 2, 4, 8]
    - Filters: [64, 64, 128, 128] (progressive widening)
    - BatchNorm after each conv
    - Flatten + Dense(64) + Dense(1)
    - Output: (1,) control signal u[t] with TANH ACTIVATION

    Args:
        cfg: Model configuration

    Returns:
        Compiled Keras model with constrained output
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

    # *** KEY CHANGE: Add tanh activation to constrain output ***
    u = layers.Dense(1, activation="tanh", name="u")(x)  # Constrained: u ∈ [-1, +1]

    # Optional output scaling (for fine-tuning if needed)
    if hasattr(cfg, 'output_scale') and cfg.output_scale != 1.0:
        u = layers.Lambda(lambda x: x * cfg.output_scale, name="scale_output")(u)

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
    # Test model creation with wider config
    from phase_2_new.training.config import get_baseline_config

    cfg = get_baseline_config()
    cfg.model.model_name = "tcn_constrained_wider"
    cfg.model.filters = (64, 64, 128, 128)  # Wider
    cfg.model.dilations = (1, 2, 4, 8)       # 4 layers
    cfg.model.dense_units = 64               # Larger dense

    model = build_tcn_constrained_wider(cfg.model)
    model.summary()

    print(f"\nModel: {cfg.model.model_name}")
    print(f"Filters: {cfg.model.filters}")
    print(f"Dilations: {cfg.model.dilations}")
    print(f"Dense units: {cfg.model.dense_units}")
    print(f"Window length: {cfg.model.sequence_length}")
    print(f"Output activation: tanh (constrained to [-1, +1])")
    print(f"Total params: {model.count_params():,}")

    # Test output range
    import numpy as np
    test_input = np.random.randn(1, cfg.model.sequence_length, 1).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    print(f"\nTest output range: [{test_output.min():.6f}, {test_output.max():.6f}]")
    print(f"Expected: output in [-1, +1] range ✓")
