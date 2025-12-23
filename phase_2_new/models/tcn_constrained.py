# phase_2_new/models/tcn_constrained.py
"""
Constrained TCN model with bounded output - Fix for amplitude overshooting.

**Key change from tcn_baseline.py:**
- Output layer uses tanh activation: u ∈ [-1, +1] instead of (-∞, +∞)
- Prevents runaway amplitudes that caused 2.5x overshooting (RMS 0.524 vs 0.4)

**Root cause addressed:**
Current model outputs unbounded control signals, leading to:
- Wrong amplitude: 2.5x too loud
- Wrong phase: corr(err, y_ctrl) = 0.02 (should be negative)
- Result: -3.74 dB amplification instead of reduction

**Expected improvement:**
- Control signal properly bounded to normalized range
- +1-2 dB improvement in engine band
- Fewer scenarios with negative performance
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from phase_2_new.training.config import ModelConfig


def build_tcn_constrained(cfg: ModelConfig) -> tf.keras.Model:
    """
    Build constrained TCN with bounded output activation.

    Architecture:
    - Input: (sequence_length, 1) reference mic window
    - Causal Conv1D blocks with dilations
    - BatchNorm after each conv
    - Flatten + Dense layers
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
    # Test model creation
    from phase_2_new.training.config import get_baseline_config

    cfg = get_baseline_config()
    cfg.model.model_name = "tcn_constrained"
    model = build_tcn_constrained(cfg.model)
    model.summary()

    print(f"\nModel: {cfg.model.model_name}")
    print(f"Filters: {cfg.model.filters}")
    print(f"Dilations: {cfg.model.dilations}")
    print(f"Window length: {cfg.model.sequence_length}")
    print(f"Output activation: tanh (constrained to [-1, +1])")
    print(f"Total params: {model.count_params():,}")

    # Test output range
    import numpy as np
    test_input = np.random.randn(1, cfg.model.sequence_length, 1).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    print(f"\nTest output range: [{test_output.min():.6f}, {test_output.max():.6f}]")
    print(f"Expected: output in [-1, +1] range ✓")
