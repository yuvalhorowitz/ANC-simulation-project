# phase_2_new/models/model_factory.py
"""
Model factory for creating different TCN variants.

This provides a unified interface to create any model architecture
based on experiment configuration.
"""

from __future__ import annotations

import tensorflow as tf

from phase_2_new.training.config import ExperimentConfig, ModelConfig
from phase_2_new.models.tcn_baseline import build_tcn_baseline
from phase_2_new.models.tcn_wider import build_tcn_wider
from phase_2_new.models.tcn_deeper import build_tcn_deeper
from phase_2_new.models.tcn_constrained import build_tcn_constrained
from phase_2_new.models.tcn_constrained_wider import build_tcn_constrained_wider


def create_model(cfg: ExperimentConfig) -> tf.keras.Model:
    """
    Create a model based on experiment configuration.

    Args:
        cfg: Complete experiment configuration

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = cfg.model.model_name

    if model_name == "tcn_baseline":
        return build_tcn_baseline(cfg.model)
    elif model_name == "tcn_wider":
        return build_tcn_wider(cfg.model)
    elif model_name == "tcn_deeper":
        return build_tcn_deeper(cfg.model)
    elif model_name == "tcn_constrained":
        return build_tcn_constrained(cfg.model)
    elif model_name == "tcn_constrained_wider":
        return build_tcn_constrained_wider(cfg.model)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: tcn_baseline, tcn_wider, tcn_deeper, tcn_constrained, tcn_constrained_wider"
        )


def print_model_info(model: tf.keras.Model, cfg: ExperimentConfig) -> None:
    """Print useful model information."""
    print(f"\n{'='*60}")
    print(f"Model: {cfg.model.model_name}")
    print(f"Experiment: {cfg.experiment_id}")
    print(f"{'='*60}")
    print(f"Architecture:")
    print(f"  Filters: {cfg.model.filters}")
    print(f"  Dilations: {cfg.model.dilations}")
    print(f"  Window length: {cfg.model.sequence_length} samples")
    print(f"  Dense units: {cfg.model.dense_units}")
    print(f"\nTraining:")
    print(f"  Initial LR: {cfg.training.learning_rate}")
    print(f"  Weight decay: {cfg.training.weight_decay}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Block length: {cfg.training.block_len}")
    print(f"\nModel summary:")
    print(f"  Total params: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Trainable params: {trainable:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test all model types
    from phase_2_new.training.config import (
        get_baseline_config,
        get_wider_config,
        get_deeper_config,
    )

    for get_cfg in [get_baseline_config, get_wider_config, get_deeper_config]:
        cfg = get_cfg()
        model = create_model(cfg)
        print_model_info(model, cfg)
