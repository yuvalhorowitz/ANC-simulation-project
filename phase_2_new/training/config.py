"""
Centralized configuration for Phase 2 New experiments.

This module provides base configurations that can be inherited and modified
for different experiments.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Dataset configuration."""
    fs: int = 16000
    train_dir: str = "data/train"
    val_dir: str = "data/val"


@dataclass
class ModelConfig:
    """Base model architecture configuration."""
    sequence_length: int = 200
    channels: int = 1
    kernel_size: int = 3

    # These will be overridden in specific experiments
    filters: Tuple[int, ...] = (32, 32, 64)
    dilations: Tuple[int, ...] = (1, 2, 4)

    use_batchnorm: bool = True
    dropout: float = 0.0
    dense_units: int = 32

    # Optimizer params (used by model builders)
    initial_lr: float = 1e-3
    decay_steps: int = 4000
    decay_rate: float = 0.7
    weight_decay: float = 1e-5

    model_name: str = "tcn_baseline"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Block-based training (preserve convolution physics)
    block_len: int = 4096
    batch_windows: int = 512

    # Optimization
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Learning rate schedule
    decay_steps: int = 4000
    decay_rate: float = 0.7

    # Early stopping
    patience: int = 3

    # Loss function selection
    # Options: "mse", "amplitude_penalty", "frequency_weighted", "combined"
    loss_type: str = "mse"

    # Amplitude penalty parameters (for "amplitude_penalty" and "combined" loss)
    lambda_amp: float = 0.01  # Weight for amplitude penalty (0.01 = 1% of residual MSE)

    # Frequency weighting parameters (for "frequency_weighted" and "combined" loss)
    alpha_engine: float = 0.85  # Weight for engine band (0.85 = 85% engine, 15% full-band)
    engine_cutoff_hz: int = 400  # Low-pass cutoff for engine band (50-400 Hz)
    engine_fir_taps: int = 65  # FIR filter length (reduced from 129 for speed)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str = "baseline"
    description: str = "Baseline Phase 2 model"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Output paths
    save_dir: str = "phase_2_new/models"
    results_dir: str = "phase_2_new/results"

    @property
    def model_path(self) -> str:
        """Path to save trained model."""
        return f"{self.save_dir}/{self.experiment_id}.keras"

    @property
    def results_path(self) -> str:
        """Path to save results."""
        return f"{self.results_dir}/{self.experiment_id}"


# Predefined experiment configurations

def get_baseline_config() -> ExperimentConfig:
    """Original Phase 2 baseline for comparison."""
    return ExperimentConfig(
        experiment_id="exp_00_baseline",
        description="Baseline: TCN [32,32,64], dil [1,2,4], w=200, MSE loss",
    )


def get_wider_config() -> ExperimentConfig:
    """Experiment 1B: Wider network."""
    cfg = ExperimentConfig(
        experiment_id="exp_1b_wider",
        description="Wider TCN: [64,64,128,128], same dilations",
    )
    cfg.model.filters = (64, 64, 128, 128)
    cfg.model.dilations = (1, 2, 4, 8)
    cfg.model.model_name = "tcn_wider"
    return cfg


def get_deeper_config() -> ExperimentConfig:
    """Experiment 1A: Deeper network with extended dilations."""
    cfg = ExperimentConfig(
        experiment_id="exp_1a_deeper",
        description="Deeper TCN: 6 layers, dil [1,2,4,8,16,32]",
    )
    cfg.model.filters = (32, 32, 64, 64, 128, 128)
    cfg.model.dilations = (1, 2, 4, 8, 16, 32)
    cfg.model.model_name = "tcn_deeper"
    return cfg


def get_weighted_loss_config() -> ExperimentConfig:
    """Experiment 3B: Frequency-weighted loss with high engine emphasis."""
    cfg = ExperimentConfig(
        experiment_id="exp_3b_weighted_095",
        description="Weighted loss: alpha=0.95 engine emphasis",
    )
    cfg.training.loss_type = "weighted_freq"
    cfg.training.alpha_engine = 0.95
    return cfg


def get_longer_context_config() -> ExperimentConfig:
    """Experiment 2A: Longer context window."""
    cfg = ExperimentConfig(
        experiment_id="exp_2a_long_context",
        description="Long context: 400 samples (25ms)",
    )
    cfg.model.sequence_length = 400
    cfg.model.model_name = "tcn_long_context"
    return cfg


# ============================================================================
# Experiment 4: Model Architecture Fixes (Constrained Output + Enhanced Loss)
# ============================================================================


def get_exp_4a_constrained() -> ExperimentConfig:
    """
    Experiment 4A: Constrained output (tanh) with MSE loss.

    Tests output constraint alone to measure impact on amplitude overshooting.
    Expected: +1-2 dB improvement, fewer negative scenarios.
    """
    cfg = ExperimentConfig(
        experiment_id="exp_4a_constrained",
        description="Constrained TCN: tanh activation, MSE loss",
    )
    cfg.model.model_name = "tcn_constrained"
    cfg.training.loss_type = "mse"
    return cfg


def get_exp_4b_amp_penalty() -> ExperimentConfig:
    """
    Experiment 4B: Constrained + Amplitude penalty.

    Tests amplitude penalty loss to further reduce control signal overshooting.
    Expected: +0.5-1 dB additional improvement, lower u amplitude.
    """
    cfg = ExperimentConfig(
        experiment_id="exp_4b_amp_penalty",
        description="Constrained + Amplitude penalty: lambda=0.01",
    )
    cfg.model.model_name = "tcn_constrained"
    cfg.training.loss_type = "amplitude_penalty"
    cfg.training.lambda_amp = 0.01
    return cfg


def get_exp_4c_freq_weighted() -> ExperimentConfig:
    """
    Experiment 4C: Constrained + Frequency weighted loss.

    Tests frequency weighting to prioritize engine band (50-400 Hz).
    Expected: +2-3 dB engine band improvement.
    """
    cfg = ExperimentConfig(
        experiment_id="exp_4c_freq_weighted",
        description="Constrained + Frequency weighted: alpha=0.85",
    )
    cfg.model.model_name = "tcn_constrained"
    cfg.training.loss_type = "frequency_weighted"
    cfg.training.alpha_engine = 0.85
    return cfg


def get_exp_4d_combined() -> ExperimentConfig:
    """
    Experiment 4D: Combined (RECOMMENDED).

    Combines all fixes: tanh output + amplitude penalty + frequency weighting.
    This is the recommended approach addressing all root causes.
    Expected: +3-5 dB engine band improvement (cumulative).
    """
    cfg = ExperimentConfig(
        experiment_id="exp_4d_combined",
        description="Combined: tanh + amp penalty + freq weighted (RECOMMENDED)",
    )
    cfg.model.model_name = "tcn_constrained"
    cfg.training.loss_type = "combined"
    cfg.training.alpha_engine = 0.85
    cfg.training.lambda_amp = 0.001  # Reduced from 0.01 - was too strong
    return cfg


def get_exp_4e_combined_wider() -> ExperimentConfig:
    """
    Experiment 4E: Combined + Wider model.

    If 4D succeeds, test with enhanced model capacity.
    Filters: [64, 64, 128, 128] instead of [32, 32, 64]
    Dilations: [1, 2, 4, 8] (4 layers)
    Dense: 64 units
    Expected: +5-8 dB engine band (approaching 15 dB goal).
    """
    cfg = ExperimentConfig(
        experiment_id="exp_4e_combined_wider",
        description="Combined + Wider: [64,64,128,128], dil [1,2,4,8], d=64",
    )
    cfg.model.model_name = "tcn_constrained_wider"
    cfg.model.filters = (64, 64, 128, 128)
    cfg.model.dilations = (1, 2, 4, 8)
    cfg.model.dense_units = 64
    cfg.training.loss_type = "combined"
    cfg.training.alpha_engine = 0.85
    cfg.training.lambda_amp = 0.01
    cfg.training.epochs = 15  # More capacity needs more training
    return cfg

