# Phase 2 New - Systematic ANC Improvement

## Overview

This folder contains a systematic approach to improving the Phase 2 ANC controller performance from the current **~3.6 dB** engine band reduction to the target **≥15 dB**.

## Quick Start

### 1. Review the Workflow
```bash
cat phase_2_new/docs/WORKFLOW.md
```
This contains the complete improvement strategy organized into 5 tracks with 15+ experiments.

### 2. Check Current Status
```bash
cat phase_2_new/docs/experiment_log.md
```
This logs all completed experiments with detailed results.

### 3. Run an Experiment

Each experiment has a predefined configuration in `training/config.py`:

```python
# Example: Run baseline reproduction
python -m phase_2_new.training.train --config baseline

# Example: Run wider network experiment
python -m phase_2_new.training.train --config exp_1b_wider

# Example: Run with custom config
python -m phase_2_new.training.train --config exp_custom --filters 128,128,256
```

### 4. Evaluate Results
```python
python -m phase_2_new.testing.evaluate --experiment exp_1b_wider
```

### 5. Compare Experiments
```python
python -m phase_2_new.testing.compare --experiments baseline,exp_1b_wider,exp_1a_deeper
```

## Improvement Tracks

### Track 1: Architecture (Model Capacity)
- **1A**: Deeper TCN (6 layers, dilations up to 32)
- **1B**: Wider TCN (128-256 filters)
- **1C**: Hybrid architectures (TCN + LSTM, multi-branch)

### Track 2: Input/Output (Context Length)
- **2A**: Longer windows (400, 800, 1600 samples)
- **2B**: Multi-resolution inputs

### Track 3: Training Strategy (Loss & Data)
- **3A**: More training data (50-100 scenarios)
- **3B**: Frequency-weighted loss (α=0.9-0.95)
- **3C**: Adversarial or perceptual loss

### Track 4: Control Constraints (Stability)
- **4A**: Control signal regularization
- **4B**: Phase-aware loss

### Track 5: Simulation (Data Quality)
- **5A**: Validation set analysis
- **5B**: More realistic secondary paths

## File Organization

```
phase_2_new/
├── README.md                    # This file
├── docs/
│   ├── WORKFLOW.md              # Complete improvement strategy
│   └── experiment_log.md        # Detailed results log
├── models/
│   ├── tcn_baseline.py          # Baseline architecture
│   ├── tcn_wider.py             # Wider network (Exp 1B)
│   ├── tcn_deeper.py            # Deeper network (Exp 1A)
│   └── model_factory.py         # Factory for creating models
├── training/
│   ├── config.py                # Experiment configurations
│   ├── train.py                 # Main training script
│   └── losses.py                # Loss functions (MSE, weighted, etc.)
├── testing/
│   ├── evaluate.py              # Evaluation script
│   └── compare.py               # Multi-experiment comparison
├── utils/
│   ├── dataset_builder.py       # Dataset loading (copied from phase_2)
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting utilities
└── results/
    └── [experiment_id]/         # Results per experiment
        ├── metrics.txt
        ├── config.json
        └── audio/
```

## Current Status

**Baseline Performance** (phase_2):
- Engine band: 3.62 dB ± 1.24 dB
- Gap to goal: **11.4 dB**

**Target Performance**:
- Engine band: **≥15 dB**
- Road band: ≥10 dB
- Cabin band: ≥10 dB

**Experiments Completed**: 0 / 15+

See `docs/experiment_log.md` for detailed results.

## Development Workflow

### Adding a New Experiment

1. **Define configuration** in `training/config.py`:
```python
def get_my_experiment_config() -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_id="exp_X_name",
        description="What makes this different",
    )
    # Modify cfg.model, cfg.training as needed
    return cfg
```

2. **Create model architecture** (if needed) in `models/`:
```python
# models/tcn_my_variant.py
def build_tcn_my_variant(cfg: ModelConfig) -> tf.keras.Model:
    # Your architecture here
    pass
```

3. **Run training**:
```bash
python -m phase_2_new.training.train --config exp_X_name
```

4. **Evaluate**:
```bash
python -m phase_2_new.testing.evaluate --experiment exp_X_name
```

5. **Log results** in `docs/experiment_log.md`:
- Copy template from experiment_log.md
- Fill in all sections
- Analyze what worked/didn't work
- Propose next steps

### Best Practices

- **One variable at a time**: Change one thing per experiment for clear attribution
- **Fair comparison**: Always evaluate on the same validation set
- **Document everything**: Log hyperparameters, training time, observations
- **Save audio**: Keep before/after audio for subjective quality assessment
- **Version control**: Commit after each complete experiment

## Next Steps

### Immediate Priority (Week 1)
1. Generate more training data (50 scenarios)
2. Run Experiment 1B (wider network)
3. Run Experiment 3B (weighted loss α=0.95)

### Medium Priority (Week 2)
1. Run Experiment 1A (deeper network)
2. Run Experiment 2A (longer context)
3. Analyze which approaches are most promising

### Long-term (Week 3)
1. Combine best approaches
2. Fine-tune hyperparameters
3. Validate on test set
4. Reach ≥15 dB target

## Questions or Issues?

Refer to:
- `docs/WORKFLOW.md` for complete strategy
- `phase_2/docs/phase2_step2_freeze_metrics_summary.txt` for baseline context
- `CLAUDE.md` (project root) for overall project documentation
