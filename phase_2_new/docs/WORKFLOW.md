# Phase 2 New - Workflow and Improvement Strategy

**Date Started**: 2024-12-19
**Goal**: Achieve ≥15 dB engine band reduction (current: ~3.6 dB, gap: ~11.4 dB)

## Current State Analysis

### Baseline Performance (phase_2)
From `phase_2/results/phase2_metrics.txt`:
- Full-band: 2.76 dB ± 0.88 dB
- Engine (50-400Hz): **3.62 dB ± 1.24 dB** (TARGET: ≥15 dB)
- Road (20-2000Hz): 3.01 dB ± 0.90 dB (TARGET: ≥10 dB)
- Cabin (0-1000Hz): 2.81 dB ± 0.99 dB (TARGET: ≥10 dB)

### Baseline Architecture (phase_2)
- **Model**: TCN with causal Conv1D
- **Filters**: [32, 32, 64] across 3 layers
- **Dilations**: [1, 2, 4]
- **Window length**: 200 samples (12.5ms @ 16kHz)
- **Dense layer**: 32 units → 1 output
- **Training**: Block-based with overlap, MSE loss on residual

## Systematic Improvement Plan

### Track 1: Architecture Improvements
**Hypothesis**: Current model lacks capacity to learn complex secondary path interactions

#### Experiment 1A: Deeper TCN
- Add more layers (5-6 layers instead of 3)
- Increase dilations: [1, 2, 4, 8, 16, 32]
- Keep filter sizes same initially
- **Expected impact**: Better long-range dependencies for low frequencies

#### Experiment 1B: Wider TCN
- Increase filters: [64, 64, 128, 128] or [128, 128, 256]
- Keep original dilation pattern
- **Expected impact**: More representational capacity

#### Experiment 1C: Hybrid Architecture
- TCN encoder + LSTM refinement
- Or: Multi-branch TCN (separate paths for different frequency bands)
- **Expected impact**: Capture both local and global patterns

### Track 2: Input/Output Improvements
**Hypothesis**: 200-sample window (12.5ms) is too short for low-frequency engine harmonics

#### Experiment 2A: Longer Context Window
- Increase window_length: 400, 800, 1600 samples
- Test: 25ms, 50ms, 100ms context
- **Trade-off**: Memory vs. low-frequency representation

#### Experiment 2B: Multi-Resolution Input
- Provide both raw signal and downsampled versions
- Parallel processing of different timescales
- **Expected impact**: Better frequency coverage

### Track 3: Training Strategy Improvements
**Hypothesis**: Training data or loss function is insufficient

#### Experiment 3A: More Training Data
- Generate 50-100 train scenarios (current: 10)
- More diverse noise types and room configurations
- **Expected impact**: Better generalization

#### Experiment 3B: Enhanced Frequency-Weighted Loss
- Build on `train_ref2u_with_LF_attempt.py`
- Try different α values: 0.9, 0.95 (more engine emphasis)
- Add per-band weighting for engine harmonics (60Hz, 120Hz, 180Hz, etc.)
- **Expected impact**: Force model to prioritize engine band

#### Experiment 3C: Adversarial or Perceptual Loss
- Add discriminator for residual signal quality
- Or: Perceptual loss in frequency domain
- **Expected impact**: Better perceptual noise cancellation

### Track 4: Control Signal Constraints
**Hypothesis**: Unconstrained control signal may be suboptimal

#### Experiment 4A: Control Signal Regularization
- Add L2 penalty on u(t) magnitude
- Limit dynamic range of control output
- **Expected impact**: Prevent over-driving, improve stability

#### Experiment 4B: Causality and Delay Compensation
- Explicit delay modeling in secondary path
- Phase-aware loss terms
- **Expected impact**: Better phase alignment at error mic

### Track 5: Data and Simulation Improvements
**Hypothesis**: Training/validation mismatch or simulation limitations

#### Experiment 5A: Validation Set Analysis
- Analyze which scenarios fail most
- Check if validation is representative
- **Expected impact**: Better understanding of failure modes

#### Experiment 5B: Improved Secondary Path Modeling
- More realistic h_s (longer RIRs, more reflections)
- Test with max_order = 5, 10 instead of 0
- **Expected impact**: Better real-world applicability

## Experiment Tracking Template

For each experiment, document:
```
### Experiment [ID]: [Name]
**Date**: YYYY-MM-DD
**Hypothesis**: [What we're testing]
**Changes**: [Specific code/config changes]
**Results**:
  - Engine band: X.XX dB (Δ from baseline: +/- Y.YY dB)
  - Road band: X.XX dB
  - Cabin band: X.XX dB
  - Full band: X.XX dB
**Observations**: [Qualitative findings]
**Next Steps**: [What to try next]
```

## Execution Priority

### Phase 1: Quick Wins (Week 1)
1. **Experiment 3A**: Generate more training data (50 scenarios)
2. **Experiment 1B**: Try wider TCN [64, 64, 128, 128]
3. **Experiment 3B**: Test frequency-weighted loss with α=0.95
4. **Target**: Get to 6-8 dB engine band reduction

### Phase 2: Architecture Search (Week 2)
1. **Experiment 1A**: Deeper TCN with extended dilations
2. **Experiment 2A**: Longer context window (400-800 samples)
3. **Experiment 4A**: Control signal regularization
4. **Target**: Get to 10-12 dB engine band reduction

### Phase 3: Advanced Techniques (Week 3)
1. **Experiment 1C** or **2B**: Hybrid/multi-resolution architectures
2. **Experiment 3C**: Perceptual or adversarial loss
3. **Target**: Reach or exceed 15 dB engine band reduction

## Code Organization

```
phase_2_new/
├── models/
│   ├── tcn_baseline.py          # Copy of phase_2 baseline
│   ├── tcn_deeper.py            # Experiment 1A
│   ├── tcn_wider.py             # Experiment 1B
│   ├── tcn_hybrid.py            # Experiment 1C
│   └── model_factory.py         # Factory for easy switching
├── training/
│   ├── train_baseline.py        # Standard training
│   ├── train_weighted_loss.py   # Experiment 3B
│   └── train_config.py          # Centralized config
├── testing/
│   ├── evaluate.py              # Standard evaluation
│   └── compare_models.py        # Side-by-side comparison
├── utils/
│   ├── dataset_builder.py       # Enhanced dataset creation
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting utilities
├── docs/
│   ├── WORKFLOW.md              # This file
│   └── experiment_log.md        # Detailed experiment results
└── results/
    └── [experiment_id]/         # Per-experiment results
```

## Success Criteria

### Minimum Viable Performance
- Engine band: ≥10 dB (2/3 of goal)
- Road band: ≥7 dB
- Cabin band: ≥7 dB
- Memory: ≤150 MB

### Target Performance
- Engine band: ≥15 dB ✓
- Road band: ≥10 dB ✓
- Cabin band: ≥10 dB ✓
- Memory: ≤150 MB ✓

### Stretch Goals
- Engine band: ≥20 dB
- Low latency: <10ms inference time
- Robust to position changes

## Notes

- Always save models with experiment ID
- Always run evaluation on same validation set for fair comparison
- Document all hyperparameters in experiment log
- Keep audio outputs for subjective quality assessment
