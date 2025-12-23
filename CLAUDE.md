# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project #3214**: Active System for Noise Reduction in a Vehicle
**Institution**: Tel Aviv University, School of Electrical Engineering
**Students**: Ariel Turnovsky, Yuval Horowitz
**Supervisor**: Dr. Lior Arbel

This is an Active Noise Cancellation (ANC) simulation project using deep learning (Temporal Convolutional Networks) to generate anti-noise signals that cancel unwanted ambient noise in a car cabin through destructive interference. The project simulates a car cabin environment using pyroomacoustics and trains TCN models to minimize noise at the driver's ear position (error microphone).

The TCN approach is chosen over classic adaptive filters (like FxLMS) for its superior ability to adapt quickly to environmental changes (e.g., driver head movement) and to model complex non-linear acoustic relationships while maintaining causality.

## Project Structure

The codebase is organized into three phases representing different ANC approaches:

### Phase 1 - Baseline (ref→err mapping)
- **Location**: `phase1/`
- **Purpose**: Baseline experiments mapping reference mic directly to error mic
- **Key finding**: Simple ref→err prediction fails in realistic environments (see phase1/docs/phase1_summary.txt)
- **Models**: `phase1/models/tcn_model_ref2err.py`
- **Training**: `phase1/training/train_tcn_ref2err.py`

### Phase 2 - Control Signal Learning (ref→u with secondary path)
- **Location**: `phase_2/`
- **Purpose**: TCN learns to output control signal u(t) that is then filtered through secondary path h_s
- **Architecture**: Reference window → TCN → control u(t) → (convolve h_s) → residual at error mic
- **Models**: `phase_2/models/tcn_ref2u.py`
- **Training**: `phase_2/training/train_ref2u.py`
- **Testing**: `phase_2/testing/test_ref2u.py`
- **Dataset utils**: `phase_2/utils/dataset_builder.py`

### Phase 3 - Interactive Simulation
- **Location**: `phase_3/`
- **Purpose**: Real-time scenario runner with configurable room geometry
- **Key module**: `phase_3/simulation/run_scenario.py`
- **Visualization**: `phase_3/visualization/plot_*.py`

### Root-level Simulation Setup
- **File**: `simulation_setup.py`
- **Purpose**: Generate training/validation datasets with pyroomacoustics
- **Outputs**: Creates `data/train/` and `data/val/` with scenario files

## Common Commands

### Generate Dataset
```bash
python simulation_setup.py
```
This creates scenarios in `data/train/` and `data/val/` with files:
- `*_ref.wav` - reference microphone signal
- `*_err.wav` - error microphone signal (at driver's ear)
- `*_rir_src_to_ref.npy` - RIR from noise source to reference mic
- `*_rir_src_to_err.npy` - RIR from noise source to error mic
- `*_rir_spk_to_err.npy` - RIR from speaker to error mic (secondary path h_s)

### Train Phase 2 Model (Current Best Approach)
```bash
python -m phase_2.training.train_ref2u
```
- Trains TCN controller with secondary path in the loss
- Saves best model to `phase_2/models/tcn_ref2u.keras`
- Uses block-based training with overlap to preserve convolution physics

### Evaluate Phase 2 Model
```bash
python -m phase_2.testing.test_ref2u
```
- Evaluates on validation set
- Computes full-band and band-limited metrics (engine 50-400Hz, road 20-2000Hz, cabin 0-1000Hz)
- Outputs before/after audio to `phase_2/testing/audio_outputs/`

### Run Phase 3 Scenario
```bash
python -m phase_3.simulation.run_scenario
```
- Interactive ANC simulation with trained model
- Configurable room geometry, materials, and noise types

### Visualize Results
```bash
python -m phase_3.visualization.plot_waveforms
python -m phase_3.visualization.plot_room
```

## Key Architecture Concepts

### Signal Flow (Phase 2/3)
1. **Noise source** → propagates to **reference mic** (x[t]) and **error mic** (d[t])
2. **Reference mic** → TCN controller → **control signal u[t]**
3. **Control signal u[t]** → speaker → **secondary path h_s** → **y_ctrl[t]** at error mic
4. **Residual** = d[t] + y_ctrl[t] (what driver hears after ANC)

### Critical Alignment (dataset_builder.py)
For each time index t:
- Input window: `ref[t-window_length : t]`
- Target: `err[t]`
- This preserves causality matching FxLMS timing

### Training Loss (Phase 2)
The loss is computed on the residual at the error microphone AFTER applying secondary path:
```
u(t) = TCN(ref[t-L:t])
y_ctrl(t) = conv(u, h_s)
residual(t) = err(t) + y_ctrl(t)
loss = MSE(residual)
```

### Block Training Strategy
Phase 2 uses contiguous time blocks with overlap = len(h_s) - 1 to preserve convolution physics. Shuffling individual samples would break the temporal coupling of the secondary path convolution.

## Model Configuration

### TCN Architecture (phase_2/models/tcn_ref2u.py)
- Input: (200, 1) - 200-sample reference window (12.5ms at 16kHz)
- Causal Conv1D blocks with dilations [1, 2, 4] and filters [32, 32, 64]
- BatchNorm after each conv
- Flatten → Dense(32) → Dense(1) output
- Output: single control sample u[t]

### Sampling Rate
- Fixed at **16 kHz** throughout all phases
- Defined as `FS = 16000` in simulation and training scripts

## Project Goals (Quantitative Requirements)

Official specifications from project work plan (Project #3214):

### 1. Noise Reduction Targets
- **Engine noise** (50-400 Hz): **≥15 dB reduction** (primary goal)
- **Road noise** (20-2000 Hz): **≥10 dB reduction**
- **Cabin/AC noise** (≤1000 Hz): **≥10 dB reduction**
- Overall goal: Average noise reduction of **at least 10 dB** at listening points (driver's seat)

### 2. Memory Footprint
- **≤150 MB RAM** total for entire system during operation
- Includes: model parameters + intermediate activation tensors during inference + audio buffers

### 3. Model Accuracy
- **MSE ≤0.001** for normalized signals when comparing TCN-generated anti-noise vs ideal anti-noise
- This measures accuracy in phase, amplitude, and timing

## Data Organization

### Scenario File Naming Convention
```
{split}_scenario_{idx:03d}_{suffix}
```
Examples:
- `train_scenario_000_ref.wav`
- `val_scenario_002_rir_spk_to_err.npy`

### Noise Types
Generated in `simulation_setup.py`:
- `engine`: Harmonic tones with AM modulation (base 50-120 Hz)
- `road`: Filtered broadband noise (rumble)
- `wind`: High-pass filtered noise with AM
- `mixed`: Combination of above

## Important Notes

### Phase 2 vs Phase 1
Phase 1 attempted to predict the error signal directly and invert it. This failed because:
1. Error signal depends on secondary path (speaker→ear)
2. No knowledge of h_s in the model
3. Results: -1.16 dB (worse than baseline) in realistic conditions

Phase 2 solves this by:
1. Outputting control signal u(t) instead of anti-noise
2. Including secondary path h_s in the training loss
3. Learning optimal control considering the full acoustic path

### Phase 2 Performance Challenges

**Current Results** (from phase_2/results/phase2_metrics.txt):
- **Full-band reduction**: 2.76 dB average (std: 0.88 dB)
- **Engine band (50-400Hz)**: 3.62 dB average (std: 1.24 dB)
- **Road band (20-2000Hz)**: 3.01 dB average (std: 0.90 dB)
- **Cabin band (0-1000Hz)**: 2.81 dB average (std: 0.99 dB)

**Gap to Project Goals**:
- Engine band target: **≥15 dB** → Achieved: **~3.6 dB** → **Gap: ~11.4 dB shortfall**
- Road band target: **≥10 dB** → Achieved: **~3.0 dB** → **Gap: ~7.0 dB shortfall**
- Cabin band target: **≥10 dB** → Achieved: **~2.8 dB** → **Gap: ~7.2 dB shortfall**

**Attempted Solutions**:
1. **Frequency-weighted loss** (train_ref2u_with_LF_attempt.py):
   - Added engine-band emphasis using low-pass FIR filter (≤400 Hz)
   - Weighted loss: `α * loss_engine + (1-α) * loss_full` where α=0.8
   - Goal: Prioritize engine band reduction over broadband
   - Status: Implementation exists but results not documented

**Potential Issues**:
1. **Model capacity**: TCN may be too small (32-32-64 filters) to learn complex secondary path interactions
2. **Training data**: Only 10 train + 3 val scenarios may be insufficient
3. **Window length**: 200 samples (12.5ms) may not capture low-frequency engine harmonics well
4. **Secondary path complexity**: Room reflections and reverberation create difficult non-minimum-phase paths
5. **Control signal amplitude**: May need constraints or regularization to prevent over-driving speaker
6. **Receptive field**: Dilation pattern [1,2,4] may not extend far enough for low frequencies

### TensorFlow Environment Variables
Phase 3 sets these for stability:
```python
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
```

### Metric Interpretation
- Positive dB values indicate noise reduction (better)
- Negative dB values indicate noise amplification (worse)
- dE = 10*log10(MSE_before / MSE_after)

## Project Stages (Work Plan)

The project is structured in four main stages:

### Stage A: Building the Room Simulation
- Define room dimensions and wall properties
- Create noise sources and microphone locations
- Simulate acoustic transfer function
- Tools: Python, NumPy, SciPy, `pyroomacoustics`
- Output: Functional acoustic simulation model with wrapper interface

### Stage B: Data Collection and Selection
- Sample/create various urban noises (engine, road, cabin/AC, wind)
- Run through simulation to generate input-output training pairs
- Output: Large processed dataset ready for training

### Stage C: Model Selection, Design, and Training
- Choose appropriate TCN architecture
- Train on collected data
- Optimize for output quality and minimal memory footprint
- Output: Trained and optimized TCN model

### Stage D: Design and Testing
- Integrate trained model into simulation
- Run system on unseen noise files (batch processing)
- Implement offline feedback loop for tuning
- Output: Residual noise audio files and quantitative verification

## Testing and Verification Methods

1. **Direct measurement** of noise reduction (dB)
2. **Generalization testing** with unseen noise sources
3. **Frequency domain analysis** of signals before/after processing
4. **Latency measurement** of model processing time

## Dependencies

From `requirements.txt`:
- `numpy` - array operations
- `scipy` - signal processing, convolution
- `pyroomacoustics` - room acoustic simulation
- `torch` - (legacy, not used in Phase 2/3)
- `matplotlib` - visualization
- Additional (implicit): `tensorflow`, `librosa`, `keras`
