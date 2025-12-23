# Phase 2 New: Model Failure and Fix Summary

**Date:** December 21, 2025
**Project:** Active Noise Cancellation System for Vehicle Cabin
**Team:** Ariel Turnovsky, Yuval Horowitz

---

## Executive Summary

**Problem:** The Phase 2 baseline model amplified noise by -3.74 dB instead of reducing it.

**Root Cause:** Unbounded output layer (no activation) + inadequate loss function.

**Solution:** Added tanh activation constraint + amplitude penalty + frequency-weighted loss.

**Expected Improvement:** 6-8 dB engine band reduction (vs 3.26 dB baseline), no amplification.

---

## What Failed: Baseline Model (exp_00_baseline)

### Architecture Issue

```python
# PROBLEM: No activation on output layer
x = layers.Dense(32, activation="relu")(x)
u = layers.Dense(1, name="u")(x)  # ❌ Unbounded output: u ∈ (-∞, +∞)
```

**Consequence:**
- Observed output range: [-1.00, +1.26] (overshooting)
- RMS(u) = 0.524 (2.5x too loud for normalized audio)
- Model learned to output uncorrelated noise

### Loss Function Issue

```python
# PROBLEM: Only cares about residual, ignores control amplitude
loss = MSE(residual)
```

**Consequences:**
- No penalty for large control signals
- All frequency bands treated equally
- Model can "cheat" by using excessive amplitude

### Performance Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Full-band | **-3.74 dB** | >0 dB | ❌ **AMPLIFYING** |
| Engine (50-400Hz) | 3.26 dB | ≥15 dB | ❌ 11.74 dB shortfall |
| High-freq (2-8kHz) | **-28.30 dB** | ~0 dB | ❌ **676x amplification!** |
| Negative scenarios | 3/10 | 0/10 | ❌ 30% failure rate |

**User Feedback:** "Residual sounds LOUDER than the original error signal"

### Why It Happened

**Model learned wrong solution:**
1. **Correct frequency content**: corr(u, ref) = 0.52 ✓ (learned from reference)
2. **Wrong amplitude**: RMS 2.5x too high ❌
3. **Wrong phase**: corr(err, y_ctrl) = 0.02 ❌ (essentially uncorrelated)

**Result:** Added random noise instead of destructive interference.

---

## Root Cause Analysis

### Issue 1: Unbounded Output
- **Problem:** `Dense(1)` with no activation → u ∈ (-∞, +∞)
- **Effect:** Model outputs arbitrary amplitudes
- **Evidence:** RMS = 0.524 (should be ~0.4), range exceeds [-1, +1]

### Issue 2: No Amplitude Constraint
- **Problem:** Loss = MSE(residual) doesn't penalize large |u|
- **Effect:** Model uses excessive control energy
- **Evidence:** Post-processing tests showed optimal scaling = 0.10x (90% reduction needed)

### Issue 3: No Frequency Prioritization
- **Problem:** All frequencies weighted equally in loss
- **Effect:** Model reduces some bands while destroying others
- **Evidence:** High-freq amplified 676x while engine band only reduced 3.26 dB

### Issue 4: Wrong Phase
- **Problem:** No constraint ensuring anti-phase relationship
- **Effect:** Control signal uncorrelated with error
- **Evidence:** corr(err, y_ctrl) = 0.02 (should be negative for cancellation)

---

## Current Approach: Three-Part Fix

### Fix 1: Output Constraint (Hardware)

```python
# NEW: Tanh activation bounds output
u = layers.Dense(1, activation="tanh", name="u")(x)  # ✓ u ∈ [-1, +1]
```

**Effect:**
- Hard limit at ±1
- Expected RMS: 0.3-0.4 (healthy range)
- Prevents overshooting

### Fix 2: Amplitude Penalty (Software)

```python
# NEW: Penalize large control signals
loss = MSE(residual) + 0.01 * MSE(u)
```

**Effect:**
- Soft constraint encouraging efficiency
- lambda=0.01 means 1% penalty
- Model learns to use minimal control energy

### Fix 3: Frequency Weighting (Task Priority)

```python
# NEW: Emphasize engine band (project goal)
residual_engine = LowPassFilter_400Hz(residual)
loss = 0.85 * MSE(residual_engine) + 0.15 * MSE(residual_full)
```

**Effect:**
- 85% training effort on engine band (50-400 Hz)
- 15% training effort on full-band
- Directly optimizes for project goal (≥15 dB engine reduction)

### Combined Loss (exp_4d - RECOMMENDED)

```python
# Combines all three fixes
loss = [0.85 * MSE(residual_engine) + 0.15 * MSE(residual_full)] + 0.01 * MSE(u)
# Subject to: u ∈ [-1, +1] (enforced by tanh)
```

---

## Implementation: 5 Experiments

| Config | Model | Loss | What It Tests |
|--------|-------|------|---------------|
| **exp_4a** | constrained | MSE | Tanh alone |
| **exp_4b** | constrained | MSE + amplitude | Tanh + penalty |
| **exp_4c** | constrained | Freq-weighted | Tanh + frequency |
| **exp_4d** ⭐ | constrained | **Combined** | **All fixes** |
| **exp_4e** | wider | Combined | Combined + more capacity |

**Current Status:** Training exp_4d_combined (all fixes together)

---

## Expected Improvements

### Quantitative Predictions

| Metric | Baseline | exp_4d (Expected) | Improvement |
|--------|----------|-------------------|-------------|
| **Full-band** | -3.74 dB ❌ | +4-6 dB | **+7.7 to +9.7 dB** |
| **Engine band** | 3.26 dB | **6-8 dB** | **+2.7 to +4.7 dB** |
| **Road band** | 2.89 dB | 5-7 dB | +2.1 to +4.1 dB |
| **Negative scenarios** | 3/10 | 0/10 | **All positive** |
| **Control RMS** | 0.524 | ~0.35 | **33% reduction** |

### Qualitative Improvements

**Control Signal:**
- Baseline: Overshooting, uncorrelated, adds noise
- Expected: Properly bounded, anti-phase, cancels noise

**Frequency Response:**
- Baseline: High-freq amplified 676x
- Expected: High-freq impact <2x, engine band focused

**User Experience:**
- Baseline: Residual sounds LOUDER
- Expected: Residual sounds quieter, especially engine drone

---

## Key Changes Summary

### Architecture
- **File:** `phase_2_new/models/tcn_constrained.py` (new)
- **Change:** Added `activation="tanh"` to output layer
- **Effect:** u ∈ [-1, +1] instead of (-∞, +∞)

### Loss Function
- **File:** `phase_2_new/training/loss_functions.py` (new, 280 lines)
- **Changes:**
  - Amplitude penalty: `+ 0.01 * MSE(u)`
  - Frequency weighting: `0.85 * MSE(engine) + 0.15 * MSE(full)`
  - Combined: Both penalties together
- **Effect:** Multi-objective optimization with constraints

### Training
- **File:** `phase_2_new/training/train.py` (modified)
- **Changes:**
  - Configurable loss function support
  - Loss component logging (4 metrics)
  - Removed `@tf.function` (compatibility fix)

### Configuration
- **File:** `phase_2_new/training/config.py` (modified)
- **Added:**
  - loss_type: "mse", "amplitude_penalty", "frequency_weighted", "combined"
  - lambda_amp: 0.01 (amplitude penalty weight)
  - alpha_engine: 0.85 (engine band weight)
  - 5 new experiment configs (exp_4a through exp_4e)

### Data
- **Expanded:** 50 → 100 training scenarios
- **Curriculum:** 70% anechoic (easy), 30% first-order reflections (hard)
- **File:** `simulation_setup_phase_2_curriculum.py`

---

## Diagnostic Process (How We Found the Problem)

**Phase 1:** User heard amplification despite positive metrics
**Phase 2:** Created METRIC_VERIFICATION.py → Found -3.74 dB true result
**Phase 3:** Created correlation analysis → Found phase mismatch
**Phase 4:** Created amplitude analysis → Found 2.5x overshooting
**Phase 5:** Created frequency analysis → Found 676x high-freq amplification
**Phase 6-8:** Tested polarity, normalization, post-processing → All failed

**Total Diagnostic Work:**
- 9 analysis scripts
- 24 audio files (user listened to confirm)
- 3 comprehensive reports (80+ pages total)
- Archived in: `diagnostics_archive/dec21_debugging/`

**Key Insight:** Systematic debugging revealed multi-faceted problem requiring holistic solution.

---

## Success Criteria

### Minimum Viable (Must Achieve)
- ✓ Full-band: ≥0 dB (no amplification)
- ✓ Engine band: ≥6 dB (2x baseline)
- ✓ No scenarios with <-1 dB
- ✓ Control RMS: 0.3-0.5 range

### Target (Phase 2 Goal)
- ✓ Engine band: ≥10 dB
- ✓ Road band: ≥7 dB
- ✓ All scenarios positive

### Stretch (Project Requirements)
- Engine band: ≥15 dB
- Road band: ≥10 dB
- Cabin band: ≥10 dB

---

## Evaluation Plan

**After Training Completes:**
```bash
python -m phase_2_new.testing.evaluate --experiment exp_4d_combined
```

**Compare to baseline:**
- Baseline: Engine 3.26 dB, Full -3.74 dB
- Target: Engine ≥6 dB, Full ≥4 dB

**If Successful:**
```bash
# Train wider model for ≥10 dB
python -m phase_2_new.training.train --config exp_4e_combined_wider
```

---

## Conclusion

**Problem Identified:** Unbounded output + inadequate loss → model amplified noise

**Solution Implemented:** Tanh constraint + amplitude penalty + frequency weighting

**Expected Outcome:** 6-8 dB engine band reduction (2-2.5x improvement over baseline)

**Confidence:** 90% probability of achieving ≥6 dB (minimum viable success)

**Next Steps:**
1. Complete exp_4d training (in progress)
2. Evaluate and verify improvement
3. If successful, train exp_4e (wider model) for ≥10 dB

---

**Document Version:** 1.0 (Concise)
**Full Details:** See PHASE_2_FAILURE_AND_FIX_DETAILED.md
**Status:** Training in progress
