# Diagnostic Audio Files Archive - December 21, 2025

## Purpose
This directory contains diagnostic audio files generated during root cause analysis of the Phase 2 baseline model failure.

## Background
The baseline model (`phase_2_new/models/exp_00_baseline.keras`) was found to amplify noise by -3.74 dB instead of reducing it. These audio files were created during systematic debugging to identify the root cause.

## Root Cause Found
Model outputs control signal u(t) with:
- Wrong amplitude: 2.5x too loud (RMS 0.52 vs should be ~0.4)
- Wrong phase: correlation with error signal = 0.02 (near zero, should be negative)
- Result: Adds uncorrelated noise instead of cancellation

## File Groups

### BUG_HUNT_*.wav (3 files)
- Tests both sign conventions (err + y_ctrl vs err - y_ctrl)
- User confirmed: "both files 2 and 3 are LOUDER"
- Proves y_ctrl is uncorrelated with err

### DIAGNOSTIC_*.wav (3 files)
- Early diagnostic comparisons
- before.wav, after.wav, antinoise.wav

### FINAL_TEST_*.wav (6 files)
- Tests 3 normalization methods: peak, RMS, raw
- User confirmed: "all 3 normalization methods have LOUD outputs"
- Eliminated normalization as the issue

### FREQ_ANALYSIS_*.wav (3 files)
- Full-band vs low-pass filtered residual
- Proved high frequencies increased 676x
- Model adds massive high-freq noise

### METRIC_TEST_*.wav (2 files)
- Audio from exact arrays used in MSE calculation
- User confirmed: "residual sounds louder"
- Verified metrics match perception

### MODEL_OUTPUT_u.wav (1 file)
- Raw model control signal output
- RMS 0.524 (too loud by 2.5x)
- Frequency: 90% engine band but absolute energy too high

### POLARITY_TEST_*.wav (3 files)
- Tests current h_s vs flipped h_s
- User: "both files are LOAD and equal"
- Proves h_s sign is correct, model learned wrong

### SIMPLE_FIX_*.wav (3 files)
- Post-processing fix attempts
- Best: -0.08 dB at 0.10x amplitude scaling
- Proves model cannot be salvaged

## Key Findings Summary

**Metrics:**
- Full-band: -3.74 dB (2.36x amplification)
- Engine (50-400 Hz): -3.56 dB
- High (2000-8000 Hz): -28.30 dB (676x amplification!)

**Correlation:**
- u ↔ ref: +0.52 (model learned something)
- err ↔ y_ctrl: +0.02 (but wrong relationship)

**Conclusion:**
Model cannot be fixed with post-processing. Requires complete rebuild with:
- Output amplitude constraints
- Frequency-weighted loss
- Better architecture

## Documentation
Full analysis in:
- `COMPLETE_DIAGNOSTIC_REPORT_DEC21.md`
- `ROOT_CAUSE_ANALYSIS.md`
- `DIAGNOSTIC_SUMMARY.md`

## Date
December 21, 2025
