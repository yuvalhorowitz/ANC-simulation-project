# Quick Start: Generate 50 Training Scenarios

## ✓ Script Created: simulation_setup_phase_2.py

A standalone script that generates 50 diverse training scenarios and 10 validation scenarios for Phase 2 New.

## How to Run

### Step 1: Activate Your Python Environment

Your environment must have:
- pyroomacoustics
- scipy
- numpy

```bash
# Example (adjust for your environment):
conda activate your-env-name
# or
source venv/bin/activate
```

### Step 2: Navigate to Project Root

```bash
cd /Users/adminaccount/Documents/personal/Finals\ Project/ANC-simulation-project
```

### Step 3: Run the Script

```bash
python simulation_setup_phase_2.py
```

**Expected time:** 10-20 minutes

**What it does:**
- Replaces existing 10 scenarios with fresh 50 training scenarios
- Replaces existing 3 scenarios with fresh 10 validation scenarios
- More diverse noise types (engine, road, wind, AC, mixed)
- Variable room acoustics (different sizes and reflections)
- Progress updates every 10 scenarios

## Step 4: Verify Data

```bash
# Count training scenarios (should be 50)
ls data/train/*_ref.wav | wc -l

# Count validation scenarios (should be 10)
ls data/val/*_ref.wav | wc -l

# Check metadata
cat data/train/train_metadata.json | head -50
```

## Step 5: Train with New Data

Once data generation completes:

```bash
# Train baseline
python -m phase_2_new.training.train --config baseline

# Evaluate
python -m phase_2_new.testing.evaluate --experiment exp_00_baseline
```

## What's Different from Original 10 Scenarios?

### Original (10 scenarios):
- Limited noise variety
- Fixed room configurations
- Resulted in 3.6 dB engine band reduction

### New (50 scenarios):
- **5 noise types** with varied characteristics:
  - Engine: 40-150 Hz harmonics (idle to highway)
  - Road: Variable filtered broadband
  - Wind: High-pass with gusts
  - AC: 60Hz harmonics + air noise
  - Mixed: Combinations of above
- **Variable room dimensions**: 3.5-5.0m × 2.5-3.5m × 1.8-2.2m
- **Mixed acoustic conditions**: Reflection orders 0-3
- **Random positioning**: More realistic position distributions

**Expected improvement:**
- Better generalization
- Reduced overfitting
- Likely 5-7 dB baseline (vs 3.6 dB)
- Foundation for reaching 15 dB target

## Troubleshooting

**If you get "ModuleNotFoundError: No module named 'pyroomacoustics'":**
- Your Python environment isn't activated
- Or packages aren't installed in that environment

**If script is slow:**
- Normal! 50 scenarios take time
- Each scenario: 4 seconds of audio + room simulation
- Watch progress updates every 10 scenarios

**If you want to stop:**
- Press Ctrl+C
- Data generated so far will be saved
- You can resume or start fresh

## After Data Generation

You'll be ready to:
1. Train models with 5x more data
2. Test if architectural improvements (wider/deeper) help
3. Work toward the 15 dB engine band target
4. Document which approaches work best

---

**Ready?** Activate your environment and run:
```bash
python simulation_setup_phase_2.py
```
