# ROOT CAUSE FOUND: Acoustic Complexity from Reflections

## Problem Summary

The newly trained baseline model achieved **-5.32 dB** (amplifying noise!) instead of the expected +5-7 dB reduction.

## Root Cause: MAX_ORDER Mismatch

**Original phase_2 (successful):**
```python
MAX_ORDER = 0  # Anechoic (no wall reflections)
Result: +3.62 dB engine band reduction ✓
```

**New phase_2_new (failed):**
```python
MAX_ORDER_OPTIONS = [0, 1, 2, 3]  # Mixed reflections
Result: -5.32 dB engine band (AMPLIFYING!) ✗
```

##Diagnostic Results

### 1. Old Model on New Data: -44.94 dB
- Control signal: `u ∈ [-60.58, 12.41]` (100x too large!)
- Old model completely incompatible with new acoustic conditions

### 2. New Model on New Data: -1.50 dB
- Control signal: `u ∈ [-0.32, 0.37]` ✓ (appropriate scale)
- Model learned something, but still negative overall

### 3. New Model on Training Data: -2.21 ± 2.54 dB
- Model FAILS on its own training data!
- Training loss (0.097) = -2.87 dB equivalent
- Model "converged" to a BAD solution

### 4. Per-Scenario Analysis:
```
Scenario    Reduction    h_s length    Note
--------    ---------    ----------    ----
Val 0       -1.50 dB     527 samples   Long RIR
Val 1       +1.90 dB     94 samples    Short RIR - ONLY positive result!
Val 2       -6.02 dB     515 samples   Long RIR
Train 0     -0.85 dB     316 samples
Train 1     -5.77 dB     (not checked)
Train 2     -0.02 dB     (not checked)
```

**Pattern**: Shorter RIRs tend to work better. Val 1 (94 samples) is the ONLY positive result!

## Why Reflections Cause Failure

### Anechoic (MAX_ORDER=0):
- Direct path only
- Simple time delay and scaling
- Linear relationship: `err ≈ α * delay(ref)`
- Easy for 3-layer TCN to learn inverse

### With Reflections (MAX_ORDER=1-3):
- Multiple paths (direct + reflections)
- Complex interference patterns
- Non-linear: `err = Σ(delayed, scaled, interfered paths)`
- Too complex for simple TCN to learn inverse
- Different scenarios have vastly different RIR characteristics (94 to 527 samples!)

## Solution: Start with Anechoic Data

### Step 1: Regenerate Data (ANECHOIC)

```bash
cd "/Users/adminaccount/Documents/personal/Finals Project/ANC-simulation-project"
source venv/bin/activate

# Generate 50 train + 10 val scenarios with MAX_ORDER=0
python simulation_setup_phase_2_anechoic.py
```

**Expected time**: 10-20 minutes

### Step 2: Train Baseline

```bash
python -m phase_2_new.training.train --config baseline
```

**Expected result**: Should achieve **+5-7 dB** (or similar to old +3.62 dB)

### Step 3: Evaluate

```bash
python -m phase_2_new.testing.evaluate --experiment exp_00_baseline
```

**Expected**: Positive dB across all bands!

## Why This Will Work

1. **Matches successful phase_2 conditions**: MAX_ORDER=0 like original
2. **Model capacity sufficient**: 3-layer TCN can handle anechoic
3. **Proven approach**: Original phase_2 achieved +3.62 dB with anechoic
4. **Val 1 proved it**: The ONE scenario with shortest RIR got +1.90 dB

## Future Work: Gradually Add Complexity

Once anechoic works:

### Approach 1: Larger Model First
- Train deeper/wider TCN on anechoic data
- Verify it still works well
- THEN try adding reflections

### Approach 2: Curriculum Learning
- Start with 100% MAX_ORDER=0
- Add 25% MAX_ORDER=1
- Gradually increase reflection complexity
- Monitor when performance degrades

### Approach 3: Reflection-Aware Architecture
- Add explicit RIR encoding to model
- Learn to condition on room characteristics
- May handle variable acoustics better

## Key Lessons Learned

1. **More complex data ≠ Better**: Complex reflections made problem too hard
2. **Diagnostics are critical**: Without testing old model on new data, we wouldn't have found this
3. **Start simple**: Anechoic → small reflections → full reflections
4. **Val 1 was the clue**: The only working scenario had the shortest RIR

## Files Created

- `simulation_setup_phase_2_anechoic.py` - Fixed data generator (MAX_ORDER=0)
- `DIAGNOSTIC_SUMMARY.md` - Initial investigation
- `diagnostic_step1.py` - Test old model on new data
- `diagnostic_step1_corrected.py` - With proper normalization
- `compare_models.py` - Compare old vs new model outputs
- `test_train_vs_val.py` - Test model on training vs validation
- `check_rir_causality.py` - Analyze RIR properties
- `compare_audio_loading.py` - Verify audio loading consistency

## Summary

**Problem**: Reflections (MAX_ORDER 1-3) made acoustic problem too complex for simple TCN
**Solution**: Use MAX_ORDER=0 (anechoic) like original successful phase_2
**Expected**: Positive dB reduction matching or exceeding original +3.62 dB
**Next Step**: Run `python simulation_setup_phase_2_anechoic.py`
