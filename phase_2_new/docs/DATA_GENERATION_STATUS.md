# Data Generation Status

## Issue Encountered

When attempting to run the enhanced data generation script, we encountered a Python environment issue:
- System python3 does not have required packages (pyroomacoustics, scipy, matplotlib, tensorflow)
- Proxy issues prevent package installation via pip

## Current Data Status

**Existing Training Data**: 10 scenarios
```bash
$ ls data/train/*.wav | wc -l
30  # (10 scenarios × 3 wav files each: _ref.wav, _err.wav, _mic.wav)
```

**Existing Validation Data**: Need to check
```bash
$ ls data/val/*.wav | wc -l
```

## Options to Proceed

### Option 1: Use Existing Python Environment (Recommended)
You likely have a conda/venv environment that was used to generate the existing data.

**Steps**:
1. Activate your Python environment:
   ```bash
   # If using conda:
   conda activate anc-env  # (or whatever your env is named)

   # If using venv:
   source venv/bin/activate  # (or path to your venv)
   ```

2. Verify packages:
   ```bash
   python --version
   pip list | grep -E "(pyroomacoustics|numpy|scipy|tensorflow)"
   ```

3. Run data generation:
   ```bash
   python -m phase_2_new.utils.generate_dataset_enhanced
   ```

### Option 2: Generate Data Manually with simulation_setup.py

Modify the existing `simulation_setup.py` to generate more scenarios:

```python
# At the bottom of simulation_setup.py, change:
if __name__ == "__main__":
    generate_dataset(base_out_dir="data", split="train",
                     num_scenarios=50,  # Changed from 10
                     duration_seconds=4.0,
                     randomize_positions=True)
    generate_dataset(base_out_dir="data", split="val",
                     num_scenarios=10,  # Changed from 3
                     duration_seconds=4.0,
                     randomize_positions=True)
```

Then run:
```bash
python simulation_setup.py  # Using your configured Python environment
```

### Option 3: Start with Existing 10 Scenarios

We can start experiments with the existing 10 training scenarios:
- It's suboptimal but allows us to test our architecture improvements
- Once we validate the approach works, we can generate more data
- This follows the "start small, scale up" principle

## Recommendation

**Start with Option 3** to avoid blocking progress:

1. ✓ Use existing 10 training scenarios
2. ✓ Implement and test improved architectures
3. ✓ See if wider/deeper networks help with limited data
4. Later: Generate 50 scenarios once environment is sorted

This way we can:
- Test if our architectural improvements work
- Validate the experimental framework
- Identify which approach is most promising
- Then scale up data generation for final training

## Next Steps

Do you want to:
- **A)** Try to activate your Python environment and generate 50 scenarios now
- **B)** Proceed with 10 scenarios and focus on architecture improvements first
- **C)** Manually modify simulation_setup.py and generate data yourself

Let me know and I'll adjust the plan accordingly!
