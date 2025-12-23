#!/usr/bin/env python3
"""
Test NEW model on training data vs validation data
"""

import numpy as np
import tensorflow as tf
from scipy.signal import fftconvolve
from scipy.io import wavfile

def test_scenario(model, scenario_path, name):
    """Test model on one scenario"""
    # Load and normalize
    fs, ref = wavfile.read(f'{scenario_path}_ref.wav')
    _, err = wavfile.read(f'{scenario_path}_err.wav')
    h_s_raw = np.load(f'{scenario_path}_rir_spk_to_err.npy')

    ref = ref.astype(np.float32) / 32767.0
    err = err.astype(np.float32) / 32767.0
    h_s = h_s_raw.astype(np.float32)

    ref = ref / (np.max(np.abs(ref)) + 1e-9)
    err = err / (np.max(np.abs(err)) + 1e-9)
    h_s = h_s / (np.max(np.abs(h_s)) + 1e-9)

    # Build windows
    window_length = 200
    n_samples = len(ref) - window_length + 1
    X = np.zeros((n_samples, window_length, 1), dtype=np.float32)
    for t in range(n_samples):
        X[t, :, 0] = ref[t : t + window_length]

    # Predict
    u = model.predict(X, batch_size=512, verbose=0).flatten()

    # Apply secondary path
    y_ctrl = fftconvolve(u, h_s, mode='same')
    err_t = err[:len(y_ctrl)]
    residual = err_t + y_ctrl

    # Metrics
    mse_before = np.mean(err_t ** 2)
    mse_after = np.mean(residual ** 2)
    delta_dB = 10.0 * np.log10(mse_before / mse_after)

    print(f"\n{name}:")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  u mean: {u.mean():.6f}, std: {u.std():.6f}")
    print(f"  MSE before: {mse_before:.6e}")
    print(f"  MSE after:  {mse_after:.6e}")
    print(f"  Reduction:  {delta_dB:+.2f} dB")

    return delta_dB

print("="*60)
print("Test NEW Model: Training vs Validation Data")
print("="*60)

# Load model
print("\nLoading NEW model...")
model = tf.keras.models.load_model('phase_2_new/models/exp_00_baseline.keras', compile=False)
print("✓ Model loaded")

# Test on 3 training scenarios
print("\n" + "="*60)
print("TRAINING SCENARIOS:")
print("="*60)
train_results = []
for i in range(3):
    scenario = f'data/train/train_scenario_{i:03d}'
    delta = test_scenario(model, scenario, f"Train {i}")
    train_results.append(delta)

# Test on 3 validation scenarios
print("\n" + "="*60)
print("VALIDATION SCENARIOS:")
print("="*60)
val_results = []
for i in range(3):
    scenario = f'data/val/val_scenario_{i:03d}'
    delta = test_scenario(model, scenario, f"Val {i}")
    val_results.append(delta)

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Training:   {np.mean(train_results):+.2f} ± {np.std(train_results):.2f} dB")
print(f"Validation: {np.mean(val_results):+.2f} ± {np.std(val_results):.2f} dB")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
if np.mean(train_results) < 0:
    print("⚠️  Model gets NEGATIVE dB even on TRAINING data!")
    print("   → Problem is with DATA GENERATION, not overfitting")
    print("   → Something is fundamentally wrong with how data is created")
elif np.mean(val_results) < 0 and np.mean(train_results) > 0:
    print("✓ Model works on training data but fails on validation")
    print("  → Severe overfitting OR train/val distributions are different")
else:
    print("✓ Model works on both training and validation")
    print("  → Evaluation code might have issues")
print("="*60 + "\n")
