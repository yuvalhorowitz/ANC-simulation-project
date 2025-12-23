#!/usr/bin/env python3
"""
TEST SIMPLE FIXES - Can we fix the model output without retraining?

Hypothesis: u(t) is too loud and has 0.02% high-freq leakage
Fix 1: Scale down u(t) to match err amplitude
Fix 2: Low-pass filter u(t) to remove high-freq component
Fix 3: Both

This tests if simple post-processing can salvage the trained model.
"""

import numpy as np
from scipy.signal import fftconvolve, butter, filtfilt
from scipy.io import wavfile
import tensorflow as tf

print("="*70)
print("TEST SIMPLE FIXES")
print("="*70)

scenario = 'data/val/val_scenario_000'

# Load data
fs, ref = wavfile.read(f'{scenario}_ref.wav')
_, err = wavfile.read(f'{scenario}_err.wav')
h_s = np.load(f'{scenario}_rir_spk_to_err.npy')

ref = ref.astype(np.float32) / 32767.0 / (np.max(np.abs(ref.astype(np.float32) / 32767.0)) + 1e-9)
err = err.astype(np.float32) / 32767.0 / (np.max(np.abs(err.astype(np.float32) / 32767.0)) + 1e-9)
h_s = h_s.astype(np.float32) / (np.max(np.abs(h_s)) + 1e-9)

# Get model prediction
model = tf.keras.models.load_model('phase_2_new/models/exp_00_baseline.keras', compile=False)

window_length = 200
n_samples = len(ref) - window_length + 1
X = np.zeros((n_samples, window_length, 1), dtype=np.float32)
for t in range(n_samples):
    X[t, :, 0] = ref[t : t + window_length]

u_original = model.predict(X, batch_size=512, verbose=0).flatten()

# Baseline
y_ctrl_original = fftconvolve(u_original, h_s, mode='same')
err_aligned = err[:len(y_ctrl_original)]
residual_original = err_aligned + y_ctrl_original
mse_err = np.mean(err_aligned ** 2)
mse_original = np.mean(residual_original ** 2)
dB_original = 10 * np.log10(mse_err / mse_original)

print(f"\nBASELINE (no fixes):")
print(f"  u RMS: {np.sqrt(np.mean(u_original**2)):.6f}")
print(f"  Reduction: {dB_original:+.2f} dB")

# Fix 1: Scale down u(t) to match err RMS
u_rms = np.sqrt(np.mean(u_original ** 2))
err_rms = np.sqrt(mse_err)
scale_factor = err_rms / (u_rms + 1e-9)

print(f"\n" + "="*70)
print(f"FIX 1: Scale down u(t) to match err amplitude")
print(f"="*70)
print(f"  Scale factor: {scale_factor:.4f}")

u_scaled = u_original * scale_factor
y_ctrl_scaled = fftconvolve(u_scaled, h_s, mode='same')
residual_scaled = err_aligned + y_ctrl_scaled
mse_scaled = np.mean(residual_scaled ** 2)
dB_scaled = 10 * np.log10(mse_err / mse_scaled)

print(f"  u RMS after scaling: {np.sqrt(np.mean(u_scaled**2)):.6f}")
print(f"  Reduction: {dB_scaled:+.2f} dB")
print(f"  Improvement: {dB_scaled - dB_original:+.2f} dB")

# Fix 2: Low-pass filter u(t) at 500 Hz (above engine band)
print(f"\n" + "="*70)
print(f"FIX 2: Low-pass filter u(t) at 500 Hz")
print(f"="*70)

nyq = fs / 2
cutoff = 500 / nyq
b, a = butter(4, cutoff, btype='low')

u_filtered = filtfilt(b, a, u_original)
y_ctrl_filtered = fftconvolve(u_filtered, h_s, mode='same')
residual_filtered = err_aligned + y_ctrl_filtered
mse_filtered = np.mean(residual_filtered ** 2)
dB_filtered = 10 * np.log10(mse_err / mse_filtered)

print(f"  u RMS after filtering: {np.sqrt(np.mean(u_filtered**2)):.6f}")
print(f"  Reduction: {dB_filtered:+.2f} dB")
print(f"  Improvement: {dB_filtered - dB_original:+.2f} dB")

# Fix 3: Both scaling AND filtering
print(f"\n" + "="*70)
print(f"FIX 3: Both scale AND low-pass filter")
print(f"="*70)

u_both = filtfilt(b, a, u_original) * scale_factor
y_ctrl_both = fftconvolve(u_both, h_s, mode='same')
residual_both = err_aligned + y_ctrl_both
mse_both = np.mean(residual_both ** 2)
dB_both = 10 * np.log10(mse_err / mse_both)

print(f"  u RMS: {np.sqrt(np.mean(u_both**2)):.6f}")
print(f"  Reduction: {dB_both:+.2f} dB")
print(f"  Improvement: {dB_both - dB_original:+.2f} dB")

# Fix 4: Optimize scale factor
print(f"\n" + "="*70)
print(f"FIX 4: Optimize scale factor (grid search)")
print(f"="*70)

best_dB = dB_original
best_scale = 1.0

for scale in np.linspace(0.1, 2.0, 20):
    u_test = u_original * scale
    y_test = fftconvolve(u_test, h_s, mode='same')
    res_test = err_aligned + y_test
    mse_test = np.mean(res_test ** 2)
    dB_test = 10 * np.log10(mse_err / mse_test)

    if dB_test > best_dB:
        best_dB = dB_test
        best_scale = scale

print(f"  Best scale: {best_scale:.4f}")
print(f"  Best reduction: {best_dB:+.2f} dB")
print(f"  Improvement: {best_dB - dB_original:+.2f} dB")

# Apply best fix
u_best = u_original * best_scale
y_ctrl_best = fftconvolve(u_best, h_s, mode='same')
residual_best = err_aligned + y_ctrl_best

# Fix 5: Optimize scale + filter
print(f"\n" + "="*70)
print(f"FIX 5: Optimize scale + low-pass filter")
print(f"="*70)

best_dB_filt = dB_original
best_scale_filt = 1.0

for scale in np.linspace(0.1, 2.0, 20):
    u_test = filtfilt(b, a, u_original) * scale
    y_test = fftconvolve(u_test, h_s, mode='same')
    res_test = err_aligned + y_test
    mse_test = np.mean(res_test ** 2)
    dB_test = 10 * np.log10(mse_err / mse_test)

    if dB_test > best_dB_filt:
        best_dB_filt = dB_test
        best_scale_filt = scale

print(f"  Best scale: {best_scale_filt:.4f}")
print(f"  Best reduction: {best_dB_filt:+.2f} dB")
print(f"  Improvement: {best_dB_filt - dB_original:+.2f} dB")

u_best_filt = filtfilt(b, a, u_original) * best_scale_filt
y_ctrl_best_filt = fftconvolve(u_best_filt, h_s, mode='same')
residual_best_filt = err_aligned + y_ctrl_best_filt

# Summary
print(f"\n" + "="*70)
print(f"SUMMARY:")
print(f"="*70)

results = [
    ("Baseline (no fix)", dB_original),
    ("Scale to err RMS", dB_scaled),
    ("Low-pass 500Hz", dB_filtered),
    ("Scale + Filter", dB_both),
    (f"Optimized scale ({best_scale:.2f}x)", best_dB),
    (f"Optimized scale + filter ({best_scale_filt:.2f}x)", best_dB_filt),
]

print(f"\n{'Method':<40} {'dB Reduction':<15} {'vs Baseline':<15}")
print("-"*70)
for name, dB in results:
    improvement = dB - dB_original
    print(f"{name:<40} {dB:>+14.2f} {improvement:>+14.2f}")

# Find best overall
best_method_idx = np.argmax([r[1] for r in results])
best_method_name, best_method_dB = results[best_method_idx]

print(f"\n" + "="*70)
print(f"BEST METHOD: {best_method_name}")
print(f"Reduction: {best_method_dB:+.2f} dB (improvement: {best_method_dB - dB_original:+.2f} dB)")
print(f"="*70)

# Generate audio for best method
if best_method_idx == 4:
    residual_best_audio = residual_best
    method_suffix = "optimized_scale"
else:
    residual_best_audio = residual_best_filt
    method_suffix = "optimized_scale_filter"

# Save audio comparison
global_peak = max(np.max(np.abs(err_aligned)), np.max(np.abs(residual_original)), np.max(np.abs(residual_best_audio)))
scale_audio = 32767 * 0.95 / (global_peak + 1e-9)

err_audio = (err_aligned * scale_audio).astype(np.int16)
residual_original_audio = (residual_original * scale_audio).astype(np.int16)
residual_fixed_audio = (residual_best_audio * scale_audio).astype(np.int16)

wavfile.write('SIMPLE_FIX_err.wav', fs, err_audio)
wavfile.write('SIMPLE_FIX_baseline.wav', fs, residual_original_audio)
wavfile.write(f'SIMPLE_FIX_{method_suffix}.wav', fs, residual_fixed_audio)

print(f"\n" + "="*70)
print(f"AUDIO FILES GENERATED:")
print(f"="*70)
print(f"  SIMPLE_FIX_err.wav - Original noise")
print(f"  SIMPLE_FIX_baseline.wav - Model output (amplified, {dB_original:+.2f} dB)")
print(f"  SIMPLE_FIX_{method_suffix}.wav - Fixed ({best_method_dB:+.2f} dB)")
print(f"")
print(f"Listen to verify the fix works!")

if best_method_dB > 0:
    print(f"\n" + "="*70)
    print(f"✓ SUCCESS! Simple post-processing gives POSITIVE reduction")
    print(f"="*70)
    print(f"\nThis proves:")
    print(f"  1. Model learned useful correlation with ref")
    print(f"  2. But output amplitude is wrong (too loud)")
    print(f"  3. Simple scaling fixes the problem!")
    print(f"")
    print(f"NEXT STEPS:")
    print(f"  1. Add output scaling/normalization to model architecture")
    print(f"  2. OR add amplitude penalty to training loss")
    print(f"  3. OR normalize u(t) by its own statistics before h_s convolution")
else:
    print(f"\n" + "="*70)
    print(f"✗ Even with fixes, still no positive reduction")
    print(f"="*70)
    print(f"  Model truly learned nothing useful - need to retrain from scratch")

print("="*70 + "\n")
