# phase_2/testing/test_ref2u.py
"""
Phase 2 Testing Script (Option D)

Evaluates a trained Ref->TCN->u controller by measuring:
  - Full-band noise reduction
  - Band-limited noise reduction (PDF requirements)

Metrics:
  MSE_before, MSE_after
  ΔE_full (dB)
  ΔE_engine (50–400 Hz)
  ΔE_road   (20–2000 Hz)
  ΔE_cabin  (<=1000 Hz)
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from scipy.signal import fftconvolve
from scipy.fft import rfft, rfftfreq

from phase_2.models.tcn_ref2u import build_tcn_ref2u, TCNConfig
from phase_2.utils.dataset_builder import build_training_example, list_scenarios


# -----------------------
# Configuration
# -----------------------
FS = 16000
WINDOW_LENGTH = 200
MODEL_PATH = "phase_2/models/tcn_ref2u.keras"
VAL_DIR = "data/val"


# -----------------------
# Utility functions
# -----------------------
def band_energy(signal: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    """
    Compute band-limited energy using FFT.
    """
    spec = np.abs(rfft(signal)) ** 2
    freqs = rfftfreq(len(signal), d=1 / fs)

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return np.sum(spec[mask])


def evaluate_scenario(
    model: tf.keras.Model,
    scenario_prefix: str,
) -> dict:
    """
    Evaluate one validation scenario.
    """

    # Load aligned data
    X, err_t, h_s = build_training_example(
        scenario_prefix,
        fs=FS,
        window_length=WINDOW_LENGTH,
    )

    # Predict control signal u(t)
    u = model.predict(X, batch_size=512, verbose=0).flatten()

    # Apply secondary path
    y_ctrl = fftconvolve(u, h_s, mode="same")

    # Align error signal
    err = err_t.flatten()[: len(y_ctrl)]
    residual = err + y_ctrl

    # ---- Full-band metrics ----
    mse_before = np.mean(err**2)
    mse_after = np.mean(residual**2)
    delta_full = 10 * np.log10(mse_before / mse_after)

    # ---- Band-limited metrics ----
    bands = {
        "engine_50_400": (50, 400),
        "road_20_2000": (20, 2000),
        "cabin_0_1000": (0, 1000),
    }

    band_results = {}
    for name, (f_lo, f_hi) in bands.items():
        e_before = band_energy(err, FS, f_lo, f_hi)
        e_after = band_energy(residual, FS, f_lo, f_hi)
        band_results[name] = 10 * np.log10(e_before / e_after)

    return {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "delta_full": delta_full,
        **band_results,
    }


# -----------------------
# Main testing routine
# -----------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load trained model
    model_cfg = TCNConfig(sequence_length=WINDOW_LENGTH)
    model = build_tcn_ref2u(model_cfg)
    model.load_weights(MODEL_PATH)

    scenarios = list_scenarios(VAL_DIR)
    if not scenarios:
        raise RuntimeError("No validation scenarios found.")

    print(f"Loaded {len(scenarios)} validation scenarios.\n")

    results = []

    for i, prefix in enumerate(scenarios):
        tag = os.path.basename(prefix)
        r = evaluate_scenario(model, prefix)
        results.append(r)

        print(
            f"[{i:02d}] {tag} | "
            f"ΔE_full={r['delta_full']:.2f} dB | "
            f"Engine={r['engine_50_400']:.2f} dB | "
            f"Road={r['road_20_2000']:.2f} dB | "
            f"Cabin={r['cabin_0_1000']:.2f} dB"
        )

    # ---- Aggregate statistics ----
    print("\n=== Aggregate over validation set ===")
    for key in results[0].keys():
        vals = np.array([r[key] for r in results])
        print(f"{key:>18}: mean={np.mean(vals):6.2f} | std={np.std(vals):6.2f}")


if __name__ == "__main__":
    # Run from project root:
    #   python -m phase_2.testing.test_ref2u
    main()
