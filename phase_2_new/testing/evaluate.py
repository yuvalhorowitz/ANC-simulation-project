# phase_2_new/testing/evaluate.py
"""
Evaluation script for Phase 2 New experiments.

Computes full metrics including band-limited reduction for comparison
with project goals.

Usage:
    python -m phase_2_new.testing.evaluate --experiment exp_00_baseline
"""

from __future__ import annotations

import os
import argparse
import json
from typing import Dict, List

import numpy as np
import tensorflow as tf
from scipy.signal import fftconvolve

from phase_2_new.utils.dataset_builder import build_training_example, list_scenarios


def compute_band_energy(signal: np.ndarray, fs: int, band: tuple) -> float:
    """Compute energy in a frequency band using FFT."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)

    # Find indices in band
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

    # Energy in band
    energy = np.sum(np.abs(fft[idx]) ** 2)
    return float(energy)


def evaluate_scenario(
    model: tf.keras.Model,
    scenario_prefix: str,
    fs: int,
    window_length: int,
) -> Dict:
    """Evaluate one scenario and return detailed metrics."""

    # Load data
    X, err_t, h_s = build_training_example(scenario_prefix, fs, window_length)

    # Predict control signal
    u = model.predict(X, batch_size=512, verbose=0).flatten()

    # Apply secondary path
    y_ctrl = fftconvolve(u, h_s, mode="same").astype(np.float32)

    # Align lengths
    err = err_t.flatten()[:len(y_ctrl)]
    residual = err + y_ctrl

    # Full-band metrics
    mse_before = float(np.mean(err ** 2))
    mse_after = float(np.mean(residual ** 2))
    delta_full = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

    # Band-limited metrics
    bands = {
        "engine_50_400": (50, 400),
        "road_20_2000": (20, 2000),
        "cabin_0_1000": (0, 1000),
    }

    band_metrics = {}
    for name, (f_low, f_high) in bands.items():
        e_before = compute_band_energy(err, fs, (f_low, f_high))
        e_after = compute_band_energy(residual, fs, (f_low, f_high))
        delta_band = 10.0 * np.log10((e_before + 1e-12) / (e_after + 1e-12))
        band_metrics[name] = delta_band

    return {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "delta_full": delta_full,
        **band_metrics,
    }


def evaluate_experiment(experiment_id: str, val_dir: str = "data/val"):
    """Evaluate an experiment on validation set."""

    model_path = f"phase_2_new/models/{experiment_id}.keras"
    results_path = f"phase_2_new/results/{experiment_id}"

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_id}")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Get window length from model input shape
    window_length = model.input_shape[1]
    fs = 16000

    # Load validation scenarios
    val_prefixes = list_scenarios(val_dir)
    print(f"Found {len(val_prefixes)} validation scenarios\n")

    # Evaluate each scenario
    all_results = []
    for i, prefix in enumerate(val_prefixes):
        tag = os.path.basename(prefix)
        results = evaluate_scenario(model, prefix, fs, window_length)
        all_results.append(results)

        print(
            f"[{i:02d}] {tag:20s} | "
            f"Full={results['delta_full']:5.2f} dB | "
            f"Engine={results['engine_50_400']:5.2f} dB | "
            f"Road={results['road_20_2000']:5.2f} dB | "
            f"Cabin={results['cabin_0_1000']:5.2f} dB"
        )

    # Aggregate statistics
    print(f"\n{'='*60}")
    print("Aggregate Statistics")
    print(f"{'='*60}")

    metrics = {
        "mse_before": [r["mse_before"] for r in all_results],
        "mse_after": [r["mse_after"] for r in all_results],
        "delta_full": [r["delta_full"] for r in all_results],
        "engine_50_400": [r["engine_50_400"] for r in all_results],
        "road_20_2000": [r["road_20_2000"] for r in all_results],
        "cabin_0_1000": [r["cabin_0_1000"] for r in all_results],
    }

    for name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name:20s}: {mean_val:7.2f} ± {std_val:5.2f}")

    # Save results
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "metrics.json")

    with open(results_file, 'w') as f:
        json.dump({
            "experiment_id": experiment_id,
            "per_scenario": [
                {
                    "scenario": os.path.basename(val_prefixes[i]),
                    **all_results[i]
                }
                for i in range(len(all_results))
            ],
            "aggregate": {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for name, values in metrics.items()
            }
        }, f, indent=2)

    print(f"\nResults saved: {results_file}")
    print(f"{'='*60}\n")

    # Compare to goals
    engine_mean = np.mean(metrics["engine_50_400"])
    road_mean = np.mean(metrics["road_20_2000"])
    cabin_mean = np.mean(metrics["cabin_0_1000"])

    print("Comparison to Project Goals:")
    print(f"  Engine (50-400Hz):  {engine_mean:5.2f} dB / 15.00 dB goal  {'✓' if engine_mean >= 15 else '✗'}")
    print(f"  Road (20-2000Hz):   {road_mean:5.2f} dB / 10.00 dB goal  {'✓' if road_mean >= 10 else '✗'}")
    print(f"  Cabin (0-1000Hz):   {cabin_mean:5.2f} dB / 10.00 dB goal  {'✓' if cabin_mean >= 10 else '✗'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 New experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment ID to evaluate (e.g., exp_00_baseline)"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/val",
        help="Validation data directory"
    )
    args = parser.parse_args()

    evaluate_experiment(args.experiment, args.val_dir)


if __name__ == "__main__":
    main()
