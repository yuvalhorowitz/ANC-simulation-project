"""
Final v6 Dual-Band Blind Test
Evaluates the model on unseen Scenarios 45-49.
Isolates 50-800Hz (Drone) and 800-2000Hz (Precision) performance.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TEST_SCENARIOS = range(45, 50) 
MODEL_PATH = 'anc_v6_dual_band.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Re-register Dual-Band Loss for Loading ---
@tf.keras.utils.register_keras_serializable()
class DualBandPrecisionLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        idx_800 = int(800 / (fs / window_size))
        idx_2000 = int(2000 / (fs / window_size))
        weights[:idx_800] = 10.0
        weights[idx_800:idx_2000] = 60.0
        weights[idx_2000:] = 0.01
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error_fft = tf.signal.rfft(y_true[:, :, 0] - y_pred[:, :, 0])
        return tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)

def run_v6_blind_test():
    print(f"--- Loading Final v6 Dual-Band Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'DualBandPrecisionLoss': DualBandPrecisionLoss})

    scenario_results = []
    
    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

        # Standard Pre-processing
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Sliding window inference
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        # Physical Superposition
        rec, gen = mic[WINDOW_SIZE:], u_pred[:, -1, 0]
        res = rec + gen
        
        # Band-Specific Metrics
        def get_band_db(orig, resid, f_range):
            f, p1 = welch(orig, FS, nperseg=1024)
            _, p2 = welch(resid, FS, nperseg=1024)
            mask = (f >= f_range[0]) & (f <= f_range[1])
            return 10 * np.log10(np.sum(p1[mask]) / np.sum(p2[mask]))

        db_low = get_band_db(rec, res, [50, 800])
        db_high = get_band_db(rec, res, [800, 2000])
        total_red = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
        
        scenario_results.append((rec, res, total_red, db_low, db_high, scn_id))
        print(f"Scenario {scn_id}: Total={total_red:.1f}dB | 50-800Hz={db_low:.1f}dB | 800-2000Hz={db_high:.1f}dB")

    # --- 3. Visualization ---
    scenario_results.sort(key=lambda x: x[2])
    rec, res, total, low, high, sid = scenario_results[len(scenario_results)//2] # Median

    fig, axs = plt.subplots(3, 1, figsize=(15, 14))

    # Plot 1: Phase Alignment Check
    axs[0].plot(rec[2000:2600], label="Original Noise", alpha=0.5)
    axs[0].plot(res[2000:2600], label="Residual (v6)", color='green', lw=2)
    axs[0].set_title(f"v6 Dual-Band: Typical Blind Performance (Scn {sid})\nLow-Band: {low:.1f} dB | High-Band: {high:.1f} dB")
    axs[0].legend(); axs[0].grid(True)

    # Plot 2: Spectral Harmonic Suppression
    
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    axs[1].plot(f, 10*np.log10(p_orig+1e-12), label="Original", alpha=0.4)
    axs[1].plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled (v6)", color='green')
    axs[1].axvspan(50, 800, color='blue', alpha=0.05, label='Drone Band')
    axs[1].axvspan(800, 2000, color='red', alpha=0.05, label='Precision Band')
    axs[1].set_title("PSD: Harmonic Isolation Test (Unseen Data)"); axs[1].set_xlim(0, 2200); axs[1].legend()

    # Plot 3: Stability Bar Chart
    all_totals = [r[2] for r in scenario_results]
    axs[2].bar([f"Scn {r[5]}" for r in scenario_results], all_totals, color='skyblue')
    axs[2].axhline(y=np.mean(all_totals), color='red', linestyle='--', label=f"Avg: {np.mean(all_totals):.1f} dB")
    axs[2].set_title("Aggregate Spatial Stability"); axs[2].set_ylabel("dB Reduction"); axs[2].legend()

    plt.tight_layout()
    plt.savefig("v6_blind_test_dual_band_report.png")
    print(f"\n--- Final v6 Average Reduction: {np.mean(all_totals):.2f} dB ---")

if __name__ == "__main__":
    run_v6_blind_test()