"""
Final v4 Blind Test: Evaluating the Regulated TCN
Evaluates Scenarios 45-49 with focus on Harmonic Suppression.
The goal is a 'Clean PSD' where the green line never crosses the blue line.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TEST_SCENARIOS = range(45, 50) 
MODEL_PATH = 'anc_v4_regulated_model.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Re-register Custom Loss for Loading ---
@tf.keras.utils.register_keras_serializable()
class SpectralRegulatedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        idx_engine = int(400 / (fs / window_size))
        idx_road = int(2000 / (fs / window_size))
        weights[:idx_engine] = 20.0
        weights[idx_engine:idx_road] = 40.0
        weights[idx_road:] = 0.05
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        cancellation_loss = tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)
        pred_fft = tf.signal.rfft(y_pred[:, :, 0])
        energy_penalty = tf.reduce_mean(tf.square(tf.abs(pred_fft)) * self.weights) * 0.02
        return cancellation_loss + energy_penalty

# --- 3. Run Test Loop ---
def run_v4_blind_test():
    print(f"--- Loading Final v4 Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'SpectralRegulatedLoss': SpectralRegulatedLoss})

    results = []
    
    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

        # Prep (Filtered-X)
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Batch Inference
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        # Superposition in the Ear
        rec, gen = mic[WINDOW_SIZE:], u_pred[:, -1, 0]
        res = rec + gen
        reduction = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
        
        results.append((rec, res, reduction, scn_id))
        print(f"Scenario {scn_id}: {reduction:.2f} dB Reduction")

    # --- 4. Deep Visualization ---
    # Sort results to visualize the median (typical) performance
    results.sort(key=lambda x: x[2])
    rec, res, red, sid = results[len(results)//2]

    plt.figure(figsize=(15, 14))

    # Plot A: Time Domain Alignment
    plt.subplot(3, 1, 1)
    plt.plot(rec[2000:2600], label="Original Noise", alpha=0.5)
    plt.plot(res[2000:2600], label="Residual (v4)", color='green', lw=2)
    plt.title(f"v4 Time Domain: Typical Blind Performance (Scn {sid}) - NR: {red:.2f} dB")
    plt.legend(); plt.grid(True)

    # Plot B: PSD suppression (Harmonic Check)
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    plt.subplot(3, 1, 2)
    plt.plot(f, 10*np.log10(p_orig+1e-12), label="Original Spectrum", alpha=0.5)
    plt.plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled Spectrum", color='green')
    plt.title("v4 Spectral Check: Suppression vs. Harmonics (0-2000Hz Target)")
    plt.xlim(0, 2200); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True)

    # Plot C: Stability Bar Chart
    plt.subplot(3, 1, 3)
    db_vals = [r[2] for r in results]
    plt.bar([f"Scn {r[3]}" for r in results], db_vals, color='skyblue')
    plt.axhline(y=np.mean(db_vals), color='red', linestyle='--', label=f"Avg: {np.mean(db_vals):.2f} dB")
    plt.title("Aggregate Spatial Stability across Unseen Head Positions")
    plt.ylabel("dB Reduction"); plt.legend()

    plt.tight_layout()
    plt.savefig("v4_blind_test_report.png")
    print(f"\n--- AVERAGE v4 BLIND REDUCTION: {np.mean(db_vals):.2f} dB ---")

if __name__ == "__main__":
    run_v4_blind_test()