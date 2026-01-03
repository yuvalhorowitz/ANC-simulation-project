"""
Aggregate Blind Test: Evaluating the Deep Bulk Model
Tests the model on all 5 unseen scenarios (45-49).
Provides a statistical summary and a visual report of the best/worst performance.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TEST_SCENARIOS = range(45, 50) # The 5 scenarios never seen by the model
MODEL_PATH = 'anc_bulk_driver_deep_model.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Custom Loss Class (Needed for Loading) ---
@tf.keras.utils.register_keras_serializable()
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        weights[int(20/(fs/window_size)):int(400/(fs/window_size))] = 15.0 
        weights[int(400/(fs/window_size)):int(2000/(fs/window_size))] = 5.0
        weights[int(2000/(fs/window_size)):] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        return tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)

def run_aggregate_test():
    print(f"--- Loading Deep Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'BroadbandTargetedLoss': BroadbandTargetedLoss})

    results_db = []
    all_residuals = []
    
    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
        mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
        hs = np.load(f"{prefix}_hs.npy").astype(np.float32)

        # Normalization identical to training
        ref /= (np.max(np.abs(ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Filtered-X logic
        f_ref = convolve(ref, hs, mode='same')
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)

        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        rec, gen = mic[WINDOW_SIZE:], u_pred[:, -1, 0]
        res = rec + gen
        
        reduction = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
        results_db.append(reduction)
        all_residuals.append((rec, res, reduction, scn_id))
        print(f"Scenario {scn_id}: {reduction:.2f} dB Reduction")

    # --- 3. Visualization of the Best Scenario ---
    # We pick the median scenario to show "typical" performance
    best_idx = np.argsort(results_db)[len(results_db)//2]
    rec, res, red, scn_id = all_residuals[best_idx]

    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(rec[2000:2500], label="Original Noise", alpha=0.5)
    plt.plot(res[2000:2500], label="Residual Error", color='green')
    plt.title(f"Typical Blind Performance (Scenario {scn_id}) - Reduction: {red:.2f} dB")
    plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 2)
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    plt.plot(f, 10*np.log10(p_orig+1e-12), label="Original")
    plt.plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled", color='green')
    plt.title("Spectral Blind Test (0-2000Hz Target)"); plt.xlim(0, 2200); plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.bar([f"Scn {i}" for i in TEST_SCENARIOS], results_db, color='skyblue')
    plt.axhline(y=np.mean(results_db), color='red', linestyle='--', label=f"Avg: {np.mean(results_db):.2f} dB")
    plt.title("Aggregate Reduction across all Unseen Scenarios"); plt.ylabel("dB Reduction"); plt.legend()

    plt.tight_layout(); plt.savefig("blind_test_aggregate_report.png")
    print(f"\n--- AVERAGE BLIND REDUCTION: {np.mean(results_db):.2f} dB ---")

if __name__ == "__main__":
    run_aggregate_test()