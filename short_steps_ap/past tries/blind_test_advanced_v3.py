"""
Aggregate Blind Test for Advanced v3 Model
Evaluates Scenarios 45-49 (Unseen Data).
Checks for harmonic suppression and spatial robustness.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TEST_SCENARIOS = range(45, 50) 
MODEL_PATH = 'anc_advanced_v3_no_harmonics.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Custom Loss Class (Required to load the .keras file) ---
@tf.keras.utils.register_keras_serializable()
class StrictBroadbandLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        idx_low = int(20 / (fs / window_size))
        idx_engine_high = int(400 / (fs / window_size))
        weights[idx_low:idx_engine_high] = 15.0 
        idx_road_high = int(2000 / (fs / window_size))
        weights[idx_engine_high:idx_road_high] = 25.0 
        weights[idx_road_high:] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Blind Test Execution ---
def run_blind_test():
    print(f"--- Loading Advanced Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'StrictBroadbandLoss': StrictBroadbandLoss})

    scenario_reductions = []
    plot_data = []

    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
        mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
        hs = np.load(f"{prefix}_hs.npy").astype(np.float32)

        # Pre-process identical to training
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Batch prediction
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
        
        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        # Calculate result at the error microphone
        recorded = mic[WINDOW_SIZE:]
        anti_noise = u_pred[:, -1, 0]
        residual = recorded + anti_noise
        
        reduction = 10 * np.log10(np.mean(recorded**2) / np.mean(residual**2))
        scenario_reductions.append(reduction)
        plot_data.append((recorded, residual, reduction, scn_id))
        print(f"Scenario {scn_id}: {reduction:.2f} dB Reduction")

    # --- 4. Visualization ---
    # Select the median scenario for visualization
    mid_idx = np.argsort(scenario_reductions)[len(scenario_reductions)//2]
    rec, res, red, sid = plot_data[mid_idx]

    plt.figure(figsize=(15, 14))

    # Panel 1: Time Domain Zoom (Destructive Interference Check)
    
    plt.subplot(3, 1, 1)
    plt.plot(rec[2000:2600], label="Original Noise", alpha=0.5, color='blue')
    plt.plot(res[2000:2600], label="Residual (After ANC)", color='green', lw=2)
    plt.title(f"Time Domain Interaction (Scenario {sid}) - Reduction: {red:.2f} dB")
    plt.legend(); plt.grid(True)

    # Panel 2: PSD Suppression Check
    
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    plt.subplot(3, 1, 2)
    plt.plot(f, 10*np.log10(p_orig+1e-12), label="Original Spectrum", color='blue', alpha=0.5)
    plt.plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled Spectrum", color='green')
    plt.title("Frequency Domain: Harmonic Suppression Check (0-2000Hz)")
    plt.xlim(0, 2200); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True)

    # Panel 3: Aggregate Stability (Bar Chart)
    plt.subplot(3, 1, 3)
    bars = plt.bar([f"Scn {i}" for i in TEST_SCENARIOS], scenario_reductions, color='skyblue')
    plt.axhline(y=np.mean(scenario_reductions), color='red', linestyle='--', label=f"Avg: {np.mean(scenario_reductions):.2f} dB")
    plt.title("Spatial Robustness: Reduction Across Unseen Head Positions")
    plt.ylabel("dB Reduction"); plt.legend()

    plt.tight_layout()
    plt.savefig("blind_test_advanced_v3_report.png")
    print(f"\n--- AVERAGE NOISE REDUCTION: {np.mean(scenario_reductions):.2f} dB ---")

if __name__ == "__main__":
    run_blind_test()