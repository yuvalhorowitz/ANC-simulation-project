"""
Step 3.1: Breakthrough Blind Test Evaluation
Tests the model on unseen localized driver data.
Targets: 20Hz-2000Hz with high-frequency artifact checking.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- Configuration ---
DATA_DIR = "driver_localized_anc_data"
TEST_SCENARIO = 9  # Unseen scenario for blind validation
MODEL_PATH = 'anc_breakthrough_model.keras'
FS = 8000
WINDOW_SIZE = 512

# --- Custom Loss for Loading ---
@tf.keras.utils.register_keras_serializable()
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        e_low, e_high = int(20/(fs/window_size)), int(400/(fs/window_size))
        weights[e_low:e_high] = 15.0 
        road_high = int(2000/(fs/window_size))
        weights[e_high:road_high] = 3.0
        weights[road_high:] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

def run_blind_test():
    print(f"--- Loading Breakthrough Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'BroadbandTargetedLoss': BroadbandTargetedLoss})

    # Data Loading and Normalization
    prefix = os.path.join(DATA_DIR, f"driver_scn_{TEST_SCENARIO:03d}")
    ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
    
    # 0.8 range normalization for Tanh linear safety
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 

    # Filtered-X logic
    filtered_ref = convolve(ref, hs, mode='same')
    X_test = [filtered_ref[i:i+WINDOW_SIZE] for i in range(len(filtered_ref) - WINDOW_SIZE)]
    X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)

    # Predict Anti-Noise
    u_pred = model.predict(X_test, batch_size=128)
    recorded, generated = mic[WINDOW_SIZE:], u_pred[:, -1, 0]
    residual = recorded + generated 
    reduction = 10 * np.log10(np.mean(recorded**2) / np.mean(residual**2))

    # --- Visualization ---
    plt.figure(figsize=(15, 14))
    
    # Time Domain Interaction
    plt.subplot(3, 1, 1)
    plt.plot(recorded[2000:2600], label="Original Noise", color='blue', alpha=0.5)
    plt.plot(generated[2000:2600], label="Anti-Noise (TCN)", color='orange', linestyle='--')
    plt.plot(residual[2000:2600], label="Residual", color='green', linewidth=2)
    plt.title(f"Blind Test: Time Domain (NR: {reduction:.2f} dB)"); plt.legend(); plt.grid(True)

    # Spectral Performance (0-2000Hz Target)
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(recorded, FS, nperseg=1024)
    _, psd_resid = welch(residual, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original", color='blue', alpha=0.5)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Cancelled", color='green')
    plt.title("Frequency Domain: PSD Analysis"); plt.xlim(0, 2500); plt.ylabel("dB/Hz"); plt.grid(True)

    # Error magnitude check
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(residual), color='red', alpha=0.4)
    plt.title("Instantaneous Error Magnitude"); plt.grid(True)

    plt.tight_layout(); plt.savefig("blind_test_breakthrough_report.png")
    print(f"--- Blind Test Result: {reduction:.2f} dB. Report Saved. ---")

if __name__ == "__main__":
    run_blind_test()