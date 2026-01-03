"""
Step 3: Blind Test for Localized Driver ANC
This script evaluates the trained model on a 'Blind Scenario' (data not seen during training).
It measures the actual Noise Reduction (dB) and checks for high-frequency artifacts.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Configuration ---
DATA_DIR = "driver_localized_anc_data"
TEST_SCENARIO = 9  # A scenario NOT used in the TRAIN_SCENARIOS range
MODEL_PATH = 'anc_driver_localized_v1.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Custom Loss Class (Required for Loading) ---
@tf.keras.utils.register_keras_serializable()
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        engine_low, engine_high = int(20/(fs/window_size)), int(400/(fs/window_size))
        weights[engine_low:engine_high] = 15.0 
        road_high = int(2000/(fs/window_size))
        weights[engine_high:road_high] = 3.0
        weights[road_high:] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Loading and Inference ---
def run_blind_test():
    # Load model with custom loss object
    print(f"--- Loading Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'BroadbandTargetedLoss': BroadbandTargetedLoss})

    # Load test data
    prefix = os.path.join(DATA_DIR, f"driver_scn_{TEST_SCENARIO:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)

    # Normalize identical to training (Linear Tanh range)
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 

    # Prepare Filtered-X input
    filtered_ref = convolve(ref, hs, mode='same')
    X_test = []
    for i in range(len(filtered_ref) - WINDOW_SIZE):
        X_test.append(filtered_ref[i:i+WINDOW_SIZE])
    X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)

    # Run Prediction
    print("--- Running Inference on Blind Data ---")
    u_pred = model.predict(X_test, batch_size=128)
    
    # Extract results (Real-time simulation style)
    sig_recorded = mic[WINDOW_SIZE:]   
    sig_generated = u_pred[:, -1, 0]    
    sig_combined = sig_recorded + sig_generated 

    # --- 4. Performance Metrics and Visualization ---
    reduction = 10 * np.log10(np.mean(sig_recorded**2) / np.mean(sig_combined**2))
    print(f"\nFINAL BLIND TEST RESULT: {reduction:.2f} dB")

    plt.figure(figsize=(15, 12))

    # Time Domain Zoom
    plt.subplot(3, 1, 1)
    plt.plot(sig_recorded[1000:1500], label="Original Noise", color='blue', alpha=0.5)
    plt.plot(sig_generated[1000:1500], label="Anti-Noise (TCN)", color='orange', linestyle='--')
    plt.plot(sig_combined[1000:1500], label="Residual", color='green', linewidth=2)
    plt.title(f"Time Domain: Blind Test (Scenario {TEST_SCENARIO}) - NR: {reduction:.2f} dB")
    plt.legend(); plt.grid(True, alpha=0.3)

    # Frequency Domain (PSD) - Verification of 2000Hz target
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(sig_recorded, FS, nperseg=1024)
    _, psd_resid = welch(sig_combined, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.5)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual (After ANC)", color='green')
    plt.title("Frequency Domain: PSD Analysis (0-2000Hz Target)"); plt.xlim(0, 2500)
    plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)

    # Error magnitude
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(sig_combined), color='red', alpha=0.4)
    plt.title("Instantaneous Error Magnitude"); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig("blind_test_localized_report.png"); plt.show()

if __name__ == "__main__":
    run_blind_test()