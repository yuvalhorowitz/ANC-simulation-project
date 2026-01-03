"""
Step 2: Localized TCN Training for Driver-Area ANC
This script trains the TCN model using the driver-focused localized data.
It targets the 20Hz-2000Hz range and provides a comprehensive graphical report.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "driver_localized_anc_data"
TRAIN_SCENARIOS = range(8) # Using first 8 scenarios for training
FS = 8000
WINDOW_SIZE = 512
KERNEL_SIZE = 7 
FILTERS = 16
EPOCHS = 60
BATCH_SIZE = 64

# --- 2. Enhanced Broadband Frequency-Weighted Loss ---
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        
        # Frequency bins weights
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        
        # 1. Engine Core (20Hz - 400Hz): Highest priority
        engine_low = int(20 / (fs / window_size))
        engine_high = int(400 / (fs / window_size))
        weights[engine_low:engine_high] = 15.0 
        
        # 2. Road & High Harmonics (400Hz - 2000Hz): Secondary priority
        road_high = int(2000 / (fs / window_size))
        weights[engine_high:road_high] = 3.0
        
        # 3. Guard Band (>2000Hz): Low weight to keep the output smooth
        weights[road_high:] = 0.1
        
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        # FFT of the error signal
        error_fft = tf.signal.rfft(error[:, :, 0])
        # Weighted MSE in Frequency Domain
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Data Loading & Linear-Tanh Normalization ---
def load_and_normalize_localized(scn_id):
    prefix = os.path.join(DATA_DIR, f"driver_scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    
    # Scale to 0.8 to preserve Tanh linearity
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 
    return ref, mic, hs

def prepare_data(scenario_ids):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        ref, mic, hs = load_and_normalize_localized(scn_id)
        # Pre-filter reference with secondary path (Filtered-X logic)
        filtered_ref = convolve(ref, hs, mode='same')
        for i in range(0, len(filtered_ref) - WINDOW_SIZE, 64): # Strided for memory
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. TCN Model Definition ---
def build_tcn_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    # Dilations covering 253 samples
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    # Tanh output to respect physical constraints [-1, 1]
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    X_train, y_train = prepare_data(TRAIN_SCENARIOS)
    model = build_tcn_model()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss=BroadbandTargetedLoss(FS, WINDOW_SIZE))
    
    print(f"--- Training Localized TCN on Scenarios {list(TRAIN_SCENARIOS)} ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save('anc_driver_localized_v1.keras')

    # --- 5. Visualization & Reporting ---
    u_pred = model.predict(X_train[:1000]) # Eval on a subset
    sig_recorded = y_train[:1000, -1, 0]   
    sig_generated = u_pred[:, -1, 0]  
    sig_combined = sig_recorded + sig_generated 
    reduction = 10 * np.log10(np.mean(sig_recorded**2) / np.mean(sig_combined**2))

    plt.figure(figsize=(15, 18))
    
    # Time Domain
    plt.subplot(4, 1, 1)
    plt.plot(sig_recorded[200:600], label="Noise at Ear", color='blue', alpha=0.5)
    plt.plot(sig_generated[200:600], label="Anti-Noise Output", color='orange', linestyle='--')
    plt.plot(sig_combined[200:600], label="Residual Error", color='green', linewidth=2)
    plt.title(f"Time Domain Analysis (Reduction: {reduction:.2f} dB)"); plt.legend(); plt.grid(True, alpha=0.3)

    # Frequency Domain (Full 2000Hz target check)
    plt.subplot(4, 1, 2)
    f, psd_orig = welch(sig_recorded, FS, nperseg=1024)
    _, psd_resid = welch(sig_combined, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original", color='blue', alpha=0.5)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Cancelled", color='green')
    plt.title("Frequency Domain (PSD)"); plt.xlim(0, 2200); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)

    # Residual Magnitude
    plt.subplot(4, 1, 3)
    plt.plot(np.abs(sig_combined), color='red', alpha=0.4, label="Error Magnitude")
    plt.title("Error Magnitude Over Time"); plt.legend(); plt.grid(True, alpha=0.3)

    # Training History
    plt.subplot(4, 1, 4)
    plt.plot(history.history['loss'], color='black', label="Weighted Loss")
    plt.title("Training Convergence Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig("localized_training_report.png"); plt.show()