"""
Step 2.2: Breakthrough Training (Targeting Loss < 24.8)
This script implements:
1. Increased Receptive Field (Dilation 64) for car cabin reverb.
2. Learning Rate Scheduler to bypass the 24.8 plateau.
3. Full graphical report including Time Domain, PSD, and Error metrics.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Project Configuration ---
DATA_DIR = "driver_localized_anc_data"
TRAIN_SCENARIOS = range(8) 
FS = 8000
WINDOW_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 100 # Increased for fine-tuning breakthrough

# --- 2. Custom Broadband Loss (20Hz-2000Hz) ---
@tf.keras.utils.register_keras_serializable()
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        
        # Priority: Engine Drone (20-400Hz)
        e_low, e_high = int(20/(fs/window_size)), int(400/(fs/window_size))
        weights[e_low:e_high] = 15.0 
        
        # Secondary: Road Noise (400-2000Hz)
        road_high = int(2000/(fs/window_size))
        weights[e_high:road_high] = 3.0
        
        # High Frequency Guard Band (>2000Hz)
        weights[road_high:] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Data Loading Logic ---
def prepare_training_data(scenario_ids):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"driver_scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        
        # Normalize to 0.8 to keep Tanh in linear region
        ref /= (np.max(np.abs(ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7)
        mic *= 0.8 
        
        # Filtered-X Pre-filtering
        filtered_ref = convolve(ref, hs, mode='same')
        
        for i in range(0, len(filtered_ref) - WINDOW_SIZE, 128):
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
            
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. TCN Breakthrough Architecture ---
def build_breakthrough_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    # Extended dilations up to 64 to capture 4.5m cabin reflections
    for d in [1, 2, 4, 8, 16, 32, 64]:
        x = tf.keras.layers.Conv1D(16, 7, dilation_rate=d, padding='causal', activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(1, 1, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    X_train, y_train = prepare_training_data(TRAIN_SCENARIOS)
    model = build_breakthrough_model()

    # Gradient Clipping (clipnorm=1.0) to stabilize convergence
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss=BroadbandTargetedLoss(FS, WINDOW_SIZE))

    # Scheduler to bypass plateaus
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5)

    print("--- Starting Breakthrough Training Loop ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_callback], verbose=1)
    model.save('anc_breakthrough_model.keras')

    # --- 5. Comprehensive Result Visualization ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    recorded = y_train[:eval_len, -1, 0]
    generated = u_pred[:, -1, 0]
    residual = recorded + generated
    reduction = 10 * np.log10(np.mean(recorded**2) / np.mean(residual**2))

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))

    # Panel 1: Time Domain Interaction
    axs[0].plot(recorded[1000:1600], label="Noise", color='blue', alpha=0.5)
    axs[0].plot(generated[1000:1600], label="Anti-Noise (TCN Output)", color='orange', linestyle='--')
    axs[0].plot(residual[1000:1600], label="Residual Error", color='green', linewidth=2)
    axs[0].set_title(f"Time Domain Analysis (Reduction: {reduction:.2f} dB)"); axs[0].legend(); axs[0].grid(True)

    # Panel 2: Spectral Analysis (20-2000Hz)
    f, p_orig = welch(recorded, FS, nperseg=1024)
    _, p_resid = welch(residual, FS, nperseg=1024)
    axs[1].plot(f, 10 * np.log10(p_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.5)
    axs[1].plot(f, 10 * np.log10(p_resid + 1e-12), label="Cancelled Spectrum", color='green')
    axs[1].set_title("Frequency Domain (PSD Analysis)"); axs[1].set_xlim(0, 2200); axs[1].legend(); axs[1].grid(True)

    # Panel 3: Error Statistics
    axs[2].plot(np.abs(residual), color='red', alpha=0.4, label="Instantaneous Error")
    axs[2].set_title(f"Error Magnitude (Final Mean: {np.mean(np.abs(residual)):.4f})"); axs[2].legend(); axs[2].grid(True)

    # Panel 4: Convergence Stat
    axs[3].plot(history.history['loss'], color='black', label="Training Loss")
    axs[3].set_title("Learning Progress (Loss vs Epoch)"); axs[3].set_xlabel("Epoch"); axs[3].legend(); axs[3].grid(True)

    plt.tight_layout(); plt.savefig("breakthrough_training_results.png")
    print(f"--- Process Complete. PNG Saved. Final Loss: {history.history['loss'][-1]:.4f} ---")