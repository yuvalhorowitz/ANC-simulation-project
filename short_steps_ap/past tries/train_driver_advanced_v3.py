import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Project Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TRAIN_SCENARIOS = range(45) 
FS = 8000
WINDOW_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 150 

# --- 2. Strict Broadband Loss Function ---
@tf.keras.utils.register_keras_serializable()
class StrictBroadbandLoss(tf.keras.losses.Loss):
    """
    Penalizes high-frequency noise amplification (Waterbed Effect).
    Targets 20Hz - 2000Hz with heavy weights on the 400-2000Hz band.
    """
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        
        # Frequency bins weighting
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        
        # Engine Core (20-400Hz): High Priority
        idx_low = int(20 / (fs / window_size))
        idx_engine_high = int(400 / (fs / window_size))
        weights[idx_low:idx_engine_high] = 15.0 
        
        # Harmonic Suppression (400-2000Hz): Extreme Penalty
        # This prevents the "Green Line" from rising above the "Blue Line"
        idx_road_high = int(2000 / (fs / window_size))
        weights[idx_engine_high:idx_road_high] = 25.0 
        
        # Guard Band (>2000Hz): Low weight for stability
        weights[idx_road_high:] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        # Fast Fourier Transform (RFFT)
        error_fft = tf.signal.rfft(error[:, :, 0])
        # Weighted Mean Squared Error in Frequency Domain
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Filtered-X Data Loading & Preparation ---
def load_and_preprocess_data(scenario_ids):
    all_X, all_y = [], []
    print(f"Loading and filtering {len(scenario_ids)} scenarios...")
    
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
        mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
        hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
        
        # Secondary Path Convolution (Filtered-X Reference)
        f_ref = convolve(ref, hs, mode='same')
        
        # Normalize to Tanh linear region (0.8) to prevent clipping
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7)
        mic *= 0.8 
        
        # Windowing with striding for dataset diversity
        for i in range(0, len(f_ref) - WINDOW_SIZE, 128):
            all_X.append(f_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
            
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. Deep TCN Architecture (Dilation 128 + L2) ---
def build_advanced_tcn():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    
    # Progressive dilation up to 128 for a ~125ms receptive field
    # Added L2 regularization to keep weights small and signal clean
    for d in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv1D(16, 7, dilation_rate=d, padding='causal', 
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    # Tanh output layer for physical signal constraints
    outputs = tf.keras.layers.Conv1D(1, 1, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    X_train, y_train = load_and_preprocess_data(TRAIN_SCENARIOS)
    model = build_advanced_tcn()

    # Low clipnorm for gradient stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=StrictBroadbandLoss(FS, WINDOW_SIZE))

    # Fine-tuning callback
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)

    print("--- Starting Deep Training: Harmonic Suppression Session ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, 
                         batch_size=BATCH_SIZE, callbacks=[lr_scheduler], verbose=1)
    
    model.save('anc_advanced_v3_no_harmonics.keras')

    # --- 6. Full Visual Report Generation ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    recorded = y_train[:eval_len, -1, 0]
    generated = u_pred[:, -1, 0]
    residual = recorded + generated
    
    reduction_db = 10 * np.log10(np.mean(recorded**2) / np.mean(residual**2))

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))

    # Plot 1: Time Domain Zoom
    axs[0].plot(recorded[2000:2600], label="Original Noise", alpha=0.5)
    axs[0].plot(generated[2000:2600], label="Anti-Noise (TCN)", linestyle='--')
    axs[0].plot(residual[2000:2600], label="Residual", lw=2, color='green')
    axs[0].set_title(f"Time Domain: Signal Interaction (Reduction: {reduction_db:.2f} dB)")
    axs[0].legend(); axs[0].grid(True)

    # Plot 2: Frequency Domain (PSD)
    
    freqs, psd_orig = welch(recorded, FS, nperseg=1024)
    _, psd_resid = welch(residual, FS, nperseg=1024)
    axs[1].plot(freqs, 10*np.log10(psd_orig+1e-12), label="Original Spectrum")
    axs[1].plot(freqs, 10*np.log10(psd_resid+1e-12), label="Cancelled Spectrum", color='green')
    axs[1].set_title("PSD Analysis: 20Hz - 2000Hz (Harmonic Suppression Check)")
    axs[1].set_xlim(0, 2200); axs[1].set_ylabel("dB/Hz"); axs[1].legend(); axs[1].grid(True)

    # Plot 3: Error Magnitude Statistics
    axs[2].plot(np.abs(residual), color='red', alpha=0.4)
    axs[2].set_title("Instantaneous Absolute Error Over Time")
    axs[2].grid(True)

    # Plot 4: Training Convergence Overview
    axs[3].plot(history.history['loss'], color='black')
    axs[3].set_title("Training Convergence (Loss Curve)")
    axs[3].set_xlabel("Epoch"); axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("advanced_v3_training_report_english.png")
    print(f"Final Reduction: {reduction_db:.2f} dB. Report saved as PNG.")