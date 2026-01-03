import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TRAIN_SCENARIOS = range(45) 
FS = 8000
WINDOW_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 150 

# --- 1. v4 Regulated Loss Function ---
@tf.keras.utils.register_keras_serializable()
class SpectralRegulatedLoss(tf.keras.losses.Loss):
    """
    Combines weighted cancellation with an output energy penalty 
    to prevent high-frequency harmonic generation.
    """
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        
        # Define spectral weights
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        idx_engine = int(400 / (fs / window_size))
        idx_road = int(2000 / (fs / window_size))
        
        # Priority Weights
        weights[:idx_engine] = 20.0      # Engine focus
        weights[idx_engine:idx_road] = 40.0  # EXTREME penalty for high-freq mismatch
        weights[idx_road:] = 0.05        # Ignore ultrasonic
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Cancellation Loss: Error in frequency domain
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        cancellation_loss = tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)
        
        # Energy Penalty: Prevents the TCN from guessing wildly in high frequencies
        # If the model can't cancel perfectly, it should stay quiet.
        pred_fft = tf.signal.rfft(y_pred[:, :, 0])
        energy_penalty = tf.reduce_mean(tf.square(tf.abs(pred_fft)) * self.weights) * 0.02
        
        return cancellation_loss + energy_penalty

# --- 2. Data Preparation ---
def load_bulk_data(scenario_ids):
    all_X, all_y = [], []
    print(f"Loading {len(scenario_ids)} scenarios for v4 training...")
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            all_X.append(f_ref[j:j+WINDOW_SIZE]); all_y.append(mic[j:j+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 3. v4 High-Stability TCN Architecture ---
def build_v4_tcn():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    # Increased filters and added BatchNormalization for spectral stability
    for d in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = tf.keras.layers.Conv1D(32, 7, dilation_rate=d, padding='causal', 
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Final smoothing layer: kernel size 3 acts as an anti-aliasing filter
    outputs = tf.keras.layers.Conv1D(1, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    X_train, y_train = load_bulk_data(TRAIN_SCENARIOS)
    model = build_v4_tcn()
    
    # Clipnorm at 0.5 to keep training extremely stable
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss=SpectralRegulatedLoss(FS, WINDOW_SIZE))
    
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)

    print(f"--- Starting v4 Deep Training ({len(X_train)//BATCH_SIZE} steps/epoch) ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        callbacks=[lr_cb], verbose=1)
    
    model.save('anc_v4_regulated_model.keras')

    # --- 4. v4 Full Visual Overview Generation ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    rec, gen = y_train[:eval_len, -1, 0], u_pred[:, -1, 0]
    res = rec + gen
    reduction = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    
    # Plot 1: Time Domain (Interaction)
    axs[0].plot(rec[2000:2600], label="Noise", alpha=0.5); axs[0].plot(gen[2000:2600], label="Anti-Noise", linestyle='--')
    axs[0].plot(res[2000:2600], label="Residual", lw=2, color='green')
    axs[0].set_title(f"v4 Time Domain: Harmonic Suppression Check (NR: {reduction:.2f} dB)"); axs[0].legend(); axs[0].grid(True)
    
    # Plot 2: Frequency Domain (The critical PSD test)
    
    f, p_orig = welch(rec, FS, nperseg=1024); _, p_resid = welch(res, FS, nperseg=1024)
    axs[1].plot(f, 10*np.log10(p_orig+1e-12), label="Original"); axs[1].plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled", color='green')
    axs[1].set_title("v4 PSD: Energy Regulated Spectrum (0-2000Hz)"); axs[1].set_xlim(0, 2200); axs[1].legend(); axs[1].grid(True)
    
    # Plot 3: Error Statistics
    axs[2].plot(np.abs(res), color='red', alpha=0.3); axs[2].set_title("Absolute Error Magnitude"); axs[2].grid(True)
    
    # Plot 4: Convergence
    axs[3].plot(history.history['loss'], color='black'); axs[3].set_title("v4 Regulated Convergence Curve"); axs[3].grid(True)

    plt.tight_layout(); plt.savefig("v4_training_deep_report.png")
    print(f"v4 Training Complete. Final Reduction: {reduction:.2f} dB")