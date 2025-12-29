import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "car_anc_final_data"
TRAIN_SCENARIOS = [0, 1]
# Removed TEST_SCENARIO as requested
FS = 8000
WINDOW_SIZE = 512
KERNEL_SIZE = 7 
FILTERS = 16
EPOCHS = 60
BATCH_SIZE = 64

# --- 2. Targeted Frequency-Weighted Loss ---
class HarmonicTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        low_bin = int(100 / (fs / window_size))
        high_bin = int(300 / (fs / window_size))
        weights[low_bin:high_bin] = 10.0 
        weights[int(350 / (fs / window_size)):] = 0.05
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Data Processing (0.8 Scaling) ---
def load_and_normalize_fixed(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 
    return ref, mic, hs

def prepare_full_data(scenario_ids):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        ref, mic, hs = load_and_normalize_fixed(scn_id)
        filtered_ref = convolve(ref, hs, mode='same')
        # Using stride 1 for exhaustive learning coverage
        for i in range(len(filtered_ref) - WINDOW_SIZE):
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. TCN with Weight Regularization ---
def build_clean_tcn():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    reg = tf.keras.regularizers.l2(1e-4)
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu',
                                   kernel_regularizer=reg)(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    # Load Training Data
    X_train, y_train = prepare_full_data(TRAIN_SCENARIOS)
    model = build_clean_tcn()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss=HarmonicTargetedLoss(FS, WINDOW_SIZE))
    
    print(f"--- Training on Scenarios {TRAIN_SCENARIOS} (Exhaustive) ---")
    # FIX: Assigning to 'history' variable correctly
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # --- 5. Evaluation and Plotting (Training Performance) ---
    print("\n--- Visualizing Performance on Training Data (Seen Scenarios) ---")
    # We use a slice of the training data to see how well it learned
    u_pred_train = model.predict(X_train)
    
    # Process signals: taking the last sample of each window
    sig_recorded = y_train[:, -1, 0]   
    sig_generated = u_pred_train[:, -1, 0]  
    sig_combined = sig_recorded + sig_generated 
    
    reduction = 10 * np.log10(np.mean(sig_recorded**2) / np.mean(sig_combined**2))
    print(f"✓ Training Noise Reduction: {reduction:.2f} dB")

    # Creating the Figure
    plt.figure(figsize=(15, 18))

    # Plot 1: Time Domain - Seeing the 3 signals interaction
    plt.subplot(4, 1, 1)
    view_range = range(2000, 2600) 
    plt.plot(sig_recorded[view_range], label="1. Recorded Noise (Target)", color='blue', alpha=0.5)
    plt.plot(sig_generated[view_range], label="2. Generated Anti-Noise (Output)", color='orange', linestyle='--')
    plt.plot(sig_combined[view_range], label="3. Residual (Result)", color='green', linewidth=2)
    plt.title(f"Time Domain Analysis - Training Data (NR: {reduction:.2f} dB)")
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    

    # Plot 2: Frequency Domain - Harmonic check
    plt.subplot(4, 1, 2)
    f, psd_orig = welch(sig_recorded, FS, nperseg=1024)
    _, psd_resid = welch(sig_combined, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.5)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Spectrum", color='green')
    plt.title("Frequency Domain: PSD (Target: No harmonics above 350Hz)"); plt.xlim(0, 600)
    plt.xlabel("Frequency [Hz]"); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)
    

    # Plot 3: Residual Amplitude (How much noise is left)
    plt.subplot(4, 1, 3)
    plt.plot(np.abs(sig_combined), color='red', alpha=0.3, label="Error Magnitude")
    plt.title("Residual Noise Amplitude Over Time (Training Data)"); plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 4: Training Progress - FIXED history variable usage
    plt.subplot(4, 1, 4)
    plt.plot(history.history['loss'], color='black', label="Training Loss")
    plt.title("Improvement During Training (Loss Curve)"); plt.xlabel("Epoch"); plt.ylabel("Weighted MSE"); plt.legend(); plt.grid(True, alpha=0.3)
    

    plt.tight_layout()
    plt.savefig("anc_training_improvement_results.png")
    plt.show()
    print("✓ Analysis image saved as 'anc_training_improvement_results.png'")