import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Configuration ---
DATA_DIR = "car_anc_final_data"
TRAIN_SCENARIOS = [0, 1]
TEST_SCENARIO = 2
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

# --- 3. Data Processing with 0.8 Scaling & White Noise ---
def load_and_normalize(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 # Keep tanh in linear region
    return ref, mic, hs

def prepare_robust_data(scenario_ids, is_training=True):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        ref, mic, hs = load_and_normalize(scn_id)
        filtered_ref = convolve(ref, hs, mode='same')
        
        # ADDING WHITE NOISE AUGMENTATION (ONLY FOR TRAINING)
        if is_training:
            noise_level = 0.02 # 2% white noise
            filtered_ref += noise_level * np.random.normal(0, 1, len(filtered_ref))
        
        for i in range(len(filtered_ref) - WINDOW_SIZE):
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. TCN Model ---
def build_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    reg = tf.keras.regularizers.l2(1e-4)
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu', kernel_regularizer=reg)(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    # Load Training Data (with noise)
    X_train, y_train = prepare_robust_data(TRAIN_SCENARIOS, is_training=True)
    
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss=HarmonicTargetedLoss(FS, WINDOW_SIZE))
    
    print("--- Starting Final Robust Training (Noise Augmentation + k=7) ---")
    # THE FIX: Assigning to 'history' variable
    history = model.fit(X_train, -y_train, epochs=60, batch_size=64, verbose=1)
    
    # Evaluation on Scenario 2
    X_test, y_test_full = prepare_robust_data([TEST_SCENARIO], is_training=False)
    u_pred_full = model.predict(X_test)
    
    sig_recorded = y_test_full[:, -1, 0]
    sig_generated = u_pred_full[:, -1, 0]
    sig_combined = sig_recorded + sig_generated
    
    reduction = 10 * np.log10(np.mean(sig_recorded**2) / np.mean(sig_combined**2))
    print(f"\n✓ Blind Test Result: {reduction:.2f} dB")

    # --- Plotting Results ---
    plt.figure(figsize=(15, 16))
    plt.subplot(4, 1, 1)
    plt.plot(sig_recorded[2000:2600], label="Noise", color='blue', alpha=0.5)
    plt.plot(sig_generated[2000:2600], label="Anti-Noise", color='orange', linestyle='--')
    plt.plot(sig_combined[2000:2600], label="Residual", color='green', linewidth=2)
    plt.title(f"Time Domain (NR: {reduction:.2f} dB)"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 2)
    f, psd_orig = welch(sig_recorded, FS, nperseg=1024)
    _, psd_resid = welch(sig_combined, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original", color='blue')
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual", color='green')
    plt.title("PSD (Frequency Domain)"); plt.xlim(0, 500); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(np.abs(sig_combined), color='red', alpha=0.3, label="Error Amplitude")
    plt.title("Residual Amplitude Over Time"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(history.history['loss'], color='black', label="Training Loss")
    plt.title("Loss Progression"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig("final_robust_results.png"); plt.show()