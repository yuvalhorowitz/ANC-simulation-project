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

# --- 2. Frequency-Weighted Loss ---
# Penalizes the 200-500Hz range where the "Waterbed Effect" usually occurs
class FrequencyWeightedMSE(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        
        # Focus weighting only where harmonics actually exist (up to 300Hz)
        low_bin = int(100 / (fs / window_size))
        high_bin = int(300 / (fs / window_size)) # Reduced from 500 to 300
        
        weights[low_bin:high_bin] = 8.0 
        
        # Zero out weights above 400Hz to tell the model: "Don't care about these frequencies"
        weights[int(400 / (fs / window_size)):] = 0.1
        
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred 
        error_fft = tf.signal.rfft(error[:, :, 0]) 
        error_mag_sq = tf.square(tf.abs(error_fft))
        weighted_error = error_mag_sq * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. Data Processing with Full Stride ---
def load_and_normalize(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    
    # Normalize to 1.0 first
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    
    # SCALE TO 0.8: This keeps the model in the linear region of tanh
    mic *= 0.8 
    
    return ref, mic, hs

def prepare_data_exhaustive(scenario_ids):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        ref, mic, hs = load_and_normalize(scn_id)
        filtered_ref = convolve(ref, hs, mode='same')
        # STRIDE = 1: Uses every possible sample for training
        for i in range(len(filtered_ref) - WINDOW_SIZE):
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i:i+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), \
           np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- 4. Seq2Seq TCN Architecture ---
def build_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, 
                                   dilation_rate=d, padding='causal', activation='relu')(x)
    # Outputting a full window of anti-noise
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    # A. Training
    X_train, y_train = prepare_data_exhaustive(TRAIN_SCENARIOS)
    print(f"Total training samples (Stride 1): {len(X_train)}")
    
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss=FrequencyWeightedMSE(FS, WINDOW_SIZE))
    
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save('anc_final_weighted_model.keras')

    # B. Inference on Blind Scenario
    X_test, y_test_full = prepare_data_exhaustive([TEST_SCENARIO])
    # Predicted Anti-Noise
    u_pred_full = model.predict(X_test)
    
    # We take the last point of each window to simulate real-time stream
    orig_noise = y_test_full[:, -1, 0]
    anti_noise = u_pred_full[:, -1, 0]
    residual = orig_noise + anti_noise
    
    reduction = 10 * np.log10(np.mean(orig_noise**2) / np.mean(residual**2))

    # C. Comprehensive Plotting
    plt.figure(figsize=(15, 14))

    # Plot 1: Time Domain (Signals)
    plt.subplot(4, 1, 1)
    start, end = 2000, 2600 # Focus on 600 samples for clarity
    plt.plot(orig_noise[start:end], label="1. Original Noise (At Ear)", color='blue', alpha=0.6)
    plt.plot(anti_noise[start:end], label="2. Generated Anti-Noise (Output)", color='orange', linestyle='--')
    plt.plot(residual[start:end], label="3. Residual (Combined Result)", color='green', linewidth=2)
    plt.title(f"Time Domain Analysis - Blind Test (NR: {reduction:.2f} dB)")
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)

    # Plot 2: Frequency Domain (PSD)
    plt.subplot(4, 1, 2)
    f, psd_orig = welch(orig_noise, FS, nperseg=1024)
    _, psd_resid = welch(residual, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue')
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Spectrum", color='green')
    plt.title("Power Spectral Density (PSD) - Generalization Verification"); plt.xlim(0, 600)
    plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 3: Error Over Time (Residual Magnitude)
    plt.subplot(4, 1, 3)
    plt.plot(np.abs(residual), color='red', alpha=0.4, label="Instantaneous Error Magnitude")
    plt.title("Residual Noise Absolute Amplitude Over Time"); plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 4: Training Progress (Loss)
    plt.subplot(4, 1, 4)
    plt.plot(history.history['loss'], color='black', label="Weighted MSE Loss")
    plt.title("Training Loss Curve"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anc_final_training_output.png")
    print(f"✓ Final analysis saved as 'anc_final_training_output.png'. Total NR: {reduction:.2f} dB")