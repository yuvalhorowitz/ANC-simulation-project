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

# --- 1. Dual-Band Precision Loss ---
@tf.keras.utils.register_keras_serializable()
class DualBandPrecisionLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        idx_800 = int(800 / (fs / window_size))
        idx_2000 = int(2000 / (fs / window_size))
        
        weights[:idx_800] = 10.0      # Heavy engine drone
        weights[idx_800:idx_2000] = 60.0 # High penalty for HF harmonics
        weights[idx_2000:] = 0.01     # Filter out noise above target
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error_fft = tf.signal.rfft(y_true[:, :, 0] - y_pred[:, :, 0])
        return tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)

# --- 2. Parallel Dual-Band Architecture ---
def build_v6_dual_band_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    
    # Branch 1: Engine/Drone (50-800Hz) - Large Kernels for long waves
    a = inputs
    for d in [1, 2, 4, 8, 16, 32, 64, 128]:
        a = tf.keras.layers.Conv1D(24, 7, dilation_rate=d, padding='causal', activation='relu')(a)
        a = tf.keras.layers.BatchNormalization()(a)
    branch_a = tf.keras.layers.Conv1D(1, 1, activation='tanh')(a)
    
    # Branch 2: Road/Precision (800-2000Hz) - Small kernels for phase accuracy
    b = inputs
    for d in [1, 2, 4, 8, 16, 32]:
        b = tf.keras.layers.Conv1D(16, 3, dilation_rate=d, padding='causal', activation='relu')(b)
        b = tf.keras.layers.BatchNormalization()(b)
    # Linear activation here prevents the clipping that causes high-freq artifacts
    branch_b = tf.keras.layers.Conv1D(1, 1, activation='linear')(b)
    
    # Merge and final soft safety
    combined = tf.keras.layers.Add()([branch_a, branch_b])
    outputs = tf.keras.layers.Activation('tanh')(combined)
    
    return tf.keras.Model(inputs, outputs)

# --- 3. Data Loading ---
def load_data(scenario_ids):
    X, y = [], []
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            X.append(f_ref[j:j+WINDOW_SIZE]); y.append(mic[j:j+WINDOW_SIZE])
    return np.array(X).reshape(-1, WINDOW_SIZE, 1), np.array(y).reshape(-1, WINDOW_SIZE, 1)

if __name__ == "__main__":
    X_train, y_train = load_data(TRAIN_SCENARIOS)
    model = build_v6_dual_band_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=DualBandPrecisionLoss(FS, WINDOW_SIZE))
    
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
    
    print(f"--- Training Start: {len(X_train)//BATCH_SIZE} steps per epoch ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_cb], verbose=1)
    model.save('anc_v6_dual_band.keras')

    # --- 4. Final Data Plotting ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    rec, gen = y_train[:eval_len, -1, 0], u_pred[:, -1, 0]
    res = rec + gen
    
    # Calculate Band-Specific Reductions
    def get_reduction(sig, residual, f_range):
        f, p1 = welch(sig, FS, nperseg=1024)
        _, p2 = welch(residual, FS, nperseg=1024)
        mask = (f >= f_range[0]) & (f <= f_range[1])
        return 10 * np.log10(np.sum(p1[mask]) / np.sum(p2[mask]))

    red_low = get_reduction(rec, res, [50, 800])
    red_high = get_reduction(rec, res, [800, 2000])

    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    
    # Time Interaction
    axs[0].plot(rec[2000:2600], label="Noise", alpha=0.5)
    axs[0].plot(res[2000:2600], label="Residual", color='green', lw=2)
    axs[0].set_title(f"v6 Dual-Band Time Interaction (Low Red: {red_low:.1f}dB, High Red: {red_high:.1f}dB)")
    axs[0].legend(); axs[0].grid(True)

    # Distinct Band Plot (PSD)
    
    f, p_orig = welch(rec, FS, nperseg=1024); _, p_resid = welch(res, FS, nperseg=1024)
    axs[1].plot(f, 10*np.log10(p_orig+1e-12), label="Original", alpha=0.4)
    axs[1].plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled", color='green')
    axs[1].axvspan(50, 800, color='blue', alpha=0.1, label='Drone Band')
    axs[1].axvspan(800, 2000, color='red', alpha=0.1, label='Precision Band')
    axs[1].set_title("PSD: Dual-Band Suppression Analysis"); axs[1].set_xlim(0, 2200); axs[1].legend()

    # Absolute Error Plot
    axs[2].plot(np.abs(res), color='red', alpha=0.3); axs[2].set_title("Instantaneous Error Statistics"); axs[2].grid(True)

    plt.tight_layout(); plt.savefig("v6_dual_band_report.png")
    print("Training Report Generated Successfully.")