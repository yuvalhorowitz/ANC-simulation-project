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

# --- Loss Function for 20Hz - 2000Hz ---
@tf.keras.utils.register_keras_serializable()
class BroadbandTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        # Weighting: Engine (15x), Road (5x), Above 2000Hz (0.1x)
        weights[int(20/(fs/window_size)):int(400/(fs/window_size))] = 15.0 
        weights[int(400/(fs/window_size)):int(2000/(fs/window_size))] = 5.0
        weights[int(2000/(fs/window_size)):] = 0.1
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        return tf.reduce_mean(tf.square(tf.abs(error_fft)) * self.weights)

# --- Data Preparation ---
def load_bulk_data(scenario_ids):
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 # Tanh linear safety
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            all_X.append(f_ref[j:j+WINDOW_SIZE]); all_y.append(mic[j:j+WINDOW_SIZE])
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), np.array(all_y).reshape(-1, WINDOW_SIZE, 1)

# --- Deep TCN Architecture ---
def build_deep_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    for d in [1, 2, 4, 8, 16, 32, 64]: # Receptive field ~509 samples
        x = tf.keras.layers.Conv1D(16, 7, dilation_rate=d, padding='causal', activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(1, 1, activation='tanh')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    X_train, y_train = load_bulk_data(TRAIN_SCENARIOS)
    model = build_deep_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0), loss=BroadbandTargetedLoss(FS, WINDOW_SIZE))
    
    # Scheduler to force fine-tuning
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=8, min_lr=1e-6)

    print(f"--- Starting Session: {len(X_train)//BATCH_SIZE} steps/epoch ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_cb], verbose=1)
    model.save('anc_bulk_driver_deep_model.keras')

    # --- FULL VISUAL OVERVIEW GENERATION ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    rec, gen = y_train[:eval_len, -1, 0], u_pred[:, -1, 0]
    res = rec + gen
    reduction = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    axs[0].plot(rec[2000:2600], label="Input (Noise)", alpha=0.5); axs[0].plot(gen[2000:2600], label="Output (Anti-Noise)", linestyle='--'); axs[0].plot(res[2000:2600], label="Residual Error", lw=2)
    axs[0].set_title(f"Time Domain Overview (NR: {reduction:.2f} dB)"); axs[0].legend(); axs[0].grid(True)
    
    f, p_orig = welch(rec, FS, nperseg=1024); _, p_resid = welch(res, FS, nperseg=1024)
    axs[1].plot(f, 10*np.log10(p_orig+1e-12), label="Original"); axs[1].plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled")
    axs[1].set_title("Frequency Domain Overview (0-2000Hz)"); axs[1].set_xlim(0, 2200); axs[1].legend(); axs[1].grid(True)
    
    axs[2].plot(np.abs(res), color='red', alpha=0.4); axs[2].set_title("Error Magnitude Statistics"); axs[2].grid(True)
    axs[3].plot(history.history['loss'], color='black'); axs[3].set_title("Training Convergence Overview"); axs[3].grid(True)

    plt.tight_layout(); plt.savefig("bulk_training_deep_report.png")
    print(f"Overview saved. Final Reduction: {reduction:.2f} dB")