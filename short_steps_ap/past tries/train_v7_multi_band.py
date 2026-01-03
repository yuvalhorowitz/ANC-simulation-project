import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Configuration & Hyperparameters ---
DATA_DIR = "driver_bulk_4spk_data"
TRAIN_SCENARIOS = range(45) 
FS = 8000
WINDOW_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 150

# --- 2. Custom Multi-Band Loss ---
@tf.keras.utils.register_keras_serializable()
class MultiBandExpertLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        self.bins = fs / window_size
        # Your specific band request with escalating penalties for high-freq errors
        self.bands = [
            (50, 350, 10.0),   # Band 1: Engine Core
            (350, 800, 25.0),  # Band 2: Mid Harmonics
            (800, 1200, 50.0), # Band 3: High Precision
            (1200, 1600, 80.0),# Band 4: Critical Harmonic Zone
            (1600, 2000, 120.0)# Band 5: Absolute Stability Zone
        ]

    def call(self, y_true, y_pred):
        error_fft = tf.abs(tf.signal.rfft(y_true[:, :, 0] - y_pred[:, :, 0]))
        total_loss = 0.0
        for low, high, weight in self.bands:
            l_idx, h_idx = int(low/self.bins), int(high/self.bins)
            total_loss += tf.reduce_mean(tf.square(error_fft[:, l_idx:h_idx])) * weight
        return total_loss

# --- 3. 5-Branch Parallel Architecture ---

def build_v7_multi_band_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    
    def tcn_branch(inp, filters, kernel, dilations, activ):
        x = inp
        for d in dilations:
            x = tf.keras.layers.Conv1D(filters, kernel, dilation_rate=d, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Conv1D(1, 1, activation=activ)(x)

    # Branching: Each path is physically isolated to prevent "harmonic leaking"
    b1 = tcn_branch(inputs, 16, 9, [1, 2, 4, 8, 16, 32, 64, 128], 'tanh')
    b2 = tcn_branch(inputs, 16, 7, [1, 2, 4, 8, 16, 32, 64], 'tanh')
    b3 = tcn_branch(inputs, 12, 5, [1, 2, 4, 8, 16, 32], 'linear')
    b4 = tcn_branch(inputs, 12, 3, [1, 2, 4, 8, 16], 'linear')
    b5 = tcn_branch(inputs, 12, 3, [1, 2, 4, 8], 'linear')

    merged = tf.keras.layers.Add()([b1, b2, b3, b4, b5])
    outputs = tf.keras.layers.Activation('tanh')(merged)
    return tf.keras.Model(inputs, outputs)

# --- 4. Data Loading Logic ---
def load_data(scenario_ids):
    X, y = [], []
    print(f"Loading {len(scenario_ids)} scenarios...")
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
    model = build_v7_multi_band_model()
    
    # Lower Learning Rate for expert fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(4e-4), loss=MultiBandExpertLoss(FS, WINDOW_SIZE))
    
    print(f"--- Training v7: {len(X_train)//BATCH_SIZE} steps per epoch ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save('anc_v7_multi_band_expert.keras')

    # --- 5. Output Band Statistics ---
    eval_len = 8000
    u_pred = model.predict(X_train[:eval_len])
    rec, gen = y_train[:eval_len, -1, 0], u_pred[:, -1, 0]
    res = rec + gen
    
    print("\n" + "="*30)
    print("v7 MULTI-BAND PERFORMANCE DATA")
    print("="*30)
    
    bands = [(50, 350), (350, 800), (800, 1200), (1200, 1600), (1600, 2000)]
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    
    for l, h in bands:
        mask = (f >= l) & (f <= h)
        db = 10 * np.log10(np.sum(p_orig[mask]) / np.sum(p_resid[mask]))
        status = "PASSED (Cancelling)" if db > 0 else "FAILED (Adding Harmonics)"
        print(f"Band {l:4d}-{h:4d} Hz: {db:6.2f} dB Reduction | {status}")
    
    # --- 6. Plotting the 5 Zones ---
    
    plt.figure(figsize=(15, 8))
    plt.plot(f, 10*np.log10(p_orig+1e-12), label="Original Noise", alpha=0.4)
    plt.plot(f, 10*np.log10(p_resid+1e-12), label="Residual (v7)", color='green', lw=2)
    
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    for (l, h), c in zip(bands, colors):
        plt.axvspan(l, h, color=c, alpha=0.1)
        
    plt.xlim(0, 2200); plt.title("v7 Expert Multi-Band PSD Analysis"); plt.legend(); plt.grid(True)
    plt.savefig("v7_training_report.png")
    print("="*30)