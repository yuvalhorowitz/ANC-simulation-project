import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch
from tensorflow.keras import mixed_precision

# --- 1. GPU Diagnostic & Optimization ---
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed Precision is great for MX450 (2GB VRAM)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"Success! Found GPU: {gpus}")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("WARNING: No GPU detected by TensorFlow. Check your CUDA/cuDNN installation.")

# --- 2. Configuration ---
DATA_DIR = "driver_bulk_4spk_data"
TRAIN_SCENARIOS = range(45) 
FS = 8000
WINDOW_SIZE = 512
BATCH_SIZE = 32 
EPOCHS = 100

# --- 3. Surgical Sparsity Loss (Fixed Indexing) ---
@tf.keras.utils.register_keras_serializable()
class SurgicalSparsityLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size

    def call(self, y_true, y_pred):
        # Ensure data is Float32 and 2D for FFT stability
        # Squeezing ensures [Batch, 512, 1] becomes [Batch, 512]
        y_true = tf.squeeze(tf.cast(y_true, tf.float32), axis=-1)
        y_pred = tf.squeeze(tf.cast(y_pred, tf.float32), axis=-1)
        
        # Time-domain base loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Spectral Comparison
        true_fft = tf.abs(tf.signal.rfft(y_true))
        pred_fft = tf.abs(tf.signal.rfft(y_pred))
        
        # PHANTOM NOISE PENALTY: Suppresses output where input is silent
        silence_mask = tf.cast(true_fft < 1e-3, tf.float32)
        phantom_penalty = tf.reduce_mean(tf.square(pred_fft) * silence_mask) * 500.0
        
        # High-Frequency Error Weighting (800Hz - 2000Hz)
        error_fft = tf.abs(tf.signal.rfft(y_true - y_pred))
        bins = self.fs / self.window_size
        idx_800 = int(800 / bins)
        hf_penalty = tf.reduce_mean(tf.square(error_fft[:, idx_800:])) * 50.0

        return mse_loss + phantom_penalty + hf_penalty

# --- 4. Gated Architecture ---

def build_v8_surgical_model():
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    
    # BRANCH 1: Engine/Low-Pass
    l = inputs
    for d in [1, 2, 4, 8, 16, 32, 64, 128]:
        l = tf.keras.layers.Conv1D(16, 7, dilation_rate=d, padding='causal', activation='relu')(l)
        l = tf.keras.layers.BatchNormalization()(l)
    out_low = tf.keras.layers.Conv1D(1, 1, activation='tanh', dtype='float32')(l)

    # BRANCH 2: Precision/High-Pass
    h = inputs
    for d in [1, 2, 4, 8, 16]:
        h = tf.keras.layers.Conv1D(12, 3, dilation_rate=d, padding='causal', activation='relu')(h)
        h = tf.keras.layers.BatchNormalization()(h)
    out_high = tf.keras.layers.Conv1D(1, 1, activation='linear', 
                                      activity_regularizer=tf.keras.regularizers.l2(1.0),
                                      dtype='float32')(h)

    combined = tf.keras.layers.Add(dtype='float32')([out_low, out_high])
    return tf.keras.Model(inputs, combined)

# --- 5. Data Loading Pipeline ---
def get_dataset(scenario_ids):
    X, y = [], []
    print(f"Loading {len(scenario_ids)} scenarios...")
    for scn_id in scenario_ids:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            X.append(f_ref[j:j+WINDOW_SIZE])
            y.append(mic[j:j+WINDOW_SIZE])
    
    # Ensure explicit 3D shape (N, 512, 1) before dataset creation
    X_np = np.expand_dims(np.array(X), -1)
    y_np = np.expand_dims(np.array(y), -1)
    
    ds = tf.data.Dataset.from_tensor_slices((X_np, -y_np))
    ds = ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    train_ds = get_dataset(TRAIN_SCENARIOS)
    model = build_v8_surgical_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=SurgicalSparsityLoss(FS, WINDOW_SIZE))
    
    print(f"--- Training Start ---")
    history = model.fit(train_ds, epochs=EPOCHS, verbose=1)
    model.save('anc_v8_surgical_gpu.keras')

    # --- 6. Detailed Plotting and Analysis ---
    print("\n--- Generating Analysis ---")
    # Grab one batch for visualization
    for X_batch, y_batch in train_ds.take(1):
        u_pred = model.predict(X_batch)
        # Select first sample in batch
        rec_noise = -y_batch.numpy()[0, :, 0]
        anti_noise = u_pred[0, :, 0]
        residual = rec_noise + anti_noise
        break

    f, p_orig = welch(rec_noise, FS, nperseg=256)
    _, p_resid = welch(residual, FS, nperseg=256)
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    
    # Plot 1: Time Domain Zoom
    axs[0].plot(rec_noise[:800], label="Original Noise", alpha=0.5)
    axs[0].plot(residual[:800], label="Residual (After v8)", color='green', lw=2)
    axs[0].set_title("Time Domain: Anti-Noise Interaction")
    axs[0].legend(); axs[0].grid(True)

    # Plot 2: Frequency Domain Whistle Check
    
    axs[1].plot(f, 10*np.log10(p_orig+1e-12), label="Original Spectrum", alpha=0.4)
    axs[1].plot(f, 10*np.log10(p_resid+1e-12), label="Cancelled Spectrum", color='green', lw=2)
    axs[1].axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Silence Zone')
    axs[1].set_title("PSD Analysis: 0-2000Hz (Harmonic Suppression Verification)")
    axs[1].set_xlim(0, 2200); axs[1].legend(); axs[1].grid(True)

    # Plot 3: Convergence
    axs[2].plot(history.history['loss'], color='black')
    axs[2].set_title("Training Loss Convergence")
    axs[2].set_xlabel("Epoch"); axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("v8_gpu_detailed_report.png")
    
    # Final Table Output
    mask_hf = (f >= 1600) & (f <= 2000)
    db_hf = 10 * np.log10(np.sum(p_orig[mask_hf]) / np.sum(p_resid[mask_hf]))
    print("\n" + "="*30)
    print(f"HF Band (1.6k-2kHz) Improvement: {db_hf:.2f} dB")
    print("Goal: Near 0dB (No whistling) or positive (Reduction)")
    print("="*30)