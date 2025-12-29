import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Configuration ---
DATA_DIR = "fast_data_8k"
SCENARIO_ID = 0
FS = 8000
WINDOW_SIZE = 512
KERNEL_SIZE = 5
FILTERS = 16
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0001 

# --- 2. Temporal Weighted Loss ---
def temporal_weighted_loss(y_true, y_pred):
    """Loss function optimized for 8kHz phase accuracy."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    abs_error = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.8 * mse + 0.2 * abs_error

# --- 3. Data Loading with Normalization ---
def load_and_normalize_data(scn_id, target_max=0.8):
    """
    Loads data and scales it so the maximum amplitude is target_max.
    This prevents 'Clipping' in the tanh output layer.
    """
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy")
    mic = np.load(f"{prefix}_mic.npy")
    hs = np.load(f"{prefix}_hs.npy")
    
    # Calculate Global Scale Factor based on the Mic signal
    # We want the highest peak in the Mic to be exactly target_max (0.8)
    scale_factor = np.max(np.abs(mic)) / target_max
    
    # Apply normalization to both signals to maintain relative energy
    ref_norm = ref / scale_factor
    mic_norm = mic / scale_factor
    
    print(f"✓ Data Normalized. Global Scale Factor: {scale_factor:.2f}")
    return ref_norm, mic_norm, hs

def prepare_sequences_8k_fixed(ref, mic, hs_rir, window_size):
    """Strictly causal alignment."""
    filtered_ref = convolve(ref, hs_rir, mode='full')[:len(ref)] 
    X, y = [], []
    for i in range(len(filtered_ref) - window_size - 1):
        X.append(filtered_ref[i:i+window_size])
        y.append(mic[i+window_size]) 
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

# --- 4. TCN Model ---
def build_stable_tcn(window_size):
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    # Output limited to [-1, 1]. Normalization ensures we stay within this.
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- 5. Visualization with Normalization Context ---
def plot_anc_normalized_results(y_orig, u_anti, y_resid, history):
    plt.figure(figsize=(14, 14))
    reduction = 10 * np.log10(np.mean(y_orig**2) / np.mean(y_resid**2))

    # Plot 1: Time Domain (Should show no clipping now!)
    plt.subplot(3, 1, 1)
    start, end = 2000, 2400
    plt.plot(y_orig[start:end], label="Normalized Noise (Mic)", color='blue', alpha=0.4, linewidth=2)
    plt.plot(u_anti[start:end], label="Anti-Noise (Correction)", color='orange', linestyle='--')
    plt.plot(y_resid[start:end], label="Residual", color='green', linewidth=1.5)
    plt.title(f"Time Domain: Normalization Fixed (Reduction: {reduction:.2f} dB)")
    plt.ylim([-1.1, 1.1]) # Showing the Tanh limits
    plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 2: PSD
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(y_orig, FS, nperseg=1024)
    _, psd_resid = welch(y_resid, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original (Normalized)", color='blue', alpha=0.4)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual", color='green')
    plt.title("PSD Analysis: High-Frequency Waterbed Effect Suppression")
    plt.xlim(0, 600); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 3: Learning Curve
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], color='red')
    plt.title("Training Loss: Should drop below 0.2 after normalization")
    plt.xlabel("Epoch"); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anc_normalized_evaluation.png")
    print(f"\n✓ Analysis saved as 'anc_normalized_evaluation.png'")

if __name__ == "__main__":
    # 1. Load and Normalize
    ref, mic, hs = load_and_normalize_data(SCENARIO_ID, target_max=0.8)
    X_train, y_train = prepare_sequences_8k_fixed(ref, mic, hs, WINDOW_SIZE)
    
    # 2. Setup Model
    model = build_stable_tcn(WINDOW_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=temporal_weighted_loss)
    
    # 3. Train
    print("--- Starting Training (Normalized v3) ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 4. Predict
    u_anti = model.predict(X_train).flatten()
    residual = y_train + u_anti
    
    # 5. Plot
    plot_anc_normalized_results(y_train, u_anti, residual, history)