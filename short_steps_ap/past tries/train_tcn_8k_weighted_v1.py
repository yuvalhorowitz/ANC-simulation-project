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
LEARNING_RATE = 0.0001 # Stable learning rate

# --- 2. Loss & Data Prep ---
def temporal_weighted_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    abs_error = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.8 * mse + 0.2 * abs_error

def load_data(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    return np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

def prepare_sequences_8k_fixed(ref, mic, hs_rir, window_size):
    # Causal alignment to ensure no "future peeking"
    filtered_ref = convolve(ref, hs_rir, mode='full')[:len(ref)] 
    X, y = [], []
    for i in range(len(filtered_ref) - window_size - 1):
        X.append(filtered_ref[i:i+window_size])
        y.append(mic[i+window_size]) 
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

# --- 3. Model ---
def build_stable_tcn(window_size):
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs
    for d in [1, 2, 4, 8, 16, 32]: # 253 samples receptive field
        x = tf.keras.layers.Conv1D(FILTERS, KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- 4. Enhanced Visualization (The 3 Signals) ---
def plot_anc_complete_results(y_orig, u_anti, y_resid, history):
    """
    Plots Original, Correction (Anti-noise), and Combined signals.
    """
    plt.figure(figsize=(14, 14))
    reduction = 10 * np.log10(np.mean(y_orig**2) / np.mean(y_resid**2))

    # Plot 1: Time Domain Comparison (The 3 Signals)
    plt.subplot(3, 1, 1)
    start, end = 2000, 2400 # Focusing on 400 samples for clarity
    plt.plot(y_orig[start:end], label="Original Noise (Target)", color='blue', alpha=0.4, linewidth=2)
    plt.plot(u_anti[start:end], label="Model Anti-Noise (Correction)", color='orange', linestyle='--')
    plt.plot(y_resid[start:end], label="Combined Residual (Result)", color='green', linewidth=1.5)
    plt.title(f"Time Domain Analysis - Total Reduction: {reduction:.2f} dB")
    plt.xlabel("Sample Index"); plt.ylabel("Amplitude")
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)

    # Plot 2: Frequency Domain (PSD)
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(y_orig, FS, nperseg=1024)
    _, psd_resid = welch(y_resid, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.4)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Spectrum", color='green')
    plt.title("PSD: Effectiveness in 200Hz-500Hz Range")
    plt.xlim(0, 600); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True, alpha=0.3)

    # Plot 3: Learning Curve
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], color='red')
    plt.title("Training Loss Stagnation Check")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anc_complete_evaluation.png")
    print(f"\n✓ Complete analysis saved as 'anc_complete_evaluation.png'")

if __name__ == "__main__":
    ref, mic, hs = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences_8k_fixed(ref, mic, hs, WINDOW_SIZE)
    
    model = build_stable_tcn(WINDOW_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=temporal_weighted_loss)
    
    print("--- Starting Training (Full Visualization Mode) ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # Save the model in the native Keras format
    model.save('anc_tcn_model_8k.keras') 
    print("✓ Model saved as 'anc_tcn_model_8k.keras'")

    # Generate Predictions
    u_anti = model.predict(X_train).flatten() # This is the correction signal
    residual = y_train + u_anti # Combine original and correction
    
    plot_anc_complete_results(y_train, u_anti, residual, history)