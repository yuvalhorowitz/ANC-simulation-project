import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch # Added welch for dB analysis

# Settings for pre-filtered iteration
DATA_DIR = "fast_data_reverb"
SCENARIO_ID = 0
WINDOW_SIZE = 200 
EPOCHS = 50
BATCH_SIZE = 64

def load_data(scn_id):
    """Load signals with float32 precision."""
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs_rir = np.load(f"{prefix}_hs.npy").astype(np.float32)
    return ref, mic, hs_rir

def prepare_sequences_rir(ref, mic, hs_rir, window_size):
    """Pre-filter the input Reference with the room's RIR (Hs)."""
    filtered_ref = convolve(ref, hs_rir, mode='same')
    X, y = [], []
    for i in range(len(filtered_ref) - window_size):
        X.append(filtered_ref[i:i+window_size])
        y.append(mic[i+window_size])
    return np.array(X, dtype=np.float32).reshape(-1, window_size, 1), np.array(y, dtype=np.float32)

def build_compact_tcn(window_size):
    """Lightweight TCN for ANC."""
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- NEW FUNCTION START ---
def plot_anc_performance(y_orig, y_resid, history, fs=16000):
    """
    Plots a comprehensive comparison including:
    1. Time Domain (Amplitude)
    2. Frequency Domain (Power in dB)
    3. Learning Curve (MSE)
    """
    plt.figure(figsize=(12, 12))

    # 1. Time Domain Comparison (Zoomed)
    plt.subplot(3, 1, 1)
    start, end = 2000, 2500 
    plt.plot(y_orig[start:end], label="Original Noise (Mic)", alpha=0.6, color='blue')
    plt.plot(y_resid[start:end], label="Residual Noise (After ANC)", color='green', linewidth=1.5)
    plt.title("Time Domain: Signal at Driver's Ear")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Frequency Domain Comparison (PSD in dB)
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(y_orig, fs, nperseg=1024)
    _, psd_resid = welch(y_resid, fs, nperseg=1024)
    
    # Convert to dB
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Noise Power", color='blue', alpha=0.6)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Noise Power", color='green')
    
    plt.title("Frequency Domain: Intensity (dB)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("dB/Hz")
    plt.xlim(0, 500) # Engine harmonics range
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Learning Curve
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], color='red')
    plt.title("Learning Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # Global Reduction Metric
    reduction = 10 * np.log10(np.mean(y_orig**2) / np.mean(y_resid**2))
    plt.suptitle(f"ANC Analysis - Total Reduction: {reduction:.2f} dB", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("anc_performance_db.png")
    print(f"\n✓ Analysis saved to 'anc_performance_db.png'")
    print(f"✓ Measured Noise Reduction: {reduction:.2f} dB")
# --- NEW FUNCTION END ---

if __name__ == "__main__":
    print("--- Starting RIR Pre-Filtered Training ---")
    
    # 1. Load and Condition Data
    ref_sig, mic_sig, hs_rir = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences_rir(ref_sig, mic_sig, hs_rir, WINDOW_SIZE)
    
    # 2. Setup
    model = build_compact_tcn(WINDOW_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # 3. Train
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    # 4. Evaluate
    u_pred = model.predict(X_train).flatten()
    residual = y_train + u_pred 
    
    # 5. Call the new visualization function
    plot_anc_performance(y_train, residual, history)