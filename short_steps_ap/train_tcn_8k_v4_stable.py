import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- Optimized Configuration (Based on Lessons Learned) ---
DATA_DIR = "fast_data_8k"
SCENARIO_ID = 0
FS = 8000
WINDOW_SIZE = 512   
KERNEL_SIZE = 5     
FILTERS = 16        
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def load_data(scn_id):
    """Load acoustic signals from the local directory."""
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    return np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

def prepare_sequences_8k(ref, mic, hs_rir, window_size):
    """
    Pre-filter the reference signal with the room's Impulse Response (Hs).
    This process is based on the 'Filtered-x' logic to align phase.
    """
    filtered_ref = convolve(ref, hs_rir, mode='same')
    X, y = [], []
    for i in range(len(filtered_ref) - window_size):
        X.append(filtered_ref[i:i+window_size])
        y.append(mic[i+window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def build_stable_tcn(window_size):
    """
    Builds a TCN using Causal Convolutions [cite: 73, 91] and 
    Exponential Dilations[cite: 75, 103].
    """
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs
    
    # Dilation rates from 2^0 to 2^5 to ensure sufficient receptive field [cite: 146]
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, dilation_rate=d, 
                                   padding='causal', activation='relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def plot_anc_results(y_orig, y_resid, history):
    """
    Visualizes training effectiveness across three domains:
    1. Time Domain (Waveform Alignment)
    2. Frequency Domain (Noise Power in dB)
    3. Optimization Domain (MSE Loss Convergence)
    """
    plt.figure(figsize=(14, 12))

    # Calculate Total Reduction in dB
    # NR = 10 * log10(Power_Original / Power_Residual)
    reduction = 10 * np.log10(np.mean(y_orig**2) / np.mean(y_resid**2))

    # 1. Time Domain Comparison (Zoomed View)
    plt.subplot(3, 1, 1)
    plt.plot(y_orig[2000:2500], label="Original Noise (At Ear)", alpha=0.5, color='blue')
    plt.plot(y_resid[2000:2500], label="Residual Noise (After ANC)", color='green', linewidth=1.5)
    plt.title(f"Time Domain: Signal Cancellation (Total Reduction: {reduction:.2f} dB)")
    plt.xlabel("Sample Index"); plt.ylabel("Amplitude")
    plt.legend(); plt.grid(True, alpha=0.3)

    # 2. Frequency Domain Comparison (PSD in dB)
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(y_orig, FS, nperseg=1024)
    _, psd_resid = welch(y_resid, FS, nperseg=1024)
    
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.5)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Spectrum", color='green')
    plt.title("Frequency Domain: Power Spectral Density")
    plt.xlabel("Frequency [Hz]"); plt.ylabel("dB/Hz")
    plt.xlim(0, 500) # Focusing on engine harmonics (50Hz-500Hz)
    plt.legend(); plt.grid(True, alpha=0.3)

    # 3. Learning Curve (MSE Loss)
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], color='red', label="Training Loss")
    plt.title("Learning Curve: Optimization Progress")
    plt.xlabel("Epoch"); plt.ylabel("Mean Squared Error (MSE)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anc_training_evaluation.png")
    print(f"\n✓ Plots saved as 'anc_training_evaluation.png'")

if __name__ == "__main__":
    # 1. Prepare Data
    ref, mic, hs = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences_8k(ref, mic, hs, WINDOW_SIZE)
    
    # 2. Build and Compile
    model = build_stable_tcn(WINDOW_SIZE)
    # Using clipnorm=1.0 to stabilize gradients as suggested by the paper [cite: 232]
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    
    # 3. Train
    print("--- Starting Training ---")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 4. Predict and Evaluate
    u_pred = model.predict(X_train).flatten()
    residual = y_train + u_pred 
    
    # 5. Visualize
    plot_anc_results(y_train, residual, history)