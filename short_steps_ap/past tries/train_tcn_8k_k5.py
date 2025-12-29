import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# Optimized Settings
DATA_DIR = "fast_data_8k"
SCENARIO_ID = 0
WINDOW_SIZE = 256 # Matches RF and RIR length
FS = 8000
K = 5
EPOCHS = 60
BATCH_SIZE = 64

def load_data(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    return np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

def prepare_sequences_8k(ref, mic, hs_rir, window_size):
    # Pre-filter reference with room RIR (System ID logic)
    filtered_ref = convolve(ref, hs_rir, mode='same')
    X, y = [], []
    for i in range(len(filtered_ref) - window_size):
        X.append(filtered_ref[i:i+window_size])
        y.append(mic[i+window_size])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)

def build_optimized_tcn(window_size):
    """
    TCN with k=5 and 6 layers. 
    RF = 1 + (5-1)*(2^6 - 1) = 253 samples.
    """
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = inputs
    # Dilations from 2^0 to 2^5 
    for d in [1, 2, 4, 8, 16, 32 , 64]:
        x = tf.keras.layers.Conv1D(16, kernel_size=K, dilation_rate=d, 
                                   padding='causal', activation='relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def plot_results(y_orig, y_resid, history):
    plt.figure(figsize=(12, 10))
    # Time Domain
    plt.subplot(3, 1, 1)
    plt.plot(y_orig[1000:1400], label="Original (8kHz)", alpha=0.6)
    plt.plot(y_resid[1000:1400], label="Residual (ANC ON)", color='green')
    plt.title(f"Performance at 8kHz with k={K}")
    plt.legend(); plt.grid(True)
    
    # Frequency Domain (dB)
    plt.subplot(3, 1, 2)
    f, p_orig = welch(y_orig, FS, nperseg=1024)
    _, p_resid = welch(y_resid, FS, nperseg=1024)
    plt.plot(f, 10*np.log10(p_orig+1e-10), label="Original Noise")
    plt.plot(f, 10*np.log10(p_resid+1e-10), label="After ANC", color='green')
    plt.xlim(0, 500); plt.ylabel("dB/Hz"); plt.legend(); plt.grid(True)
    
    # Loss
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], color='red')
    plt.title("Learning Curve"); plt.grid(True)
    
    reduction = 10 * np.log10(np.mean(y_orig**2) / np.mean(y_resid**2))
    plt.suptitle(f"Total Reduction: {reduction:.2f} dB", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("anc_8k_k5_result.png")

if __name__ == "__main__":
    ref, mic, hs = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences_8k(ref, mic, hs, WINDOW_SIZE)
    
    model = build_optimized_tcn(WINDOW_SIZE)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    
    print("Training Optimized TCN...")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    u_pred = model.predict(X_train).flatten()
    residual = y_train + u_pred
    plot_results(y_train, residual, history)
    print(f"✓ Analysis saved to 'anc_8k_k5_result.png'")