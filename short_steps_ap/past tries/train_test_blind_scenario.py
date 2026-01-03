import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- Configuration ---
DATA_DIR = "fast_data_8k"
TRAIN_SCENARIOS = [0, 1]  # Multi-scenario training
FS = 8000
WINDOW_SIZE = 512
KERNEL_SIZE = 5
FILTERS = 16
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def load_and_normalize(scn_id):
    """Load signals and apply the successful v3 normalization."""
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    
    # Normalization logic: Signal / max(abs(Signal))
    ref /= np.max(np.abs(ref))
    mic /= np.max(np.abs(mic))
    
    return ref, mic, hs

def prepare_multi_data(scenario_ids):
    """Stack multiple normalized scenarios for training."""
    all_X, all_y = [], []
    for scn_id in scenario_ids:
        ref, mic, hs = load_and_normalize(scn_id)
        # Pre-filter with RIR (System ID)
        filtered_ref = convolve(ref, hs, mode='same')
        
        for i in range(len(filtered_ref) - WINDOW_SIZE):
            all_X.append(filtered_ref[i:i+WINDOW_SIZE])
            all_y.append(mic[i+WINDOW_SIZE])
            
    return np.array(all_X).reshape(-1, WINDOW_SIZE, 1), np.array(all_y)

def build_stable_tcn():
    """Architecture from lessons learned: Simple Causal Stack."""
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, 1))
    x = inputs
    for d in [1, 2, 4, 8, 16, 32]:
        x = tf.keras.layers.Conv1D(FILTERS, kernel_size=KERNEL_SIZE, 
                                   dilation_rate=d, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    X_train, y_train = prepare_multi_data(TRAIN_SCENARIOS)
    model = build_stable_tcn()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0), loss='mse')
    
    print(f"Training on Scenarios {TRAIN_SCENARIOS}...")
    history = model.fit(X_train, -y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # Save the model for the testing script
    model.save('anc_tcn_v4_multi.keras')
    print("✓ Model saved as 'anc_tcn_v4_multi.keras'")