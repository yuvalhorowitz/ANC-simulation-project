import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import config
import simulation_setup

def create_dataset():
    print(f"--- GENERATING BLOCK-BASED DATASET (Phase 4) ---")
    
    # 1. Ensure Simulation Environment Exists
    if not os.path.exists(config.PRIMARY_PATH_FILE):
        simulation_setup.generate_synthetic_paths()

    primary_path, _ = sf.read(config.PRIMARY_PATH_FILE)
    
    # 2. Load Raw Audio
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.wav')]
    if not raw_files:
        print(f"ERROR: No .wav files found in {config.RAW_DATA_DIR}")
        return

    X_windows = [] 
    d_blocks = [] # TARGETS ARE NOW BLOCKS

    for file in raw_files:
        path = os.path.join(config.RAW_DATA_DIR, file)
        audio, _ = sf.read(path)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # Physics: Ref Mic -> Primary Path -> Noise at Ear
        noise_at_ear = signal.fftconvolve(audio, primary_path, mode='same')

        # --- WINDOW LOOP WITH BLOCK TARGETS ---
        num_samples = len(audio)
        
        # Stop early to accommodate the Block size and Latency
        stop_at = num_samples - config.BLOCK_SIZE - config.LATENCY_SAMPLES
        
        for t in range(config.WINDOW_SIZE, stop_at, config.HOP_SIZE):
            
            # Input (X): History window from Reference Mic
            x_window = audio[t - config.WINDOW_SIZE : t]
            
            # Target (y): Future Block starting after hardware latency
            start = t + config.LATENCY_SAMPLES
            d_target_block = noise_at_ear[start : start + config.BLOCK_SIZE]

            X_windows.append(x_window.reshape(-1, 1))
            d_blocks.append(d_target_block.reshape(-1, 1))

    X_arr = np.array(X_windows, dtype=np.float32)
    d_arr = np.array(d_blocks, dtype=np.float32)

    print(f"Generated {len(X_arr)} samples.")
    print(f"X shape: {X_arr.shape} | y shape: {d_arr.shape}")

    # 3. Save to Disk
    # Note: Using 'dataset.npz' to match your current train.py
    save_path = os.path.join(config.PROCESSED_DATA_DIR, "dataset.npz")
    np.savez(save_path, X=X_arr, y=d_arr)

    print(f"--- SUCCESS: Dataset saved to {save_path} ---")

if __name__ == "__main__":
    create_dataset()