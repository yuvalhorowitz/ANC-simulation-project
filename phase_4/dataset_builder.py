import numpy as np
import soundfile as sf
import os
import scipy.signal as signal
import config

class AudioDataset:
    def __init__(self):
        # Load the simulated paths
        self.primary_ir, _ = sf.read(config.PRIMARY_PATH_FILE)
        self.secondary_ir, _ = sf.read(config.SECONDARY_PATH_FILE)
    
    def load_and_process(self):
        """
        1. Loads raw WAVs from data/train
        2. Simulates 'Noise at Ear' (d_t) using Primary Path
        3. Windows data for TCN input
        """
        if not os.path.exists(config.RAW_DATA_DIR):
            raise FileNotFoundError(f"Raw data directory not found: {config.RAW_DATA_DIR}")

        raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.wav')]
        if not raw_files:
            raise ValueError(f"No .wav files found in {config.RAW_DATA_DIR}")

        X_buffer = [] # Input: Reference Mic History
        d_buffer = [] # Target: Noise at Error Mic (Current)
        
        print(f"Processing {len(raw_files)} files from {config.RAW_DATA_DIR}...")
        
        for file in raw_files:
            path = os.path.join(config.RAW_DATA_DIR, file)
            raw_noise, sr = sf.read(path)
            
            # Simple Resample Check (Basic)
            if sr != config.SAMPLE_RATE:
                # In production, use librosa.resample. For now, just skip or warn.
                print(f"  [WARN] Skipping {file}: Rate is {sr}, expected {config.SAMPLE_RATE}")
                continue
            
            # Simulate physics: Noise @ Ear = Raw Noise * Primary Path
            # We use fftconvolve for efficiency
            noise_at_ear = signal.fftconvolve(raw_noise, self.primary_ir, mode='same')
            
            # Create Sliding Windows
            # We need history [t-Window, t] to predict action at t
            num_samples = len(raw_noise)
            for i in range(config.WINDOW_SIZE, num_samples, config.HOP_SIZE):
                # Input: The "Reference" noise history
                window = raw_noise[i-config.WINDOW_SIZE : i]
                
                # Target: The noise happening RIGHT NOW at the ear
                target = noise_at_ear[i]
                
                X_buffer.append(window.reshape(-1, 1))
                d_buffer.append(target)
        
        X_arr = np.array(X_buffer, dtype=np.float32)
        d_arr = np.array(d_buffer, dtype=np.float32)
        
        print(f"Dataset Created. Shape: {X_arr.shape}")
        return X_arr, d_arr

    def get_secondary_path_tensor(self):
        """Returns S(z) formatted for TensorFlow Custom Loss"""
        # Shape: (Kernel_Size, In_Channels, Out_Channels) -> (64, 1, 1)
        return self.secondary_ir.reshape(-1, 1, 1).astype(np.float32)