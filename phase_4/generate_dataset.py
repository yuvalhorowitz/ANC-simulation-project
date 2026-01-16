import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import librosa
import config
import simulation_setup

def extract_selective_features(audio_window, sr):
    """
    Extracts key features used for Selective Noise Cancellation.
    Based on research from Sule et al. (2023).
    """
    # 1. Pitch Extraction (using PiPTrack)
    pitches, magnitudes = librosa.piptrack(y=audio_window, sr=sr)
    # Select the pitch with the highest magnitude
    pitch_idx = magnitudes.argmax()
    pitch_val = pitches.flatten()[pitch_idx] if np.any(pitches) else 0.0
    
    # 2. Amplitude Envelope (RMS)
    # Measures sound wave strength over time [cite: 121]
    envelope = np.sqrt(np.mean(audio_window**2))
    
    return [pitch_val, envelope]

def create_dataset():
    print(f"--- GENERATING SELECTIVE MULTI-INPUT DATASET ---")
    
    if not os.path.exists(config.PRIMARY_PATH_FILE):
        simulation_setup.generate_synthetic_paths()

    primary_path, _ = sf.read(config.PRIMARY_PATH_FILE)
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('.wav')]
    
    if not raw_files:
        print(f"ERROR: No .wav files found in {config.RAW_DATA_DIR}")
        return

    X_audio = []    # Input 1: Raw Reference Mic Audio
    X_features = [] # Input 2: Selective Features (Pitch, Envelope)
    y_blocks = []   # Target: Future Noise Block for SI-SNR Loss

    for file in raw_files:
        path = os.path.join(config.RAW_DATA_DIR, file)
        audio, _ = sf.read(path)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # Physics: Ref Mic -> Primary Path -> Noise at Ear
        noise_at_ear = signal.fftconvolve(audio, primary_path, mode='same')

        num_samples = len(audio)
        stop_at = num_samples - config.BLOCK_SIZE - config.LATENCY_SAMPLES
        
        for t in range(config.WINDOW_SIZE, stop_at, config.HOP_SIZE):
            # A. Audio Window
            x_window = audio[t - config.WINDOW_SIZE : t]
            
            # B. Extract Research-Based Features [cite: 134, 135]
            # These help the model distinguish noise from desired audio [cite: 118]
            feats = extract_selective_features(x_window, config.SAMPLE_RATE)
            
            # C. Target Block (Starting after hardware latency)
            start = t + config.LATENCY_SAMPLES
            d_target_block = noise_at_ear[start : start + config.BLOCK_SIZE]

            X_audio.append(x_window.reshape(-1, 1))
            X_features.append(feats)
            y_blocks.append(d_target_block.reshape(-1, 1))

    # Convert to Numpy Arrays
    X_audio_arr = np.array(X_audio, dtype=np.float32)
    X_feats_arr = np.array(X_features, dtype=np.float32)
    y_arr = np.array(y_blocks, dtype=np.float32)

    print(f"Dataset Shape - Audio: {X_audio_arr.shape} | Features: {X_feats_arr.shape}")

    # Save to Disk for Multi-Input Training
    save_path = os.path.join(config.PROCESSED_DATA_DIR, "dataset.npz")
    np.savez(save_path, X_audio=X_audio_arr, X_features=X_feats_arr, y=y_arr)
    print(f"--- SUCCESS: Multi-input dataset saved to {save_path} ---")

if __name__ == "__main__":
    create_dataset()