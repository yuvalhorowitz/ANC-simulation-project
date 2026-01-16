import numpy as np
import scipy.signal as signal
import soundfile as sf
import os
import config

def generate_synthetic_paths():
    print(f"Generating ANECHOIC (Clean) Paths in {config.PROCESSED_DATA_DIR}...")
    
    if not os.path.exists(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)

    # --- 1. Secondary Path S(z) (Speaker -> Ear) ---
    # SIMPLIFIED: Pure delay + Speaker Bandpass. No reflections.
    ir_len = 64
    s_z = np.zeros(ir_len)
    
    # Hardware + Distance Delay (4 samples = ~17cm)
    delay = config.LATENCY_SAMPLES 
    
    # Main Pulse (The Speaker playing the sound)
    s_z[delay] = 1.0 
    # s_z[delay+1] = 0.5  <-- REMOVED (No resonance/smearing)
    # s_z[delay+2] = 0.2  <-- REMOVED
    
    # Apply mild Low-Pass (Speaker physics)
    # Speakers naturally can't play infinite high freq
    nyquist = config.SAMPLE_RATE / 2
    b, a = signal.butter(4, 3000 / nyquist, 'low')
    s_z_filtered = signal.lfilter(b, a, s_z)
    
    # Normalize
    s_z_filtered = s_z_filtered / np.max(np.abs(s_z_filtered))
    
    sf.write(config.SECONDARY_PATH_FILE, s_z_filtered, config.SAMPLE_RATE)
    print(f"  [OK] Secondary Path: Clean/Anechoic")

    # --- 2. Primary Path P(z) (Engine -> Ear) ---
    # SIMPLIFIED: Direct path only. No wall bounces.
    p_z = np.zeros(ir_len)
    
    # Sound travel delay (8 samples = ~34cm)
    path_delay = 8 
    
    # Direct Sound
    p_z[path_delay] = 1.0
    
    # REFLECTIONS REMOVED:
    # p_z[15] = 0.15 (Window reflection) -> DELETED
    # p_z[20] = 0.05 (Dashboard reflection) -> DELETED
    
    # Normalize
    p_z = p_z / np.max(np.abs(p_z))
    
    sf.write(config.PRIMARY_PATH_FILE, p_z, config.SAMPLE_RATE)
    print(f"  [OK] Primary Path: Clean/Anechoic")

if __name__ == "__main__":
    generate_synthetic_paths()