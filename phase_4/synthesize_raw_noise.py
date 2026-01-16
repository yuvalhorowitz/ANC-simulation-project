import numpy as np
import soundfile as sf
import os
import config
import scipy.signal as signal

def generate_engine_noise(duration_sec, rpm, fs):
    """
    Simulates engine noise using a fundamental frequency + harmonics.
    4-cylinder engine at RPM: f = (RPM / 60) * 2
    """
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # Fundamental Frequency (firing rate)
    f0 = (rpm / 60) * 2  # ~26.6 Hz at 800 RPM
    
    # Mix harmonics (1st, 2nd, 3rd, 4th order) to sound like a motor
    audio = 0.6 * np.sin(2 * np.pi * f0 * t)             # Fundamental
    audio += 0.4 * np.sin(2 * np.pi * (2 * f0) * t)      # 1st Harmonic
    audio += 0.3 * np.sin(2 * np.pi * (3 * f0) * t)      # 2nd Harmonic
    audio += 0.1 * np.random.normal(0, 0.1, len(t))      # Mechanical random noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio

def generate_road_noise(duration_sec, fs):
    """
    Simulates road/wind noise using filtered Pink/White noise.
    """
    # White Noise
    num_samples = int(fs * duration_sec)
    white = np.random.normal(0, 1, num_samples)
    
    # Low-Pass Filter to simulate "rumble" inside cabin (Cutoff ~500Hz)
    nyquist = fs / 2
    b, a = signal.butter(4, 500 / nyquist, btype='low')
    road_rumble = signal.lfilter(b, a, white)
    
    # Normalize
    road_rumble = road_rumble / np.max(np.abs(road_rumble))
    return road_rumble

def main():
    print("--- SYNTHESIZING RAW NOISE SOURCES ---")
    
    # Ensure target directory exists
    if not os.path.exists(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR)
        print(f"Created directory: {config.RAW_DATA_DIR}")
        
    duration = 10  # Seconds
    sr = config.SAMPLE_RATE
    
    # 1. Generate Engine Idle (800 RPM)
    print("Generating Engine Noise (Idle)...")
    engine_wav = generate_engine_noise(duration, 800, sr)
    out_path = os.path.join(config.RAW_DATA_DIR, "engine_idle.wav")
    sf.write(out_path, engine_wav, sr)
    print(f"  [Saved] {out_path}")
    
    # 2. Generate Engine Rev (2000 RPM)
    print("Generating Engine Noise (Revving)...")
    engine_rev = generate_engine_noise(duration, 2000, sr)
    out_path = os.path.join(config.RAW_DATA_DIR, "engine_rev.wav")
    sf.write(out_path, engine_rev, sr)
    print(f"  [Saved] {out_path}")
    
    # 3. Generate Road Rumble
    print("Generating Road Noise...")
    road_wav = generate_road_noise(duration, sr)
    out_path = os.path.join(config.RAW_DATA_DIR, "road_noise.wav")
    sf.write(out_path, road_wav, sr)
    print(f"  [Saved] {out_path}")
    
    print("--- RAW SOURCES READY ---")
    print("Now run 'generate_dataset.py'!")

if __name__ == "__main__":
    main()