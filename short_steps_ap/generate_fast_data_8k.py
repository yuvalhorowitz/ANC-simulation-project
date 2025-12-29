import numpy as np
import pyroomacoustics as pra
import os

# New Settings for 8kHz
FS = 8000 
DURATION = 1.0 
DATA_DIR = "fast_data_8k"
BASE_ENGINE_FREQ = 50

def generate_engine_noise(duration, fs, base_freq):
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    for i in range(1, 5): # 4 Harmonics
        signal += (1.0/i) * np.sin(2 * np.pi * base_freq * i * t)
    signal += 0.05 * np.random.normal(0, 1, len(t))
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def create_scenario_8k(scn_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    room_dim = [4, 3, 2.5]
    room = pra.ShoeBox(room_dim, fs=FS, max_order=3, absorption=0.2)

    source_pos = [0.5, 1.5, 1.0]
    ref_mic_pos = [0.6, 1.5, 1.0]
    error_mic_pos = [3.0, 2.0, 1.2]
    speaker_pos = [2.8, 2.0, 1.2]

    engine_noise = generate_engine_noise(DURATION, FS, BASE_ENGINE_FREQ)
    room.add_source(source_pos, signal=engine_noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, room.fs))
    room.simulate()

    ref_signal = room.mic_array.signals[0]
    mic_signal = room.mic_array.signals[1]

    # Secondary Path (Hs) calculation
    room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=3, absorption=0.2)
    room_hs.add_source(speaker_pos, signal=np.array([1.0] + [0.0]*255))
    room_hs.add_microphone(error_mic_pos)
    room_hs.compute_rir()
    hs_rir = room_hs.rir[0][0][:256].astype(np.float32)

    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", ref_signal)
    np.save(f"{prefix}_mic.npy", mic_signal)
    np.save(f"{prefix}_hs.npy", hs_rir)

if __name__ == "__main__":
    for i in range(3):
        create_scenario_8k(i)
    print(f"✓ Data generated in {DATA_DIR} at 8kHz.")