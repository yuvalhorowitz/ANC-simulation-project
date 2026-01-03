import numpy as np
import pyroomacoustics as pra
import os
import random
from scipy.signal import convolve, butter, lfilter

FS, DURATION, DATA_DIR, NUM_SCENARIOS = 8000, 2.0, "driver_bulk_4spk_data", 50

def generate_super_noise(duration, fs):
    t = np.linspace(0, duration, int(fs * duration))
    base_f = random.uniform(40, 70)
    
    # 1. 45 Harmonics (Full Spectrum up to 3kHz)
    # Using 0.4 decay to keep high frequencies VERY strong
    engine = np.sum([(1.0/(i**0.4)) * np.sin(2*np.pi*base_f*i*t + random.uniform(0, 2*np.pi)) 
                     for i in range(1, 46)], axis=0)
    
    # 2. High-Pass Spectral Fill (800-2000Hz)
    # This ensures the model CANNOT ignore the "precision zone"
    white = np.random.normal(0, 0.2, len(t))
    b, a = butter(4, [800/(fs/2), 2200/(fs/2)], btype='band')
    spectral_fill = lfilter(b, a, white)
    
    combined = engine + spectral_fill
    return (combined / (np.max(np.abs(combined)) + 1e-7)) * 0.8

def create_v9_scenario(scn_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    room_dim = [4.5, 2.5, 1.5]
    room = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=random.uniform(0.2, 0.4))

    # Positions (Same localized head-movement logic)
    engine_pos = [0.5, 1.25, 0.7]
    ref_mic_pos = [0.7, 1.25, 0.8]
    error_mic_pos = [2.4 + random.uniform(-0.05, 0.05), 
                      0.75 + random.uniform(-0.05, 0.05), 
                      1.1 + random.uniform(-0.02, 0.02)]
    
    noise = generate_super_noise(DURATION, FS)
    room.add_source(engine_pos, signal=noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, FS))

    # Calculate Hs for 4 speakers
    car_speakers = [[1.2, 0.2, 0.5], [1.2, 2.3, 0.5], [3.2, 0.2, 0.5], [3.2, 2.3, 0.5]]
    hs_rirs = []
    for spk in car_speakers:
        room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=0.3)
        room_hs.add_source(spk, signal=np.array([1.0] + [0.0]*511))
        room_hs.add_microphone(error_mic_pos)
        room_hs.compute_rir()
        hs_rirs.append(room_hs.rir[0][0][:512])
    
    room.simulate()

    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", room.mic_array.signals[0])
    np.save(f"{prefix}_mic.npy", room.mic_array.signals[1])
    np.save(f"{prefix}_hs.npy", np.mean(hs_rirs, axis=0).astype(np.float32))

if __name__ == "__main__":
    print("Generating v9 Super-Harmonic Data...")
    for i in range(NUM_SCENARIOS):
        create_v9_scenario(i)
        if (i+1)%10 == 0: print(f"Progress: {i+1}/50")
    print("Success. New training data ready.")