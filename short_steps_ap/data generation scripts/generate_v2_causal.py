import numpy as np
import pyroomacoustics as pra
import os
import random
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, correlate, correlation_lags

# --- Config ---
FS = 8000
DURATION = 2.0
DATA_DIR = "driver_bulk_4spk_data_v3"
NUM_SCENARIOS = 50

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def generate_guaranteed_scenario(scn_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    # 1. Physics Setup (Car Cabin)
    room_dim = [4.5, 2.5, 1.5]
    abs_coeff = np.random.uniform(0.15, 0.35)
    room = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)
    
    # 2. GUARANTEED CAUSAL GEOMETRY
    engine_pos = [0.5, 1.25, 0.7]
    # Ref Mic: Very close to engine (0.1m away)
    ref_mic_pos = [0.6, 1.25, 0.7] 
    # Ear (Error Mic): Rear Seat (Far away)
    error_mic_pos = [
        3.5 + np.random.uniform(-0.1, 0.1), 
        1.25 + np.random.uniform(-0.1, 0.1), 
        1.1 + np.random.uniform(-0.05, 0.05)
    ]
    
    # 4 Speakers 
    car_speakers = [
        [1.2, 0.2, 0.5], # Front Right
        [1.2, 2.3, 0.5], # Front Left
        [3.2, 0.2, 0.5], # Rear Right
        [3.2, 2.3, 0.5]  # Rear Left
    ]

    # 3. Signal Generation (Band-limited Gaussian)
    noise = np.random.normal(0, 1, int(FS * DURATION))
    b, a = butter_bandpass(50, 500, FS, order=4)
    noise = lfilter(b, a, noise)
    noise = noise / np.max(np.abs(noise)) * 0.8

    # 4. Simulation
    room.add_source(engine_pos, signal=noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, FS))
    
    # Secondary Path (Hs)
    hs_rirs = []
    hs_len = 512
    for spk in car_speakers:
        room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)
        room_hs.add_source(spk, signal=np.array([1.0] + [0.0]*1024)) # Long impulse
        room_hs.add_microphone(error_mic_pos)
        room_hs.compute_rir()
        
        # --- FIX: Padding/Slicing to Exact Length ---
        rir = room_hs.rir[0][0]
        if len(rir) >= hs_len:
            rir = rir[:hs_len]
        else:
            # Pad with zeros if too short
            rir = np.pad(rir, (0, hs_len - len(rir)))
            
        hs_rirs.append(rir)
    
    # Now valid because all arrays are (512,)
    hs_combined = np.mean(hs_rirs, axis=0)

    room.simulate()
    ref_signal = room.mic_array.signals[0]
    mic_signal = room.mic_array.signals[1]

    # 5. CAUSALITY CHECK
    corr = correlate(mic_signal, ref_signal, mode='full')
    lags = correlation_lags(mic_signal.size, ref_signal.size, mode='full')
    primary_delay = lags[np.argmax(np.abs(corr))]
    
    hs_energy = np.cumsum(hs_combined**2) / np.sum(hs_combined**2)
    secondary_delay = np.argmax(hs_energy > 0.01)
    
    budget = primary_delay - secondary_delay
    
    print(f"Scn {scn_id}: Pri={primary_delay}, Sec={secondary_delay}, Budget={budget}", end=" ")
    
    if budget < 5:
        print(" -> RETRY (Too tight)")
        return False 
    
    print(" -> KEEP")
    
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_ref.npy", ref_signal)
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_mic.npy", mic_signal)
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_hs.npy", hs_combined.astype(np.float32))
    return True

if __name__ == "__main__":
    print(f"Generating Guaranteed Causal Data in '{DATA_DIR}'...")
    count = 0
    while count < NUM_SCENARIOS:
        if generate_guaranteed_scenario(count):
            count += 1
    print("Done.")