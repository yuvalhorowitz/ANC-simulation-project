"""
Step 1.1: Refined Driver-Focused ANC Data Generation
This script generates training data for a localized zone around the driver's head.
It implements small spatial perturbations to help the TCN learn a robust local 
transfer function while staying within the 20Hz-2000Hz target range.
"""

import numpy as np
import pyroomacoustics as pra
import os
import random
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- Global Project Configuration ---
FS = 8000            # Sampling rate per Work Plan/Lessons Learned
DURATION = 2.0       # Signal length for training samples
DATA_DIR = "driver_localized_anc_data"
NUM_SCENARIOS = 10   

def plot_localized_map(room_dim, src, ref, err, spks, scn_id):
    """Saves a map of the driver's localized cancellation zone."""
    plt.figure(figsize=(10, 7))
    plt.plot([0, room_dim[0], room_dim[0], 0, 0], 
             [0, 0, room_dim[1], room_dim[1], 0], 'k-', linewidth=2)

    plt.scatter(src[0], src[1], marker='*', s=250, color='red', label='Engine')
    plt.scatter(ref[0], ref[1], marker='o', s=100, color='blue', label='Ref Mic')
    plt.scatter(err[0], err[1], marker='o', s=200, color='green', label='Error Mic (Ear)')
    
    spk_array = np.array(spks)
    plt.scatter(spk_array[:, 0], spk_array[:, 1], marker='s', s=120, color='orange', label='Speakers')

    plt.title(f"Localized Driver Map - Scenario {scn_id:03d}")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')

    png_path = os.path.join(DATA_DIR, f"driver_map_{scn_id:03d}.png")
    plt.savefig(png_path)
    plt.close()

def butter_lowpass(data, cutoff, fs, order=4):
    """Lowpass filter for road noise components (up to 2000Hz)."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def generate_targeted_noise(duration, fs):
    """
    Generates complex noise targeting the 20-2000Hz range:
    - Engine harmonics (dominant low-freq)
    - Road noise (broadband up to 2000Hz)
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Engine component (Harmonics based on 50Hz base)
    base_f = random.uniform(48, 52) 
    num_harmonics = 10 
    engine_sig = np.sum([(1.0/(i**0.9)) * np.sin(2*np.pi*base_f*i*t) for i in range(1, num_harmonics + 1)], axis=0)
    
    # Road component (Broadband 20-2000Hz)
    white_noise = np.random.normal(0, 1, len(t))
    road_sig = butter_lowpass(white_noise, 2000, fs) * 0.3
    
    combined = engine_sig + road_sig
    # Normalize to 0.8 for Tanh linearity
    max_val = np.max(np.abs(combined))
    if max_val > 0:
        combined = (combined / max_val) * 0.8
        
    return combined.astype(np.float32)

def create_localized_scenario(scn_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    room_dim = [4.5, 2.5, 1.5] # Standard car cabin per Work Plan
    abs_coeff = 0.35 # Consistent cabin absorption
    room = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)

    engine_pos = [0.5, 1.25, 0.7]
    ref_mic_pos = [0.7, 1.25, 0.8]
    
    # Localized head position (X=2.4, Y=0.75) with 5cm perturbations
    # This addresses the user's request for "small changes" in ear position
    error_mic_pos = [
        2.4 + random.uniform(-0.05, 0.05),
        0.75 + random.uniform(-0.05, 0.05),
        1.1 + random.uniform(-0.02, 0.02)
    ]
    
    # Using front-door speakers primarily for driver-focused cancellation
    car_speakers = [[1.2, 0.2, 0.5], [1.2, 2.3, 0.5]]

    plot_localized_map(room_dim, engine_pos, ref_mic_pos, error_mic_pos, car_speakers, scn_id)

    noise = generate_targeted_noise(DURATION, FS)
    room.add_source(engine_pos, signal=noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, FS))

    # Calculate Secondary Path (Hs) RIR (512 samples for reverb tails)
    hs_rirs = []
    for spk in car_speakers:
        room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)
        room_hs.add_source(spk, signal=np.array([1.0] + [0.0]*511))
        room_hs.add_microphone(error_mic_pos)
        room_hs.compute_rir()
        hs_rirs.append(room_hs.rir[0][0][:512]) 
    
    avg_hs_rir = np.mean(hs_rirs, axis=0).astype(np.float32)

    room.simulate()
    prefix = os.path.join(DATA_DIR, f"driver_scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", room.mic_array.signals[0]) 
    np.save(f"{prefix}_mic.npy", room.mic_array.signals[1]) 
    np.save(f"{prefix}_hs.npy", avg_hs_rir)

if __name__ == "__main__":
    print(f"Generating {NUM_SCENARIOS} localized driver scenarios...")
    for i in range(NUM_SCENARIOS):
        create_localized_scenario(i)
    print(f"Data and localized maps saved to '{DATA_DIR}'.")