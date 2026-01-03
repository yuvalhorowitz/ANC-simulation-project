"""
Bulk Data Generation: 50 Driver-Focused Scenarios with 4 Speakers
This script simulates 4 door speakers to create a robust secondary path (Hs).
It also saves PNG maps for every scenario and a summary distribution map.
Target: 20Hz - 2000Hz.
"""

import numpy as np
import pyroomacoustics as pra
import os
import random
import matplotlib.pyplot as plt

# --- Project Configuration ---
FS = 8000            # Optimized sampling rate
DURATION = 2.0       # Duration for spectral coverage
DATA_DIR = "driver_bulk_4spk_data"
NUM_SCENARIOS = 50 

def plot_scenario_map(room_dim, engine, ref, error_pos, speakers, scn_id, save_path):
    """Visualizes the 4-speaker car cabin layout."""
    plt.figure(figsize=(10, 6))
    plt.plot([0, room_dim[0], room_dim[0], 0, 0], [0, 0, room_dim[1], room_dim[1], 0], 'k-', lw=2)
    plt.scatter(engine[0], engine[1], marker='*', s=200, color='red', label='Engine')
    plt.scatter(ref[0], ref[1], marker='o', s=100, color='blue', label='Ref Mic')
    plt.scatter(error_pos[0], error_pos[1], marker='o', s=150, color='green', label='Ear (Error Mic)')
    
    spks = np.array(speakers)
    plt.scatter(spks[:, 0], spks[:, 1], marker='s', s=100, color='orange', label='4 Door Speakers')

    plt.title(f"4-Speaker Layout: Scenario {scn_id:03d}")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.legend(loc='upper right')
    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()

def create_bulk_scenario(scn_id, head_positions):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    room_dim = [4.5, 2.5, 1.5]
    abs_coeff = random.uniform(0.2, 0.4) 
    room = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)

    # Positions
    engine_pos = [0.5, 1.25, 0.7]
    ref_mic_pos = [0.7, 1.25, 0.8]
    # Localized head perturbations (+/- 5cm)
    error_mic_pos = [2.4 + random.uniform(-0.05, 0.05), 
                     0.75 + random.uniform(-0.05, 0.05), 
                     1.1 + random.uniform(-0.02, 0.02)]
    
    head_positions.append(error_mic_pos)
    
    # 4-Speaker configuration: Front Doors and Rear Doors
    car_speakers = [[1.2, 0.2, 0.5], [1.2, 2.3, 0.5], [3.2, 0.2, 0.5], [3.2, 2.3, 0.5]]

    # Map Generation
    plot_scenario_map(room_dim, engine_pos, ref_mic_pos, error_mic_pos, car_speakers, scn_id, 
                      os.path.join(DATA_DIR, f"map_scn_{scn_id:03d}.png"))

    # Noise setup
    t = np.linspace(0, DURATION, int(FS * DURATION))
    base_f = random.uniform(40, 70) 
    noise = np.sum([(1.0/(i**0.8)) * np.sin(2*np.pi*base_f*i*t) for i in range(1, 15)], axis=0)
    noise = (noise / (np.max(np.abs(noise)) + 1e-7)) * 0.8 # Preserve tanh linearity

    room.add_source(engine_pos, signal=noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, FS))

    # Calculate Hs (Filtered-X Path)
    hs_rirs = []
    for spk in car_speakers:
        room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)
        room_hs.add_source(spk, signal=np.array([1.0] + [0.0]*511))
        room_hs.add_microphone(error_mic_pos)
        room_hs.compute_rir()
        hs_rirs.append(room_hs.rir[0][0][:512]) # 512-sample RIR
    
    room.simulate()

    # Save data
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_ref.npy", room.mic_array.signals[0])
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_mic.npy", room.mic_array.signals[1])
    np.save(f"{DATA_DIR}/scn_{scn_id:03d}_hs.npy", np.mean(hs_rirs, axis=0).astype(np.float32))

if __name__ == "__main__":
    head_positions = []
    print(f"Generating {NUM_SCENARIOS} 4-speaker scenarios...")
    for i in range(NUM_SCENARIOS):
        create_bulk_scenario(i, head_positions)
        if (i+1)%10 == 0: print(f"Progress: {i+1}/50")

    # Save summary cloud map
    plt.figure(figsize=(10, 6))
    plt.plot([0, 4.5, 4.5, 0, 0], [0, 0, 2.5, 2.5, 0], 'k-', lw=2)
    hp = np.array(head_positions)
    plt.scatter(hp[:, 0], hp[:, 1], c='green', s=30, alpha=0.5, label='Head Movement Area')
    plt.title("Summary: Driver head distribution for 50 scenarios")
    plt.legend(); plt.savefig(os.path.join(DATA_DIR, "position_summary.png"))
    print(f"Success! Data and maps saved in '{DATA_DIR}'.")