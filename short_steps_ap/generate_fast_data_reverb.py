import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os

# Configuration
FS = 16000
DURATION = 1.0 
DATA_DIR = "fast_data_reverb"
BASE_ENGINE_FREQ = 50

def generate_engine_noise(duration, fs, base_freq):
    """Synthesize engine noise with harmonics."""
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    for i in range(1, 5):
        signal += (1.0/i) * np.sin(2 * np.pi * base_freq * i * t)
    signal += 0.05 * np.random.normal(0, 1, len(t))
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def create_reverb_scenario(scn_id):
    """
    Simulate a room with reflections (Reverberation).
    max_order > 0 enables the Image Source Method for reflections.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Room Setup with reflections
    # absorption=0.2 simulates a mix of hard (glass) and soft (seats) surfaces
    room_dim = [4, 3, 2.5]
    room = pra.ShoeBox(room_dim, fs=FS, max_order=3, absorption=0.2)

    # Positions [x, y, z]
    source_pos = [0.5, 1.5, 1.0]
    ref_mic_pos = [0.6, 1.5, 1.0]
    error_mic_pos = [3.0, 2.0, 1.2]
    speaker_pos = [2.8, 2.0, 1.2]

    engine_noise = generate_engine_noise(DURATION, FS, BASE_ENGINE_FREQ)
    room.add_source(source_pos, signal=engine_noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, room.fs))

    # Compute RIR and simulate
    room.simulate()

    ref_signal = room.mic_array.signals[0]
    mic_signal = room.mic_array.signals[1]

    # Calculate Secondary Path (Hs) WITH reflections
    room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=3, absorption=0.2)
    room_hs.add_source(speaker_pos, signal=np.array([1.0] + [0.0]*511)) # Longer impulse to capture echoes
    room_hs.add_microphone(error_mic_pos)
    room_hs.compute_rir()
    
    # We take the first 512 samples of the RIR to capture the primary reflections
    hs_rir = room_hs.rir[0][0][:512].astype(np.float32)

    # Save
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", ref_signal)
    np.save(f"{prefix}_mic.npy", mic_signal)
    np.save(f"{prefix}_hs.npy", hs_rir)

    # Visualization: Compare Anechoic vs Reverb (Implicitly)
    plt.figure(figsize=(10, 4))
    plt.plot(hs_rir)
    plt.title(f"Secondary Path RIR (with Reflections) - Scenario {scn_id}")
    plt.xlabel("Samples")
    plt.ylabel("Response")
    plt.grid(True)
    plt.savefig(f"{prefix}_rir_preview.png")
    plt.close()

if __name__ == "__main__":
    print(f"Generating scenarios with reflections in '{DATA_DIR}'...")
    for i in range(3):
        create_reverb_scenario(i)
    print("Success. The RIR now contains echoes.")