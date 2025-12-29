import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os

# Configuration for fast iteration
FS = 16000
DURATION = 1.0  # 1 second is enough for sanity checks
DATA_DIR = "fast_data"
BASE_ENGINE_FREQ = 50

def generate_engine_noise(duration, fs, base_freq):
    """
    Synthesize engine-like noise using harmonics and low-level white noise.
    """
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    
    # Add 4 harmonics (Fundamental, 2x, 3x, 4x)
    for i in range(1, 5):
        amp = 1.0 / i 
        signal += amp * np.sin(2 * np.pi * base_freq * i * t)
    
    # Add low-level broadband noise (road/wind simulation)
    signal += 0.05 * np.random.normal(0, 1, len(t))
    
    # Normalize to prevent clipping
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def create_fast_scenario(scn_id):
    """
    Simulate a simplified anechoic room and save signals as .npy files.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Setup anechoic room (max_order=0 means no reflections)
    room_dim = [4, 3, 2.5]
    room = pra.ShoeBox(room_dim, fs=FS, max_order=3)

    # Coordinates: [x, y, z]
    source_pos = [0.5, 1.5, 1.0]      # Engine location
    ref_mic_pos = [0.6, 1.5, 1.0]     # Reference mic (near engine)
    error_mic_pos = [3.0, 2.0, 1.2]   # Driver's ear (Error mic)
    speaker_pos = [2.8, 2.0, 1.2]     # ANC speaker (near ear)

    # Generate synthetic input signal
    engine_noise = generate_engine_noise(DURATION, FS, BASE_ENGINE_FREQ)

    # Add source and microphones to room
    room.add_source(source_pos, signal=engine_noise)
    mic_array = np.array([ref_mic_pos, error_mic_pos]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, room.fs))

    # Compute simulation
    room.simulate()

    # Extract signals
    ref_signal = room.mic_array.signals[0] # Signal at reference mic
    mic_signal = room.mic_array.signals[1] # Signal at error mic (Primary path)

    # Calculate Secondary Path (Hs) - Impulse response from speaker to ear
    room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=3)
    # Unit impulse for RIR extraction
    room_hs.add_source(speaker_pos, signal=np.array([1.0] + [0.0]*127)) 
    room_hs.add_microphone(error_mic_pos)
    room_hs.compute_rir()
    hs_rir = room_hs.rir[0][0].astype(np.float32)

    # Save to efficient .npy format
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", ref_signal)
    np.save(f"{prefix}_mic.npy", mic_signal)
    np.save(f"{prefix}_hs.npy", hs_rir)

    # Visual validation: save plot instead of audio
    plt.figure(figsize=(10, 4))
    plt.plot(mic_signal[:400], label="At Ear (Target)", alpha=0.8)
    plt.plot(ref_signal[:400], label="At Source (Ref)", linestyle='--')
    plt.title(f"Scenario {scn_id} - Time Domain Preview")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_plot.png")
    plt.close()

if __name__ == "__main__":
    print(f"Generating 5 quick scenarios in '{DATA_DIR}'...")
    for i in range(5):
        create_fast_scenario(i)
        print(f"  [+] Scenario {i} complete (npy + plot saved)")
    print("Success. Ready for fast training.")