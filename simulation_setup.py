import os
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io.wavfile import write
import os
import random

print("All libraries imported successfully!")
print(f"Using pyroomacoustics version: {pra.__version__}")

# =========================
# Global / default settings
# =========================

FS = 16000                # sampling frequency
ROOM_DIM = [4.0, 3.0, 2.0]  # L, W, H (meters)
MAX_ORDER = 0        # reflection order

# Positions (m) – toy “car cabin”
SOURCE_LOC     = [1.0, 0.5, 1.0]   # noise source (engine-ish)
REF_MIC_LOC    = [1.3, 0.7, 1.0]   # reference mic near source
ERR_MIC_LOC    = [3.0, 2.5, 1.0]   # error mic at driver head (far corner)
SPK_LOC        = [2.8, 2.2, 1.0]   # NEW: loudspeaker near driver



# ===================================================
# 1. Noise generator – structured car-like noises
# ===================================================
def create_noise_signal(duration_seconds=4.0, fs=FS, noise_type=None, seed=None):
    """
    Create a structured noise signal for ANC training.

    noise_type: 'engine', 'road', 'wind', 'mixed', or None (random choice).
    """

    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration_seconds * fs)
    t = np.arange(n_samples) / fs

    if noise_type is None:
        noise_type = np.random.choice(['engine', 'road', 'wind', 'mixed'])

    # ---------- ENGINE NOISE ----------
    if noise_type == 'engine':
        f0 = np.random.uniform(50, 120)  # base freq
        f1 = 2 * f0
        f2 = 3 * f0

        sig = (
            0.6 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t)
        )

        # slow AM (RPM variation)
        f_am = np.random.uniform(0.2, 1.0)
        am = 0.7 + 0.3 * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    # ---------- ROAD RUMBLE ----------
    elif noise_type == 'road':
        white = np.random.randn(n_samples).astype(np.float32)

        win_len = np.random.randint(20, 51)  # 20–50 taps
        kernel = np.ones(win_len) / win_len
        sig = np.convolve(white, kernel, mode='same')

    # ---------- WIND NOISE ----------
    elif noise_type == 'wind':
        white = np.random.randn(n_samples).astype(np.float32)

        win_len = np.random.randint(20, 51)
        kernel = np.ones(win_len) / win_len
        smooth = np.convolve(white, kernel, mode='same')
        sig = white - smooth  # high-passy

        f_am = np.random.uniform(0.1, 0.5)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    # ---------- MIXED ----------
    elif noise_type == 'mixed':
        engine = create_noise_signal(duration_seconds, fs, noise_type='engine')
        road   = create_noise_signal(duration_seconds, fs, noise_type='road')
        wind   = create_noise_signal(duration_seconds, fs, noise_type='wind')
        sig = 0.5 * engine + 0.3 * road + 0.2 * wind

    else:
        sig = np.random.randn(n_samples).astype(np.float32)

    sig = sig.astype(np.float32)
    sig /= np.max(np.abs(sig) + 1e-9)  # normalize
    return sig


# ===================================================
# 2. Single scenario simulation (2 mics)
# ===================================================
def simulate_single_scenario(
    out_dir,
    scenario_name,
    duration_seconds=4.0,
    fs=FS,
    room_dim=ROOM_DIM,
    source_loc=SOURCE_LOC,
    ref_mic_loc=REF_MIC_LOC,
    err_mic_loc=ERR_MIC_LOC,
    spk_loc=SPK_LOC,              
    max_order=MAX_ORDER,
    noise_type=None,
    save_plot=False
):

    """
    Simulate one 2-mic scenario:

      - structured noise source
      - reference mic near source
      - error mic at driver head

   Saves:
      *_ref.wav, *_err.wav,
      *_rir_src_to_ref.npy, *_rir_src_to_err.npy,
      *_rir_spk_to_err.npy
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Create room
    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order)
    print(f"[{scenario_name}] Room created.")

    # 2) Create noise signal
    noise_signal = create_noise_signal(
        duration_seconds=duration_seconds,
        fs=fs,
        noise_type=noise_type
    )
    print(f"[{scenario_name}] Noise signal created ({noise_type}).")

       # 3) Add noise source
    room.add_source(source_loc, signal=noise_signal)  # source index 0
    print(f"[{scenario_name}] Source added at {source_loc}.")

    #    Add loudspeaker as a second source with a zero signal
    spk_silence = np.zeros_like(noise_signal, dtype=np.float32)
    room.add_source(spk_loc, signal=spk_silence)      # source index 1 (speaker)
    print(f"[{scenario_name}] Speaker (for RIR) added at {spk_loc}.")

    # 4) Add 2 microphones: reference + error
    mic_positions = np.c_[ref_mic_loc, err_mic_loc]  # shape (3, 2)
    mic_array = pra.MicrophoneArray(mic_positions, fs=fs)
    room.add_microphone_array(mic_array)
    print(f"[{scenario_name}] Reference mic at {ref_mic_loc}, error mic at {err_mic_loc}.")

    # 5) Simulate
    print(f"[{scenario_name}] Running simulation...")
    room.simulate()
    print(f"[{scenario_name}] Simulation finished.")
        # Compute RIRs explicitly (for all source–mic pairs)
    room.compute_rir()


    # 6) Extract mic signals
    ref_sig = room.mic_array.signals[0, :].astype(np.float32)
    err_sig = room.mic_array.signals[1, :].astype(np.float32)

    # Normalize separately
    ref_sig /= np.max(np.abs(ref_sig) + 1e-9)
    err_sig /= np.max(np.abs(err_sig) + 1e-9)

    # 7) Save WAVs
    ref_wav_path = os.path.join(out_dir, f"{scenario_name}_ref.wav")
    err_wav_path = os.path.join(out_dir, f"{scenario_name}_err.wav")

    write(ref_wav_path, fs, (ref_sig * 32767).astype(np.int16))
    write(err_wav_path, fs, (err_sig * 32767).astype(np.int16))

    print(f"[{scenario_name}] Reference mic signal saved to: {ref_wav_path}")
    print(f"[{scenario_name}] Error mic signal saved to:     {err_wav_path}")

    # 8) Save RIRs:
    #    source -> ref mic, source -> error mic, speaker -> error mic
    rir_src_to_ref = np.array(room.rir[0][0], dtype=np.float32)  # mic0 <- src
    rir_src_to_err = np.array(room.rir[1][0], dtype=np.float32)  # mic1 <- src
    rir_spk_to_err = np.array(room.rir[1][1], dtype=np.float32)  # mic1 <- speaker

    rir_ref_path = os.path.join(out_dir, f"{scenario_name}_rir_src_to_ref.npy")
    rir_err_path = os.path.join(out_dir, f"{scenario_name}_rir_src_to_err.npy")
    rir_spk_err_path = os.path.join(out_dir, f"{scenario_name}_rir_spk_to_err.npy")  # NEW

    np.save(rir_ref_path, rir_src_to_ref)
    np.save(rir_err_path, rir_src_to_err)
    np.save(rir_spk_err_path, rir_spk_to_err)  # NEW

    print(f"[{scenario_name}] RIR source->ref saved to: {rir_ref_path}")
    print(f"[{scenario_name}] RIR source->err saved to: {rir_err_path}")
    print(f"[{scenario_name}] RIR speaker->err saved to: {rir_spk_err_path}")

    # 9) Optional plot: error mic signal
    if save_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(err_sig)
        plt.title(f"Error mic signal - {scenario_name}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{scenario_name}_err_waveform.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[{scenario_name}] Plot saved to: {plot_path}")

    return ref_wav_path, err_wav_path, rir_ref_path, rir_err_path


# ===================================================
# 3. Dataset generation (train / val / test)
# ===================================================
def generate_dataset(
    base_out_dir="data",
    split="train",
    num_scenarios=5,
    duration_seconds=4.0,
    randomize_positions=False,
    noise_types=('engine',),
):
    """
    Generate multiple 2-mic scenarios for a given split (train/val/test).
    """

    out_dir = os.path.join(base_out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_scenarios):
        scenario_name = f"{split}_scenario_{i:03d}"

        # pick a noise type
        noise_type = np.random.choice(noise_types)

        # Optionally jitter positions a bit
        if randomize_positions:
            def jitter(loc):
                loc = np.array(loc, dtype=np.float32)
                loc += np.random.uniform(-0.2, 0.2, size=3)
                loc = np.clip(loc, [0.1, 0.1, 0.5], np.array(ROOM_DIM) - [0.1, 0.1, 0.5])
                return loc.tolist()

            src_loc = jitter(SOURCE_LOC)
            ref_loc = jitter(REF_MIC_LOC)
            err_loc = jitter(ERR_MIC_LOC)
        else:
            src_loc = SOURCE_LOC
            ref_loc = REF_MIC_LOC
            err_loc = ERR_MIC_LOC

        print(f"\n=== Generating {scenario_name} (noise={noise_type}) ===")
        simulate_single_scenario(
            out_dir=out_dir,
            scenario_name=scenario_name,
            duration_seconds=duration_seconds,
            fs=FS,
            room_dim=ROOM_DIM,
            source_loc=src_loc,
            ref_mic_loc=ref_loc,
            err_mic_loc=err_loc,
            spk_loc=SPK_LOC,          # NEW (we keep speaker fixed for now)
            max_order=MAX_ORDER,
            noise_type=noise_type,
            save_plot=False
        )



if __name__ == "__main__":
    # Example: generate train + val
    generate_dataset(base_out_dir="data", split="train",
                     num_scenarios=10, duration_seconds=4.0,
                     randomize_positions=True)
    generate_dataset(base_out_dir="data", split="val",
                     num_scenarios=3, duration_seconds=4.0,
                     randomize_positions=True)
    print("2-mic dataset generation complete.")
