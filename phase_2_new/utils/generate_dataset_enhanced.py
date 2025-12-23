# phase_2_new/utils/generate_dataset_enhanced.py
"""
Enhanced dataset generation for Phase 2 New.

Improvements over original simulation_setup.py:
- More training scenarios (50 instead of 10)
- More diverse room configurations
- Varied noise characteristics
- More validation scenarios for better evaluation
"""

import os
import numpy as np
import pyroomacoustics as pra
from scipy.io.wavfile import write

# =========================
# Configuration
# =========================

FS = 16000
DURATION = 4.0

# Room dimension ranges (more variety)
ROOM_DIM_RANGES = {
    'length': (3.5, 5.0),  # meters
    'width': (2.5, 3.5),
    'height': (1.8, 2.2),
}

# Reflection orders (mix of conditions)
MAX_ORDER_OPTIONS = [0, 1, 2, 3]  # 0=anechoic, 3=realistic

# Position ranges (normalized to room dims)
POSITION_RANGES = {
    'source': {
        'x_norm': (0.2, 0.4),  # fraction of room length
        'y_norm': (0.2, 0.4),
        'z_norm': (0.4, 0.6),
    },
    'ref_mic': {
        'x_norm': (0.3, 0.5),
        'y_norm': (0.3, 0.5),
        'z_norm': (0.4, 0.6),
    },
    'err_mic': {
        'x_norm': (0.7, 0.9),  # far from source
        'y_norm': (0.7, 0.9),
        'z_norm': (0.4, 0.6),
    },
    'speaker': {
        'x_norm': (0.65, 0.85),  # near error mic
        'y_norm': (0.65, 0.85),
        'z_norm': (0.4, 0.6),
    },
}


# =========================
# Noise Generation
# =========================

def create_noise_signal(duration_seconds=DURATION, fs=FS, noise_type=None, seed=None):
    """
    Enhanced noise generation with more variety.

    Noise types:
    - 'engine': Harmonic tones with AM (various base frequencies)
    - 'road': Filtered broadband (various filter characteristics)
    - 'wind': High-pass filtered noise with AM
    - 'mixed': Combination of above
    - 'ac': HVAC-like hum (multiple harmonics)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration_seconds * fs)
    t = np.arange(n_samples) / fs

    if noise_type is None:
        noise_type = np.random.choice(['engine', 'road', 'wind', 'mixed', 'ac'])

    # ---------- ENGINE NOISE ----------
    if noise_type == 'engine':
        # More variety in engine speeds
        f0 = np.random.uniform(40, 150)  # 40-150 Hz base (idle to highway)
        n_harmonics = np.random.randint(3, 6)

        sig = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, n_harmonics + 1):
            amp = 1.0 / h  # Decreasing amplitude
            sig += amp * np.sin(2 * np.pi * f0 * h * t)

        # AM modulation (engine RPM variation)
        f_am = np.random.uniform(0.2, 1.5)
        am_depth = np.random.uniform(0.2, 0.4)
        am = (1 - am_depth) + am_depth * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    # ---------- ROAD RUMBLE ----------
    elif noise_type == 'road':
        white = np.random.randn(n_samples).astype(np.float32)

        # Variable smoothing (tire characteristics)
        win_len = np.random.randint(30, 80)
        kernel = np.ones(win_len) / win_len
        sig = np.convolve(white, kernel, mode='same')

    # ---------- WIND NOISE ----------
    elif noise_type == 'wind':
        white = np.random.randn(n_samples).astype(np.float32)

        # High-pass characteristic
        win_len = np.random.randint(30, 80)
        kernel = np.ones(win_len) / win_len
        smooth = np.convolve(white, kernel, mode='same')
        sig = white - smooth

        # Gusty AM
        f_am = np.random.uniform(0.1, 0.7)
        am_depth = np.random.uniform(0.3, 0.6)
        am = (1 - am_depth) + am_depth * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    # ---------- AC/HVAC NOISE ----------
    elif noise_type == 'ac':
        # Multiple harmonics at 60Hz and its multiples (power line hum + fan)
        f_base = 60.0
        sig = (
            0.7 * np.sin(2 * np.pi * f_base * t) +
            0.4 * np.sin(2 * np.pi * 2 * f_base * t) +
            0.2 * np.sin(2 * np.pi * 3 * f_base * t) +
            0.1 * np.sin(2 * np.pi * 4 * f_base * t)
        )
        # Add some broadband air noise
        sig += 0.3 * np.random.randn(n_samples)

    # ---------- MIXED ----------
    elif noise_type == 'mixed':
        # Random mix of noise types
        types = ['engine', 'road', 'wind', 'ac']
        weights = np.random.dirichlet(np.ones(len(types)))

        sig = np.zeros(n_samples, dtype=np.float32)
        for typ, weight in zip(types, weights):
            if weight > 0.1:  # Only include significant components
                sig += weight * create_noise_signal(duration_seconds, fs, typ, seed)

    else:
        # Fallback to white noise
        sig = np.random.randn(n_samples).astype(np.float32)

    # Normalize
    sig = sig.astype(np.float32)
    sig /= (np.max(np.abs(sig)) + 1e-9)

    return sig


# =========================
# Position Generation
# =========================

def generate_random_position(room_dims, pos_type):
    """Generate a random position within specified normalized ranges."""
    ranges = POSITION_RANGES[pos_type]

    x = room_dims[0] * np.random.uniform(*ranges['x_norm'])
    y = room_dims[1] * np.random.uniform(*ranges['y_norm'])
    z = room_dims[2] * np.random.uniform(*ranges['z_norm'])

    return [x, y, z]


# =========================
# Single Scenario Simulation
# =========================

def simulate_scenario(
    out_dir,
    scenario_name,
    duration_seconds=DURATION,
    fs=FS,
    room_dim=None,
    max_order=None,
    noise_type=None,
    seed=None,
):
    """
    Simulate one scenario with optional randomization.

    If parameters are None, they will be randomly chosen.
    """
    if seed is not None:
        np.random.seed(seed)

    # Randomize room dimensions if not provided
    if room_dim is None:
        room_dim = [
            np.random.uniform(*ROOM_DIM_RANGES['length']),
            np.random.uniform(*ROOM_DIM_RANGES['width']),
            np.random.uniform(*ROOM_DIM_RANGES['height']),
        ]

    # Randomize reflection order if not provided
    if max_order is None:
        max_order = np.random.choice(MAX_ORDER_OPTIONS)

    # Generate positions
    source_loc = generate_random_position(room_dim, 'source')
    ref_mic_loc = generate_random_position(room_dim, 'ref_mic')
    err_mic_loc = generate_random_position(room_dim, 'err_mic')
    spk_loc = generate_random_position(room_dim, 'speaker')

    os.makedirs(out_dir, exist_ok=True)

    # Create room
    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order)

    # Create noise signal
    noise_signal = create_noise_signal(
        duration_seconds=duration_seconds,
        fs=fs,
        noise_type=noise_type,
        seed=seed,
    )

    # Add sources
    room.add_source(source_loc, signal=noise_signal)  # noise source
    spk_silence = np.zeros_like(noise_signal, dtype=np.float32)
    room.add_source(spk_loc, signal=spk_silence)  # speaker (for RIR)

    # Add microphones
    mic_positions = np.c_[ref_mic_loc, err_mic_loc]
    mic_array = pra.MicrophoneArray(mic_positions, fs=fs)
    room.add_microphone_array(mic_array)

    # Simulate
    print(f"[{scenario_name}] Simulating... (room={room_dim}, order={max_order}, noise={noise_type})")
    room.simulate()
    room.compute_rir()

    # Extract signals
    ref_sig = room.mic_array.signals[0, :].astype(np.float32)
    err_sig = room.mic_array.signals[1, :].astype(np.float32)

    # Normalize
    ref_sig /= (np.max(np.abs(ref_sig)) + 1e-9)
    err_sig /= (np.max(np.abs(err_sig)) + 1e-9)

    # Save WAVs
    ref_wav_path = os.path.join(out_dir, f"{scenario_name}_ref.wav")
    err_wav_path = os.path.join(out_dir, f"{scenario_name}_err.wav")

    write(ref_wav_path, fs, (ref_sig * 32767).astype(np.int16))
    write(err_wav_path, fs, (err_sig * 32767).astype(np.int16))

    # Extract and save RIRs
    rir_src_to_ref = np.array(room.rir[0][0], dtype=np.float32)
    rir_src_to_err = np.array(room.rir[1][0], dtype=np.float32)
    rir_spk_to_err = np.array(room.rir[1][1], dtype=np.float32)

    rir_ref_path = os.path.join(out_dir, f"{scenario_name}_rir_src_to_ref.npy")
    rir_err_path = os.path.join(out_dir, f"{scenario_name}_rir_src_to_err.npy")
    rir_spk_err_path = os.path.join(out_dir, f"{scenario_name}_rir_spk_to_err.npy")

    np.save(rir_ref_path, rir_src_to_ref)
    np.save(rir_err_path, rir_src_to_err)
    np.save(rir_spk_err_path, rir_spk_to_err)

    print(f"[{scenario_name}] ✓ Complete")

    return {
        'scenario_name': scenario_name,
        'room_dim': room_dim,
        'max_order': max_order,
        'noise_type': noise_type,
        'source_loc': source_loc,
        'ref_mic_loc': ref_mic_loc,
        'err_mic_loc': err_mic_loc,
        'spk_loc': spk_loc,
    }


# =========================
# Dataset Generation
# =========================

def generate_dataset(
    base_out_dir="data",
    split="train",
    num_scenarios=50,
    duration_seconds=DURATION,
    noise_type_distribution=None,
):
    """
    Generate multiple scenarios for a given split.

    Args:
        noise_type_distribution: Dict of noise_type -> probability
            e.g., {'engine': 0.4, 'road': 0.2, 'mixed': 0.3, 'ac': 0.1}
    """
    out_dir = os.path.join(base_out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # Default noise distribution
    if noise_type_distribution is None:
        noise_type_distribution = {
            'engine': 0.35,
            'road': 0.20,
            'wind': 0.15,
            'ac': 0.10,
            'mixed': 0.20,
        }

    # Sample noise types according to distribution
    noise_types = list(noise_type_distribution.keys())
    probabilities = list(noise_type_distribution.values())
    sampled_noise_types = np.random.choice(
        noise_types,
        size=num_scenarios,
        p=probabilities,
    )

    metadata = []

    for i in range(num_scenarios):
        scenario_name = f"{split}_scenario_{i:03d}"
        noise_type = sampled_noise_types[i]

        meta = simulate_scenario(
            out_dir=out_dir,
            scenario_name=scenario_name,
            duration_seconds=duration_seconds,
            fs=FS,
            noise_type=noise_type,
            seed=None,  # Random each time
        )
        metadata.append(meta)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"\n=== Progress: {i+1}/{num_scenarios} scenarios complete ===\n")

    # Save metadata
    import json
    metadata_path = os.path.join(out_dir, f"{split}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete: {split}")
    print(f"Total scenarios: {num_scenarios}")
    print(f"Metadata saved: {metadata_path}")
    print(f"{'='*60}\n")

    return metadata


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("Enhanced Dataset Generation for Phase 2 New")
    print("=" * 60)

    # Generate training set (50 scenarios)
    print("\n[1/2] Generating training set (50 scenarios)...\n")
    train_meta = generate_dataset(
        base_out_dir="data",
        split="train",
        num_scenarios=50,
        duration_seconds=DURATION,
    )

    # Generate validation set (10 scenarios for better evaluation)
    print("\n[2/2] Generating validation set (10 scenarios)...\n")
    val_meta = generate_dataset(
        base_out_dir="data",
        split="val",
        num_scenarios=10,
        duration_seconds=DURATION,
    )

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nDataset summary:")
    print(f"  Training scenarios:   50 (in data/train/)")
    print(f"  Validation scenarios: 10 (in data/val/)")
    print("\nNext steps:")
    print("  1. Verify data: ls -lh data/train/ | head -20")
    print("  2. Start training experiments")
    print("=" * 60)
