# simulation_setup_phase_2_anechoic.py
"""
Enhanced dataset generation for Phase 2 New - ANECHOIC ONLY

FIXED: Uses MAX_ORDER=0 only (no reflections) like the original phase_2
This matches the acoustic conditions the old model was trained on.

Usage:
    python simulation_setup_phase_2_anechoic.py
"""

import os
import json
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
    'length': (3.5, 5.0),
    'width': (2.5, 3.5),
    'height': (1.8, 2.2),
}

# FIXED: Use ONLY anechoic (MAX_ORDER=0) like original phase_2
MAX_ORDER = 0  # No reflections - matches old successful approach

# Position ranges (normalized to room dims)
POSITION_RANGES = {
    'source': {
        'x_norm': (0.2, 0.4),
        'y_norm': (0.2, 0.4),
        'z_norm': (0.4, 0.6),
    },
    'ref_mic': {
        'x_norm': (0.3, 0.5),
        'y_norm': (0.3, 0.5),
        'z_norm': (0.4, 0.6),
    },
    'err_mic': {
        'x_norm': (0.7, 0.9),
        'y_norm': (0.7, 0.9),
        'z_norm': (0.4, 0.6),
    },
    'speaker': {
        'x_norm': (0.65, 0.85),
        'y_norm': (0.65, 0.85),
        'z_norm': (0.4, 0.6),
    },
}


def create_noise_signal(duration_seconds=DURATION, fs=FS, noise_type=None, seed=None):
    """Enhanced noise generation."""
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration_seconds * fs)
    t = np.arange(n_samples) / fs

    if noise_type is None:
        noise_type = np.random.choice(['engine', 'road', 'wind', 'mixed', 'ac'])

    if noise_type == 'engine':
        f0 = np.random.uniform(40, 150)
        n_harmonics = np.random.randint(3, 6)
        sig = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, n_harmonics + 1):
            amp = 1.0 / h
            sig += amp * np.sin(2 * np.pi * f0 * h * t)
        f_am = np.random.uniform(0.2, 1.5)
        am_depth = np.random.uniform(0.2, 0.4)
        am = (1 - am_depth) + am_depth * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    elif noise_type == 'road':
        white = np.random.randn(n_samples).astype(np.float32)
        win_len = np.random.randint(30, 80)
        kernel = np.ones(win_len) / win_len
        sig = np.convolve(white, kernel, mode='same')

    elif noise_type == 'wind':
        white = np.random.randn(n_samples).astype(np.float32)
        win_len = np.random.randint(30, 80)
        kernel = np.ones(win_len) / win_len
        smooth = np.convolve(white, kernel, mode='same')
        sig = white - smooth
        f_am = np.random.uniform(0.1, 0.7)
        am_depth = np.random.uniform(0.3, 0.6)
        am = (1 - am_depth) + am_depth * np.sin(2 * np.pi * f_am * t)
        sig = sig * am

    elif noise_type == 'ac':
        f_base = 60.0
        sig = (
            0.7 * np.sin(2 * np.pi * f_base * t) +
            0.4 * np.sin(2 * np.pi * 2 * f_base * t) +
            0.2 * np.sin(2 * np.pi * 3 * f_base * t) +
            0.1 * np.sin(2 * np.pi * 4 * f_base * t)
        )
        sig += 0.3 * np.random.randn(n_samples)

    elif noise_type == 'mixed':
        types = ['engine', 'road', 'wind', 'ac']
        weights = np.random.dirichlet(np.ones(len(types)))
        sig = np.zeros(n_samples, dtype=np.float32)
        for typ, weight in zip(types, weights):
            if weight > 0.1:
                sig += weight * create_noise_signal(duration_seconds, fs, typ, seed)

    else:
        sig = np.random.randn(n_samples).astype(np.float32)

    sig = sig.astype(np.float32)
    sig /= (np.max(np.abs(sig)) + 1e-9)
    return sig


def generate_random_position(room_dims, pos_type):
    """Generate random position within specified ranges."""
    ranges = POSITION_RANGES[pos_type]
    x = room_dims[0] * np.random.uniform(*ranges['x_norm'])
    y = room_dims[1] * np.random.uniform(*ranges['y_norm'])
    z = room_dims[2] * np.random.uniform(*ranges['z_norm'])
    return [x, y, z]


def simulate_scenario(out_dir, scenario_name, duration_seconds=DURATION, fs=FS,
                     room_dim=None, noise_type=None, seed=None):
    """Simulate one scenario."""
    if seed is not None:
        np.random.seed(seed)

    if room_dim is None:
        room_dim = [
            np.random.uniform(*ROOM_DIM_RANGES['length']),
            np.random.uniform(*ROOM_DIM_RANGES['width']),
            np.random.uniform(*ROOM_DIM_RANGES['height']),
        ]

    # FIXED: Always use MAX_ORDER=0 (anechoic)
    max_order = MAX_ORDER

    source_loc = generate_random_position(room_dim, 'source')
    ref_mic_loc = generate_random_position(room_dim, 'ref_mic')
    err_mic_loc = generate_random_position(room_dim, 'err_mic')
    spk_loc = generate_random_position(room_dim, 'speaker')

    os.makedirs(out_dir, exist_ok=True)

    room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order)
    noise_signal = create_noise_signal(duration_seconds, fs, noise_type, seed)

    room.add_source(source_loc, signal=noise_signal)
    spk_silence = np.zeros_like(noise_signal, dtype=np.float32)
    room.add_source(spk_loc, signal=spk_silence)

    mic_positions = np.c_[ref_mic_loc, err_mic_loc]
    mic_array = pra.MicrophoneArray(mic_positions, fs=fs)
    room.add_microphone_array(mic_array)

    print(f"[{scenario_name}] Simulating... (room={[f'{d:.1f}' for d in room_dim]}, order={max_order}, noise={noise_type})")
    room.simulate()
    room.compute_rir()

    ref_sig = room.mic_array.signals[0, :].astype(np.float32)
    err_sig = room.mic_array.signals[1, :].astype(np.float32)

    ref_sig /= (np.max(np.abs(ref_sig)) + 1e-9)
    err_sig /= (np.max(np.abs(err_sig)) + 1e-9)

    ref_wav_path = os.path.join(out_dir, f"{scenario_name}_ref.wav")
    err_wav_path = os.path.join(out_dir, f"{scenario_name}_err.wav")

    write(ref_wav_path, fs, (ref_sig * 32767).astype(np.int16))
    write(err_wav_path, fs, (err_sig * 32767).astype(np.int16))

    rir_src_to_ref = np.array(room.rir[0][0], dtype=np.float32)
    rir_src_to_err = np.array(room.rir[1][0], dtype=np.float32)
    rir_spk_to_err = np.array(room.rir[1][1], dtype=np.float32)

    np.save(os.path.join(out_dir, f"{scenario_name}_rir_src_to_ref.npy"), rir_src_to_ref)
    np.save(os.path.join(out_dir, f"{scenario_name}_rir_src_to_err.npy"), rir_src_to_err)
    np.save(os.path.join(out_dir, f"{scenario_name}_rir_spk_to_err.npy"), rir_spk_to_err)

    print(f"[{scenario_name}] ✓ Complete")

    return {
        'scenario_name': scenario_name,
        'room_dim': room_dim,
        'max_order': int(max_order),  # Convert to int for JSON
        'noise_type': noise_type,
        'source_loc': source_loc,
        'ref_mic_loc': ref_mic_loc,
        'err_mic_loc': err_mic_loc,
        'spk_loc': spk_loc,
    }


def generate_dataset(base_out_dir="data", split="train", num_scenarios=50,
                    duration_seconds=DURATION, noise_type_distribution=None):
    """Generate multiple scenarios for a given split."""
    out_dir = os.path.join(base_out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    if noise_type_distribution is None:
        noise_type_distribution = {
            'engine': 0.35,
            'road': 0.20,
            'wind': 0.15,
            'ac': 0.10,
            'mixed': 0.20,
        }

    noise_types = list(noise_type_distribution.keys())
    probabilities = list(noise_type_distribution.values())
    sampled_noise_types = np.random.choice(noise_types, size=num_scenarios, p=probabilities)

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
            seed=None,
        )
        metadata.append(meta)

        if (i + 1) % 10 == 0:
            print(f"\n=== Progress: {i+1}/{num_scenarios} scenarios complete ===\n")

    metadata_path = os.path.join(out_dir, f"{split}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete: {split}")
    print(f"Total scenarios: {num_scenarios}")
    print(f"Metadata saved: {metadata_path}")
    print(f"{'='*60}\n")

    return metadata


if __name__ == "__main__":
    print("="*60)
    print("FIXED Dataset Generation - ANECHOIC ONLY (MAX_ORDER=0)")
    print("="*60)
    print("\nThis uses MAX_ORDER=0 (no reflections) like the original phase_2")
    print("which achieved +3.62 dB successfully.\n")
    print("Generating 50 training + 10 validation scenarios...")
    print("Estimated time: 10-20 minutes\n")

    # Generate training set (50 scenarios)
    print("\n[1/2] Generating training set (50 scenarios)...\n")
    train_meta = generate_dataset(
        base_out_dir="data",
        split="train",
        num_scenarios=50,
        duration_seconds=DURATION,
    )

    # Generate validation set (10 scenarios)
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
    print(f"  Training scenarios:   50 (anechoic, MAX_ORDER=0)")
    print(f"  Validation scenarios: 10 (anechoic, MAX_ORDER=0)")
    print("\nKey differences from previous attempt:")
    print("  ✓ FIXED: MAX_ORDER=0 only (no reflections)")
    print("  ✓ Matches original phase_2 acoustic conditions")
    print("  ✓ Should achieve positive dB like original (+3.62 dB baseline)")
    print("\nNext steps:")
    print("  1. python -m phase_2_new.training.train --config baseline")
    print("  2. python -m phase_2_new.testing.evaluate --experiment exp_00_baseline")
    print("="*60)
