# simulation_setup_phase_2_curriculum.py
"""
Curriculum Learning: Gradual introduction of acoustic reflections

Expands dataset from 50 to 100 scenarios with:
- 70% MAX_ORDER=0 (anechoic, proven to work)
- 30% MAX_ORDER=1 (first-order reflections, gradual complexity)

Appends to existing dataset (scenarios 050-099).
NO wind noise (breaks feedforward ANC).
"""

import os
import json
import numpy as np
import pyroomacoustics as pra
from scipy.io.wavfile import write

FS = 16000
DURATION = 4.0

ROOM_DIM_RANGES = {
    'length': (3.5, 5.0),
    'width': (2.5, 3.5),
    'height': (1.8, 2.2),
}

# Curriculum distribution: 70% anechoic, 30% first-order reflections
MAX_ORDER_DISTRIBUTION = {
    0: 0.70,  # 70 scenarios anechoic (proven to work better)
    1: 0.30,  # 30 scenarios with first reflections (gradual complexity)
}

POSITION_RANGES = {
    'source': {'x_norm': (0.2, 0.4), 'y_norm': (0.2, 0.4), 'z_norm': (0.4, 0.6)},
    'ref_mic': {'x_norm': (0.3, 0.5), 'y_norm': (0.3, 0.5), 'z_norm': (0.4, 0.6)},
    'err_mic': {'x_norm': (0.7, 0.9), 'y_norm': (0.7, 0.9), 'z_norm': (0.4, 0.6)},
    'speaker': {'x_norm': (0.65, 0.85), 'y_norm': (0.65, 0.85), 'z_norm': (0.4, 0.6)},
}


def create_noise_signal(duration_seconds=DURATION, fs=FS, noise_type=None, seed=None):
    """Enhanced noise generation - NO WIND"""
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration_seconds * fs)
    t = np.arange(n_samples) / fs

    if noise_type is None:
        # FIXED: No 'wind' in choices!
        noise_type = np.random.choice(['engine', 'road', 'ac', 'mixed'])

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
        # FIXED: No wind in mixed!
        types = ['engine', 'road', 'ac']
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
    ranges = POSITION_RANGES[pos_type]
    x = room_dims[0] * np.random.uniform(*ranges['x_norm'])
    y = room_dims[1] * np.random.uniform(*ranges['y_norm'])
    z = room_dims[2] * np.random.uniform(*ranges['z_norm'])
    return [x, y, z]


def simulate_scenario(out_dir, scenario_name, max_order=0, duration_seconds=DURATION,
                     fs=FS, room_dim=None, noise_type=None, seed=None):
    """
    Modified to accept max_order as parameter instead of global constant
    """
    if seed is not None:
        np.random.seed(seed)

    if room_dim is None:
        room_dim = [
            np.random.uniform(*ROOM_DIM_RANGES['length']),
            np.random.uniform(*ROOM_DIM_RANGES['width']),
            np.random.uniform(*ROOM_DIM_RANGES['height']),
        ]

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
        'max_order': int(max_order),
        'noise_type': noise_type,
        'source_loc': source_loc,
        'ref_mic_loc': ref_mic_loc,
        'err_mic_loc': err_mic_loc,
        'spk_loc': spk_loc,
    }


def generate_dataset(base_out_dir="data", split="train", num_scenarios=50,
                    start_idx=0, duration_seconds=DURATION, noise_type_distribution=None,
                    max_order_distribution=None):
    """
    Modified to support:
    - start_idx: Start scenario numbering from this index (for appending)
    - max_order_distribution: Sample MAX_ORDER per scenario
    """
    out_dir = os.path.join(base_out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    if noise_type_distribution is None:
        # FIXED: No wind! Redistributed probabilities
        noise_type_distribution = {
            'engine': 0.50,  # Increased from 0.35
            'road': 0.25,    # Increased from 0.20
            'ac': 0.10,      # Same
            'mixed': 0.15,   # Decreased from 0.20 (no wind in mixed now)
        }

    if max_order_distribution is None:
        max_order_distribution = MAX_ORDER_DISTRIBUTION

    # Sample noise types and MAX_ORDER for all scenarios
    noise_types = list(noise_type_distribution.keys())
    noise_probabilities = list(noise_type_distribution.values())
    sampled_noise_types = np.random.choice(noise_types, size=num_scenarios, p=noise_probabilities)

    max_orders = list(max_order_distribution.keys())
    max_order_probabilities = list(max_order_distribution.values())
    sampled_max_orders = np.random.choice(max_orders, size=num_scenarios, p=max_order_probabilities)

    # Load existing metadata if appending
    metadata_path = os.path.join(out_dir, f"{split}_metadata.json")
    if os.path.exists(metadata_path) and start_idx > 0:
        print(f"Loading existing metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Found {len(metadata)} existing scenarios")
    else:
        metadata = []

    # Generate new scenarios
    for i in range(num_scenarios):
        scenario_idx = start_idx + i
        scenario_name = f"{split}_scenario_{scenario_idx:03d}"
        noise_type = sampled_noise_types[i]
        max_order = sampled_max_orders[i]

        meta = simulate_scenario(
            out_dir=out_dir,
            scenario_name=scenario_name,
            max_order=max_order,
            duration_seconds=duration_seconds,
            fs=FS,
            noise_type=noise_type,
            seed=None,
        )
        metadata.append(meta)

        if (i + 1) % 10 == 0:
            print(f"\n=== Progress: {i+1}/{num_scenarios} new scenarios complete ===")
            print(f"    Total scenarios so far: {len(metadata)}\n")

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete: {split}")
    print(f"New scenarios: {num_scenarios} (indices {start_idx:03d}-{start_idx+num_scenarios-1:03d})")
    print(f"Total scenarios: {len(metadata)}")
    print(f"Metadata saved: {metadata_path}")
    print(f"{'='*60}\n")

    return metadata


if __name__ == "__main__":
    print("="*60)
    print("CURRICULUM LEARNING Dataset Expansion")
    print("="*60)
    print("\nStrategy:")
    print("  - 70% MAX_ORDER=0 (anechoic, proven to work)")
    print("  - 30% MAX_ORDER=1 (first-order reflections, gradual complexity)")
    print("  - NO wind noise (correlation ~0 with ref mic)")
    print("  - Noise: engine (50%), road (25%), ac (10%), mixed (15%)")
    print("\nGenerating 50 NEW training scenarios (050-099)...")
    print("Appending to existing 50 scenarios (000-049)")
    print("Estimated time: 20-30 minutes\n")

    print("\nGenerating training set expansion (scenarios 050-099)...\n")
    train_meta = generate_dataset(
        base_out_dir="data",
        split="train",
        num_scenarios=50,
        start_idx=50,  # Start at scenario 050
        duration_seconds=DURATION,
    )

    print("\n" + "=" * 60)
    print("EXPANSION COMPLETE!")
    print("=" * 60)
    print("\nDataset summary:")
    print(f"  Training scenarios:   {len(train_meta)} total")
    print(f"    - Scenarios 000-049: All MAX_ORDER=0 (existing)")
    print(f"    - Scenarios 050-099: ~70% ORDER=0, ~30% ORDER=1 (new)")
    print(f"  Validation scenarios: 10 (unchanged, all MAX_ORDER=0)")

    # Count MAX_ORDER distribution in new scenarios
    new_scenarios = train_meta[50:]
    order_0_count = sum(1 for s in new_scenarios if s['max_order'] == 0)
    order_1_count = sum(1 for s in new_scenarios if s['max_order'] == 1)

    print(f"\nActual MAX_ORDER distribution (new scenarios 050-099):")
    print(f"  ORDER=0 (anechoic):     {order_0_count} scenarios ({order_0_count/50*100:.0f}%)")
    print(f"  ORDER=1 (reflections):  {order_1_count} scenarios ({order_1_count/50*100:.0f}%)")

    print("\nNoise distribution (same as before):")
    print("  Engine: 50% (harmonic, predictable - primary target)")
    print("  Road:   25% (broadband, low-freq)")
    print("  AC:     10% (60Hz harmonics)")
    print("  Mixed:  15% (combinations of above)")

    print("\nNext steps:")
    print("  1. Verify dataset: Check train_metadata.json has 100 entries")
    print("  2. Train model: python -m phase_2_new.training.train --config baseline")
    print("  3. Evaluate: python -m phase_2_new.testing.evaluate --experiment exp_00_baseline")
    print("  4. Compare: Check if curriculum learning improves generalization")
    print("="*60)
