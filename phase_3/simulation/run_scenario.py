# phase_3/simulation/run_scenario.py

import os
# ---- IMPORTANT: set before importing tensorflow ----
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pyroomacoustics as pra
import tensorflow as tf

FS = 16000
SEQUENCE_LENGTH = 200
DEFAULT_DURATION = 3.0
DEFAULT_MODEL_PATH = "phase_2/models/tcn_ref2u.keras"

MATERIAL_PRESETS = {
    "hard": 0.10,
    "mixed": 0.40,
    "soft": 0.70,
}


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.max(np.abs(x)) + 1e-9)


def generate_noise(noise_type: str, duration: float, fs: int) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if noise_type == "engine":
        base = 60.0
        x = (
            1.0 * np.sin(2 * np.pi * base * t)
            + 0.5 * np.sin(2 * np.pi * 2 * base * t)
            + 0.3 * np.sin(2 * np.pi * 3 * base * t)
        )
    elif noise_type == "road":
        x = np.random.randn(len(t))
    elif noise_type == "cabin":
        x = 0.8 * np.random.randn(len(t)) + 0.2 * np.sin(2 * np.pi * 120.0 * t)
    else:  # mixed
        x = 0.7 * generate_noise("engine", duration, fs) + 0.3 * np.random.randn(len(t))

    return _normalize(x)


def build_windows_strided(x: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Returns a view with shape (num_windows, seq_len, 1), where
    num_windows = N - seq_len + 1
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    if N < seq_len:
        raise ValueError(f"Signal too short: N={N} < seq_len={seq_len}")

    num = N - seq_len + 1
    stride = x.strides[0]
    X2 = np.lib.stride_tricks.as_strided(
        x,
        shape=(num, seq_len),
        strides=(stride, stride),
        writeable=False,
    )
    return X2[..., None]  # (num, seq_len, 1)


def apply_secondary_path_time(u: np.ndarray, h: np.ndarray, N: int) -> np.ndarray:
    """
    y = (h * u) truncated to length N (pure numpy time-domain convolution).
    Stable and deterministic (but slower than FFT).
    """
    u = np.asarray(u, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)

    y_full = np.convolve(u, h, mode="full").astype(np.float32)
    return y_full[:N]


def run_scenario(
    room_dims=(4.2, 3.1, 1.9),
    material_preset="mixed",
    noise_type="engine",
    noise_position=(1.0, 0.9, 1.0),
    ref_mic_position=(1.5, 1.1, 1.0),
    err_mic_position=(3.0, 2.3, 1.0),
    speaker_position=(2.8, 2.2, 1.0),
    duration=DEFAULT_DURATION,
    model_path=DEFAULT_MODEL_PATH,
    anc_enabled=True,
    max_order=3,
    predict_batch_size=64,
):
    """
    Sources:
      src0 = noise (signal injected)
      src1 = control speaker (silent, used only to get RIR)

    Mics:
      mic0 = reference mic
      mic1 = error mic

    Returns:
      dict with signals/metrics/geometry/rir
    """
    print(">>> Loading ANC model...", flush=True)
    model = tf.keras.models.load_model(model_path)
    print(">>> Model loaded", flush=True)

    absorption = MATERIAL_PRESETS.get(material_preset, 0.4)

    print(">>> Building room", flush=True)
    room = pra.ShoeBox(
        room_dims,
        fs=FS,
        materials=pra.Material(absorption),
        max_order=max_order,
    )

    print(">>> Generating noise", flush=True)
    noise = generate_noise(noise_type, duration, FS)
    N = len(noise)

    print(">>> Adding sources and microphones", flush=True)
    room.add_source(noise_position, signal=noise)

    # Add silent speaker source so pyroomacoustics computes its RIR
    room.add_source(speaker_position, signal=np.zeros(N, dtype=np.float32))

    # mic array: [ref, err]
    mic_positions = np.c_[np.array(ref_mic_position), np.array(err_mic_position)]
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, FS))

    print(">>> Running room simulation", flush=True)
    room.simulate()

    ref = room.mic_array.signals[0].astype(np.float32)
    err_before = room.mic_array.signals[1].astype(np.float32)

    print(">>> Extracting secondary path h_s (speaker -> error mic)", flush=True)
    h_s = np.array(room.rir[1][1], dtype=np.float32)

    ref_n = _normalize(ref)
    err_n = _normalize(err_before)
    h_s_n = _normalize(h_s)

    if not anc_enabled:
        mse = float(np.mean(err_n ** 2))
        return {
            "signals": {
                "ref": ref_n,
                "err_before": err_n,
                "err_after": err_n,
                "control_u": None,
                "control_at_err": None,
            },
            "rir": {"h_s": h_s_n},
            "metrics": {
                "mse_before_full": mse,
                "mse_after_full": mse,
                "delta_db_full": 0.0,
            },
            "geometry": {
                "room_dims": tuple(room_dims),
                "material": material_preset,
                "noise_type": noise_type,
                "noise_position": tuple(noise_position),
                "ref_mic_position": tuple(ref_mic_position),
                "err_mic_position": tuple(err_mic_position),
                "speaker_position": tuple(speaker_position),
            },
        }

    print(">>> Building windows (stride)", flush=True)
    X = build_windows_strided(ref_n, SEQUENCE_LENGTH)
    num_windows = X.shape[0]
    print(f">>> Windows: {X.shape}", flush=True)

    print(">>> Running ANC inference in batches", flush=True)
    u_pred = np.zeros((num_windows,), dtype=np.float32)

    # batch inference loop (stable)
    for i in range(0, num_windows, predict_batch_size):
        xb = X[i : i + predict_batch_size]
        yb = model(tf.convert_to_tensor(xb), training=False)
        yb = np.asarray(yb).reshape(-1).astype(np.float32)
        u_pred[i : i + len(yb)] = yb

    print(f">>> Predicted u length: {len(u_pred)}", flush=True)

    # Align to full time length
    u_full = np.zeros(N, dtype=np.float32)
    start = SEQUENCE_LENGTH
    end = min(start + len(u_pred), N)
    valid_len = end - start
    if valid_len > 0:
        u_full[start:end] = u_pred[:valid_len]

    print(">>> Applying secondary path (numpy convolve)", flush=True)
    y_ctrl = apply_secondary_path_time(u_full, h_s_n, N)

    print(">>> Computing residual + metrics", flush=True)
    err_after = (err_n + y_ctrl).astype(np.float32)

    mse_before = float(np.mean(err_n ** 2))
    mse_after = float(np.mean(err_after ** 2))
    delta_db = float(10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12)))

    print(">>> Simulation finished", flush=True)

    return {
        "signals": {
            "ref": ref_n,
            "err_before": err_n,
            "err_after": err_after,
            "control_u": u_full,
            "control_at_err": y_ctrl,
        },
        "rir": {"h_s": h_s_n},
        "metrics": {
            "mse_before_full": mse_before,
            "mse_after_full": mse_after,
            "delta_db_full": delta_db,
        },
        "geometry": {
            "room_dims": tuple(room_dims),
            "material": material_preset,
            "noise_type": noise_type,
            "noise_position": tuple(noise_position),
            "ref_mic_position": tuple(ref_mic_position),
            "err_mic_position": tuple(err_mic_position),
            "speaker_position": tuple(speaker_position),
        },
    }
