# phase2/testing/test_tcn_ref2ctrl.py

import os
import numpy as np
import librosa
import tensorflow as tf
from scipy.signal import fftconvolve
from scipy.io.wavfile import write as wav_write
import matplotlib.pyplot as plt

FS = 16000
SEQUENCE_LENGTH = 200
VAL_DIR = "data/val"
OUT_DIR = "phase2_outputs"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def evaluate_scenario(model, ref_path, err_path, rir_path, tag):
    # ----- load data -----
    ref, _ = librosa.load(ref_path, sr=FS, mono=True)
    err, _ = librosa.load(err_path, sr=FS, mono=True)
    h_s = np.load(rir_path)

    # normalize
    ref /= np.max(np.abs(ref)) + 1e-9
    err /= np.max(np.abs(err)) + 1e-9
    h_s /= np.max(np.abs(h_s)) + 1e-9

    # ----- predict control u[t] from reference -----
    model_input_len = len(ref) - SEQUENCE_LENGTH
    u_pred = []

    for t in range(SEQUENCE_LENGTH, len(ref)):
        window = ref[t-SEQUENCE_LENGTH:t].reshape(1, SEQUENCE_LENGTH, 1)
        u_val = model(window, training=False)[0, 0].numpy()
        u_pred.append(u_val)

    u_pred = np.array(u_pred)  # shape (T_pred,)

    # ----- propagate through secondary path -----
    # control contribution at ear (same length as u_pred)
    y_ctrl = fftconvolve(u_pred, h_s, mode="same")

    # ----- choose matching segment of error signal -----
    # we compare on the same time indices where u_pred is defined
    err_seg = err[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(y_ctrl)]
    if len(err_seg) < len(y_ctrl):
        # pad error if slightly too short
        err_seg = np.pad(err_seg, (0, len(y_ctrl) - len(err_seg)), mode="constant")
    elif len(err_seg) > len(y_ctrl):
        err_seg = err_seg[:len(y_ctrl)]

    combined = err_seg + y_ctrl

    mse_before = float(np.mean(err_seg**2))
    mse_after = float(np.mean(combined**2))
    delta_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

    print(f"[{tag}] MSE BEFORE: {mse_before:.6e}")
    print(f"[{tag}] MSE AFTER : {mse_after:.6e}")
    print(f"[{tag}] ΔE (dB)   : {delta_db:.2f}")
    print()

    # ----- save audio outputs -----
    scen_out_dir = os.path.join(OUT_DIR, tag)
    ensure_dir(scen_out_dir)

    # convert to int16
    def to_int16(x):
        x = x / (np.max(np.abs(x)) + 1e-9)
        return (x * 32767).astype(np.int16)

    wav_write(os.path.join(scen_out_dir, f"{tag}_err_eval.wav"),
              FS, to_int16(err_seg))
    wav_write(os.path.join(scen_out_dir, f"{tag}_ctrl_at_ear.wav"),
              FS, to_int16(y_ctrl))
    wav_write(os.path.join(scen_out_dir, f"{tag}_combined.wav"),
              FS, to_int16(combined))

    # ----- save a quick time-domain plot -----
    zoom_sec = 0.05  # 50 ms
    zoom_samples = min(int(zoom_sec * FS), len(err_seg))
    t = np.arange(zoom_samples) / FS

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title(f"{tag} – Error mic (zoom)")
    plt.plot(t, err_seg[:zoom_samples])
    plt.ylabel("Amp")

    plt.subplot(3, 1, 2)
    plt.title("Control contribution at ear (u * h_s, zoom)")
    plt.plot(t, y_ctrl[:zoom_samples])
    plt.ylabel("Amp")

    plt.subplot(3, 1, 3)
    plt.title("Combined (error + control, zoom)")
    plt.plot(t, combined[:zoom_samples])
    plt.ylabel("Amp")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(os.path.join(scen_out_dir, f"{tag}_time_zoom.png"))
    plt.close()

    return mse_before, mse_after, delta_db


def main():
    ensure_dir(OUT_DIR)

    # load model
    model_path = "phase2_model_ref2ctrl_gain_approx.keras"
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # find validation scenarios
    val_scenarios = []
    for fname in os.listdir(VAL_DIR):
        if fname.endswith("_ref.wav"):
            prefix = fname.replace("_ref.wav", "")
            full_prefix = os.path.join(VAL_DIR, prefix)
            val_scenarios.append(full_prefix)

    val_scenarios = sorted(set(val_scenarios))
    print(f"Found {len(val_scenarios)} validation scenarios.")
    print()

    all_deltas = []

    for scen_prefix in val_scenarios:
        tag = os.path.basename(scen_prefix)  # e.g. "val_scenario_000"
        ref_path = scen_prefix + "_ref.wav"
        err_path = scen_prefix + "_err.wav"
        rir_path = scen_prefix + "_rir_spk_to_err.npy"

        mse_before, mse_after, delta_db = evaluate_scenario(
            model, ref_path, err_path, rir_path, tag
        )

        all_deltas.append(delta_db)

    if all_deltas:
        all_deltas = np.array(all_deltas)
        print("=== Aggregate over validation scenarios ===")
        print(f"Average ΔE (dB): {np.mean(all_deltas):.2f}")
        print(f"Std ΔE (dB)    : {np.std(all_deltas):.2f}")


if __name__ == "__main__":
    main()
