# visualize_anc_scenario_terminal.py
#
# Visualization + audio export tool for ANC TCN in a plain terminal / VSCode setup.
# - Loads trained model
# - Loads *_ref.wav and *_err.wav scenario
# - Computes predicted anti-noise and combined signal
# - Prints metrics
# - Saves:
#       ref.wav, err.wav, pred.wav, combined.wav
#   and a PNG plot with all time-domain signals.
#
# Run from terminal, e.g.:
#   python visualize_anc_scenario_terminal.py \
#       --model tcn_ref2err.keras \
#       --ref_wav data/val/val_scenario_000_ref.wav \
#       --out_dir outputs/val_000_vis

import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io.wavfile import write as wav_write


def load_signals(ref_wav_path, sample_rate=16000):
    """
    Load reference and matching error signal, normalize separately,
    and return (ref_sig, err_sig, sr).
    """
    err_wav_path = ref_wav_path.replace("_ref.wav", "_err.wav")
    if not os.path.exists(err_wav_path):
        raise FileNotFoundError(f"Matching *_err.wav not found for {ref_wav_path}")

    ref_sig, sr_ref = librosa.load(ref_wav_path, sr=sample_rate, mono=True)
    err_sig, sr_err = librosa.load(err_wav_path, sr=sample_rate, mono=True)

    if sr_ref != sample_rate or sr_err != sample_rate:
        print(f"[WARN] Sampling rates ({sr_ref}, {sr_err}) differ from {sample_rate}")

    n = min(len(ref_sig), len(err_sig))
    ref_sig = ref_sig[:n].astype(np.float32)
    err_sig = err_sig[:n].astype(np.float32)

    # normalize separately to [-1, 1]
    ref_sig /= np.max(np.abs(ref_sig) + 1e-9)
    err_sig /= np.max(np.abs(err_sig) + 1e-9)

    return ref_sig, err_sig, sample_rate

def plot_room_layout(source_loc, ref_loc, err_loc, room_dim, out_png):
    """
    Create a 2D top-down schematic of the room/cabin.
    source_loc, ref_loc, err_loc are 3-element lists [x, y, z].
    room_dim = [L, W, H]
    Saves a PNG to out_png.
    """

    L, W, H = room_dim

    plt.figure(figsize=(8, 6))
    plt.title("Room / Car Interior Layout (Top-Down View)")

    # Draw room rectangle
    plt.plot([0, L, L, 0, 0], [0, 0, W, W, 0], 'k-')
    
    # Reference mic
    plt.scatter(ref_loc[0], ref_loc[1], c='blue', s=120, label="Reference Mic")
    plt.text(ref_loc[0]+0.05, ref_loc[1]+0.05, "Ref Mic", color='blue')

    # Error mic (driver)
    plt.scatter(err_loc[0], err_loc[1], c='red', s=120, label="Error Mic (Driver)")
    plt.text(err_loc[0]+0.05, err_loc[1]+0.05, "Driver", color='red')

    # Source
    plt.scatter(source_loc[0], source_loc[1], c='green', s=120, label="Noise Source")
    plt.text(source_loc[0]+0.05, source_loc[1]+0.05, "Source", color='green')

    plt.xlabel("X position (meters)")
    plt.ylabel("Y position (meters)")
    plt.xlim(-0.2, L + 0.2)
    plt.ylim(-0.2, W + 0.2)
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Room layout plot saved: {out_png}")



def compute_prediction(model, ref_sig, sequence_length):
    """
    Given a trained model and a reference signal, build sliding windows
    and compute the predicted anti-noise sequence (padded to align in time).
    """
    n = len(ref_sig)
    if n <= sequence_length + 1:
        raise ValueError("Signal too short for the given sequence_length.")

    X = []
    for i in range(n - sequence_length - 1):
        X.append(ref_sig[i : i + sequence_length])

    X = np.array(X, dtype=np.float32).reshape(-1, sequence_length, 1)
    preds = model.predict(X, verbose=0).flatten()

    # Pad so pred[t] aligns with err_sig[t] index
    preds_padded = np.pad(
        preds,
        (sequence_length + 1, 0),
        mode="constant",
        constant_values=0.0,
    )
    preds_padded = preds_padded[:n]
    return preds_padded


def save_wav(path, signal, sr):
    """
    Save a float32 [-1,1] signal as 16-bit PCM WAV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sig_int16 = np.int16(np.clip(signal, -1.0, 1.0) * 32767)
    wav_write(path, sr, sig_int16)
    print(f"[INFO] WAV saved: {path}")


def plot_time_domain(ref_sig, err_sig, pred_sig, combined_sig, sr, zoom_duration, out_png):
    """
    Time-domain plots (zoomed) of ref, error, predicted, and combined.
    Saves a PNG to out_png.
    """
    n_zoom = min(int(sr * zoom_duration), len(err_sig))
    t_axis = np.arange(n_zoom) / sr

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.title("Reference mic signal (zoomed)")
    plt.plot(t_axis, ref_sig[:n_zoom])
    plt.ylabel("Amp")

    plt.subplot(4, 1, 2)
    plt.title("Error mic signal (zoomed)")
    plt.plot(t_axis, err_sig[:n_zoom])
    plt.ylabel("Amp")

    plt.subplot(4, 1, 3)
    plt.title("Predicted anti-noise (zoomed)")
    plt.plot(t_axis, pred_sig[:n_zoom])
    plt.ylabel("Amp")

    plt.subplot(4, 1, 4)
    plt.title("Combined at error mic (err + pred, zoomed)")
    plt.plot(t_axis, combined_sig[:n_zoom])
    plt.ylabel("Amp")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved: {out_png}")


def visualize_scenario_terminal(
    model_path,
    ref_wav_path,
    out_dir,
    sample_rate=16000,
    sequence_length=200,
    zoom_duration=0.05,
):
    """
    High-level helper for terminal / VSCode:
      - load trained model
      - load ref + err
      - compute predicted anti-noise
      - compute combined signal
      - print metrics
      - save WAVs and a PNG plot
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load model
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2) Load signals
    print(f"[INFO] Loading scenario from: {ref_wav_path}")
    ref_sig, err_sig, sr = load_signals(ref_wav_path, sample_rate=sample_rate)

    # 3) Compute prediction
    print("[INFO] Computing predicted anti-noise...")
    pred_sig = compute_prediction(model, ref_sig, sequence_length)

    # 4) Combine at error mic
    combined_sig = err_sig + pred_sig

    # 5) Print simple metrics
    mse_before = float(np.mean(err_sig**2))
    mse_after = float(np.mean(combined_sig**2))
    improvement_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

    print(f"[RESULT] MSE at error mic BEFORE: {mse_before:.6e}")
    print(f"[RESULT] MSE at error mic AFTER:  {mse_after:.6e}")
    print(f"[RESULT] Energy reduction:        {improvement_db:.2f} dB")

        # Save room visualization
    out_png_layout = os.path.join(out_dir, "room_layout.png")

    # For consistency we assume these same positions used in simulation_setup.py
    room_dim = [4, 3, 2]  # L, W, H
    source_loc = [1, 0.5, 1]
    ref_loc = [1.3, 0.7, 1.0]
    err_loc = [3.0, 2.5, 1.0]

    plot_room_layout(source_loc, ref_loc, err_loc, room_dim, out_png_layout)


    # 6) Save WAVs for listening
    save_wav(os.path.join(out_dir, "ref.wav"), ref_sig, sr)
    save_wav(os.path.join(out_dir, "err.wav"), err_sig, sr)
    save_wav(os.path.join(out_dir, "pred.wav"), pred_sig, sr)
    save_wav(os.path.join(out_dir, "combined.wav"), combined_sig, sr)

    # 7) Save time-domain plot
    out_png = os.path.join(out_dir, "time_domain_zoom.png")
    plot_time_domain(ref_sig, err_sig, pred_sig, combined_sig, sr, zoom_duration, out_png)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ANC scenario in terminal (TCN ref->err)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained Keras model (.keras).",
    )
    parser.add_argument(
        "--ref_wav",
        type=str,
        required=True,
        help="Path to *_ref.wav file for the scenario.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="anc_vis_output",
        help="Output directory for WAVs and plots.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate expected by the model.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=200,
        help="Sequence length used during training.",
    )
    parser.add_argument(
        "--zoom_duration",
        type=float,
        default=0.05,
        help="Duration (s) of zoom window for time-domain plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    visualize_scenario_terminal(
        model_path=args.model,
        ref_wav_path=args.ref_wav,
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
        sequence_length=args.sequence_length,
        zoom_duration=args.zoom_duration,
    )

    
