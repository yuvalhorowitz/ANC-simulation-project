# compute_ideal_control_single.py
#
# For a single scenario:
#   - load error mic signal (e)
#   - load secondary-path RIR h_s (speaker -> error mic)
#   - compute an "ideal" control signal u such that
#         (u * h_s) ≈ -e
#     using frequency-domain deconvolution with regularization
#   - check cancellation quality
#   - save u and the combined signal to WAVs.

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write
from scipy.signal import fftconvolve


FS = 16000  # must match your simulation


def save_wav(path, signal, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sig = signal / (np.max(np.abs(signal) + 1e-9))
    sig_int16 = np.int16(np.clip(sig, -1.0, 1.0) * 32767)
    wav_write(path, sr, sig_int16)
    print(f"[INFO] WAV saved: {path}")


def compute_ideal_control_ls(err_sig, h_s, filt_len=256, reg=1e-2):
    """
    Compute a *finite-length* FIR control filter w (length filt_len) such that
        y_ctrl = H_s w ≈ -err_sig
    in least-squares sense:

        min_w ||H w + err||^2 + reg * ||w||^2

    err_sig : (N,)
    h_s     : (M,)
    filt_len: length of control filter w
    reg     : Tikhonov regularization lambda
    """
    e = err_sig.astype(np.float64)
    h = h_s.astype(np.float64)

    N = len(e)
    M = len(h)

    # Build convolution matrix H of shape (N, filt_len)
    # Column k corresponds to impulse at time k filtered by h_s
    H = np.zeros((N, filt_len), dtype=np.float64)

    for k in range(filt_len):
        start = k
        end = min(N, k + M)
        if start >= N:
            break
        H[start:end, k] = h[: end - start]

    # Solve (H^T H + reg I) w = -H^T e
    HT = H.T
    A = HT @ H + reg * np.eye(filt_len)
    b = -HT @ e

    w = np.linalg.solve(A, b)  # (filt_len,)

    return w


def main():
    # ---- choose scenario prefix here ----
    scenario_prefix = "data/train/train_scenario_000"

    err_path = scenario_prefix + "_err.wav"
    rir_spk_err_path = scenario_prefix + "_rir_spk_to_err.npy"

    if not os.path.exists(err_path):
        raise FileNotFoundError(f"Error WAV not found: {err_path}")
    if not os.path.exists(rir_spk_err_path):
        raise FileNotFoundError(f"RIR spk->err not found: {rir_spk_err_path}")

    print(f"[INFO] Loading error signal from: {err_path}")
    err_sig, sr = librosa.load(err_path, sr=FS, mono=True)

    print(f"[INFO] Loading secondary-path RIR from: {rir_spk_err_path}")
    h_s = np.load(rir_spk_err_path)

    # Normalize error and RIR for stability
    err_sig = err_sig.astype(np.float32)
    err_sig /= (np.max(np.abs(err_sig)) + 1e-9)

    h_s = h_s.astype(np.float32)
    h_s /= (np.max(np.abs(h_s)) + 1e-9)

        # ---- compute ideal *filter* and control signal ----
    print("[INFO] Computing ideal FIR control filter...")
    filt_len = 256  # you can try 128 / 256 / 512
    w = compute_ideal_control_ls(err_sig, h_s, filt_len=filt_len, reg=1e-2)

    # Control signal u is just the filter w convolved with a unit impulse at t=0,
    # but for listening/visualization it's nicer to view the *output at ear*:
    # y_ctrl = H_s w (already computed via convolution matrix, but we'll recompute via conv)

    print("[INFO] Generating control contribution at ear...")
    y_full = fftconvolve(w, h_s, mode="full")

    # Choose the first N samples as the main response (assuming causal system)
    N = len(err_sig)
    if len(y_full) >= N:
        y_ctrl = y_full[:N]
    else:
        y_ctrl = np.pad(y_full, (0, N - len(y_full)), mode="constant")

    combined = err_sig + y_ctrl

    mse_before = float(np.mean(err_sig**2))
    mse_after = float(np.mean(combined**2))
    delta_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

    print(f"[RESULT] MSE BEFORE (error mic): {mse_before:.6e}")
    print(f"[RESULT] MSE AFTER  (combined):  {mse_after:.6e}")
    print(f"[RESULT] Energy reduction:       {delta_db:.2f} dB")


    mse_before = float(np.mean(err_sig**2))
    mse_after = float(np.mean(combined**2))
    delta_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

    print(f"[RESULT] MSE BEFORE (error mic): {mse_before:.6e}")
    print(f"[RESULT] MSE AFTER  (combined):  {mse_after:.6e}")
    print(f"[RESULT] Energy reduction:       {delta_db:.2f} dB")

    print(f"[RESULT] MSE BEFORE (error mic): {mse_before:.6e}")
    print(f"[RESULT] MSE AFTER  (combined):  {mse_after:.6e}")
    print(f"[RESULT] Energy reduction:       {delta_db:.2f} dB")

    # ---- save signals ----
    # ---- save signals ----
    out_dir = "ideal_control_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # save error for comparison
    save_wav(os.path.join(out_dir, "err.wav"), err_sig, sr)

    # save effect at ear
    save_wav(os.path.join(out_dir, "ctrl_at_ear.wav"), y_ctrl, sr)

    # save combined residual
    save_wav(os.path.join(out_dir, "combined.wav"), combined, sr)

    # save filter coefficients (not audio!)
    np.save(os.path.join(out_dir, "w_control_filter.npy"), w)
    print(f"[INFO] Saved control filter to w_control_filter.npy")

    # ---- small plot for sanity ----
    zoom = int(0.05 * sr)  # 50 ms
    t = np.arange(zoom) / sr

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.title("Error mic signal (zoom)")
    plt.plot(t, err_sig[:zoom])

    plt.subplot(3, 1, 2)
    plt.title("Control contribution at ear (H_s w, zoom)")
    plt.plot(t, y_ctrl[:zoom])

    plt.subplot(3, 1, 3)
    plt.title("Combined (error + control at ear, zoom)")
    plt.plot(t, combined[:zoom])
    plt.xlabel("Time [s]")



if __name__ == "__main__":
    main()
