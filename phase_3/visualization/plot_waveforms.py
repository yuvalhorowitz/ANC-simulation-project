# phase_3/visualization/plot_waveforms.py

import os
import numpy as np
import matplotlib.pyplot as plt

from phase_3.utils.audio_io import save_wav_mono


def plot_waveforms(result: dict, fs: int = 16000, zoom_s=(0.5, 0.7), save_audio_dir="phase_3/results/audio"):
    os.makedirs(save_audio_dir, exist_ok=True)

    sig = result["signals"]
    ref = sig["ref"]
    err_b = sig["err_before"]
    err_a = sig["err_after"]
    u = sig["control_u"]
    y = sig["control_at_err"]

    # Save WAVs
    save_wav_mono(os.path.join(save_audio_dir, "ref.wav"), ref, fs)
    save_wav_mono(os.path.join(save_audio_dir, "err_before.wav"), err_b, fs)
    save_wav_mono(os.path.join(save_audio_dir, "err_after.wav"), err_a, fs)
    save_wav_mono(os.path.join(save_audio_dir, "u.wav"), u, fs)
    save_wav_mono(os.path.join(save_audio_dir, "y_ctrl.wav"), y, fs)

    t = np.arange(len(ref)) / fs
    a = int(zoom_s[0] * fs)
    b = int(zoom_s[1] * fs)

    plt.figure()
    plt.title("Error mic: BEFORE")
    plt.plot(t[a:b], err_b[a:b])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure()
    plt.title("Error mic: AFTER")
    plt.plot(t[a:b], err_a[a:b])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure()
    plt.title("Control signal u (speaker drive)")
    plt.plot(t[a:b], u[a:b])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    plt.figure()
    plt.title("Control contribution at error mic (y_ctrl)")
    plt.plot(t[a:b], y[a:b])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    print(f"Saved WAVs to: {save_audio_dir}")
