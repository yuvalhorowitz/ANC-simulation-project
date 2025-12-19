import librosa
import numpy as np
import matplotlib.pyplot as plt

FS = 16000
BASE = "phase_2/testing/audio_outputs/val_scenario_000"

err_before, _ = librosa.load(BASE + "_err_before.wav", sr=FS)
err_after, _ = librosa.load(BASE + "_err_after.wav", sr=FS)

# Time axis (seconds)
t = np.arange(len(err_before)) / FS

# Zoom window (first 100 ms)
zoom = int(0.1 * FS)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:zoom], err_before[:zoom])
plt.title("Error mic BEFORE ANC")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t[:zoom], err_after[:zoom])
plt.title("Error mic AFTER ANC")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
