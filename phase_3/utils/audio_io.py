# phase_3/utils/audio_io.py

import os
import wave
import numpy as np


def save_wav_mono(path: str, x: np.ndarray, fs: int = 16000):
    """
    Save a mono float32 signal in [-1,1] to 16-bit PCM WAV using stdlib wave.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(x_i16.tobytes())
