# phase2/testing/test_tcn_ref2ctrl.py

import numpy as np
import librosa
import tensorflow as tf
from scipy.signal import fftconvolve

FS = 16000
SEQUENCE_LENGTH = 200


def test_scenario(model_path, scenario_prefix):

    ref, _ = librosa.load(scenario_prefix + "_ref.wav", sr=FS)
    err, _ = librosa.load(scenario_prefix + "_err.wav", sr=FS)
    h_s = np.load(scenario_prefix + "_rir_spk_to_err.npy")

    ref /= np.max(np.abs(ref)) + 1e-9
    err /= np.max(np.abs(err)) + 1e-9
    h_s /= np.max(np.abs(h_s)) + 1e-9

    model = tf.keras.models.load_model(model_path)

    u_pred = []
    for t in range(SEQUENCE_LENGTH, len(ref)):
        win = ref[t-SEQUENCE_LENGTH:t].reshape(1, SEQUENCE_LENGTH, 1)
        u_pred.append(model(win)[0, 0].numpy())

    u_pred = np.array(u_pred)

    # control contribution through secondary path
    y_ctrl = fftconvolve(u_pred, h_s, mode='same')

    err_eval = err[-len(y_ctrl):]
    combined = err_eval + y_ctrl

    mse_before = np.mean(err_eval**2)
    mse_after = np.mean(combined**2)
    delta_db = 10 * np.log10(mse_before / mse_after)

    print("=== Test Scenario Results ===")
    print(f"MSE BEFORE: {mse_before:.6e}")
    print(f"MSE AFTER : {mse_after:.6e}")
    print(f"ΔE (dB)   : {delta_db:.2f}")

    return u_pred, y_ctrl, combined


if __name__ == "__main__":
    test_scenario("phase2/training/phase2_model_ref2ctrl_gain_approx.keras",
                  "data/val/val_scenario_000")
