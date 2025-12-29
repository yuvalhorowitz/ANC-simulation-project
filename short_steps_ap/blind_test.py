import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- Configuration ---
DATA_DIR = "fast_data_8k"
TEST_SCENARIO = 2  # Blind Test
MODEL_PATH = 'anc_tcn_v4_multi.keras'
FS = 8000
WINDOW_SIZE = 512

def load_and_normalize_test(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    ref /= np.max(np.abs(ref))
    mic /= np.max(np.abs(mic))
    return ref, mic, hs

if __name__ == "__main__":
    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Loaded model from {MODEL_PATH}")

    # 2. Prepare Test Data
    ref, mic, hs = load_and_normalize_test(TEST_SCENARIO)
    filtered_ref = convolve(ref, hs, mode='same')
    X_test, y_test = [], []
    for i in range(len(filtered_ref) - WINDOW_SIZE):
        X_test.append(filtered_ref[i:i+WINDOW_SIZE])
        y_test.append(mic[i+WINDOW_SIZE])
    X_test, y_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1), np.array(y_test)

    # 3. Inference
    u_pred = model.predict(X_test).flatten()
    residual = y_test + u_pred
    reduction = 10 * np.log10(np.mean(y_test**2) / np.mean(residual**2))

    # 4. Comprehensive Plotting (Detailed like v3)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(y_test[2000:2500], label="Original Noise", alpha=0.5)
    plt.plot(residual[2000:2500], label="Residual (After ANC)", color='green')
    plt.title(f"Blind Test on Scenario {TEST_SCENARIO} (Reduction: {reduction:.2f} dB)")
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    f, psd_orig = welch(y_test, FS, nperseg=1024)
    _, psd_resid = welch(residual, FS, nperseg=1024)
    plt.plot(f, 10*np.log10(psd_orig+1e-10), label="Original Spectrum")
    plt.plot(f, 10*np.log10(psd_resid+1e-10), label="Residual Spectrum", color='green')
    plt.xlim(0, 500); plt.title("PSD Analysis (dB/Hz)"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"blind_test_scn{TEST_SCENARIO}_results.png")
    print(f"✓ Blind test complete. Reduction: {reduction:.2f} dB. Plot saved.")