import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Register the Custom Loss for Loading ---
@tf.keras.utils.register_keras_serializable()
class SurgicalSparsityLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        true_fft = tf.abs(tf.signal.rfft(y_true[:, :, 0]))
        pred_fft = tf.abs(tf.signal.rfft(y_pred[:, :, 0]))
        silence_mask = tf.cast(true_fft < 1e-3, tf.float32)
        phantom_penalty = tf.reduce_mean(tf.square(pred_fft) * silence_mask) * 500.0
        return mse_loss + phantom_penalty

# --- 2. Blind Test Execution ---
def run_v8_cpu_blind_test():
    MODEL_PATH = 'anc_v8_surgical_gpu.keras' # Ensure this matches your filename
    DATA_DIR = "driver_bulk_4spk_data"
    TEST_SCENARIOS = range(45, 50)
    FS, WINDOW_SIZE = 8000, 512

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Check your file name.")
        return

    print(f"--- Loading v8 Model (CPU Mode): {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'SurgicalSparsityLoss': SurgicalSparsityLoss})

    scenario_results = []

    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref = np.load(f"{prefix}_ref.npy")
        mic = np.load(f"{prefix}_mic.npy")
        hs = np.load(f"{prefix}_hs.npy")

        # Pre-processing
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Sliding window inference
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
        
        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        # Reconstruction
        rec = mic[WINDOW_SIZE:]
        gen = u_pred[:, -1, 0]
        res = rec + gen
        
        # Performance metrics
        def get_band_db(orig, resid, f_range):
            f_axis, p1 = welch(orig, FS, nperseg=1024)
            _, p2 = welch(resid, FS, nperseg=1024)
            mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            return 10 * np.log10(np.sum(p1[mask]) / np.sum(p2[mask]))

        db_low = get_band_db(rec, res, [50, 800])
        db_high = get_band_db(rec, res, [1600, 2000])
        total = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
        
        scenario_results.append({'rec': rec, 'res': res, 'total': total, 'high': db_high, 'id': scn_id})
        print(f"Scenario {scn_id}: Total Red.={total:.1f}dB | 1.6k-2k Zone={db_high:.2f}dB")

    # --- 3. Plotting Results (Scenario 45) ---
    target = [r for r in scenario_results if r['id'] == 45][0]
    f, p_orig = welch(target['rec'], FS, nperseg=1024)
    _, p_resid = welch(target['res'], FS, nperseg=1024)

    plt.figure(figsize=(15, 7))
    plt.plot(f, 10*np.log10(p_orig+1e-12), label="Original Noise", alpha=0.4, color='blue')
    plt.plot(f, 10*np.log10(p_resid+1e-12), label="v8 Surgical Residual", color='green', lw=2)
    plt.axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Suppression Zone')
    
    plt.title(f"v8 CPU Blind Test: Scenario 45\nHigh-Freq Improvement: {target['high']:.2f} dB")
    plt.xlim(0, 2200); plt.xlabel("Frequency (Hz)"); plt.ylabel("Power (dB)"); plt.legend(); plt.grid(True)
    
    plt.savefig("v8_cpu_blind_report.png")
    print("\nBlind Test Complete. Report saved as v8_cpu_blind_report.png")

if __name__ == "__main__":
    run_v8_cpu_blind_test()