"""
v7 Expert Multi-Band Blind Test
Evaluates the 5-branch parallel TCN on Scenarios 45-49.
Locked to Scenario 45 for consistent cross-version comparison.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. Global Config ---
DATA_DIR = "driver_bulk_4spk_data"
TEST_SCENARIOS = range(45, 50) 
MODEL_PATH = 'anc_v7_multi_band_expert.keras'
FS = 8000
WINDOW_SIZE = 512

# --- 2. Custom Loss Registration (Required for Loading) ---
@tf.keras.utils.register_keras_serializable()
class MultiBandExpertLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs, self.window_size = fs, window_size
        self.bins = fs / window_size
        self.bands = [(50, 350, 10.0), (350, 800, 25.0), (800, 1200, 50.0), 
                      (1200, 1600, 80.0), (1600, 2000, 120.0)]

    def call(self, y_true, y_pred):
        error_fft = tf.abs(tf.signal.rfft(y_true[:, :, 0] - y_pred[:, :, 0]))
        total_loss = 0.0
        for low, high, weight in self.bands:
            l_idx, h_idx = int(low/self.bins), int(high/self.bins)
            total_loss += tf.reduce_mean(tf.square(error_fft[:, l_idx:h_idx])) * weight
        return total_loss

def run_v7_blind_test():
    print(f"--- Loading v7 Multi-Band Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'MultiBandExpertLoss': MultiBandExpertLoss})

    scenario_results = []
    
    for scn_id in TEST_SCENARIOS:
        prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")

        # Standard Pre-processing (Locked Scaling)
        f_ref = convolve(ref, hs, mode='same')
        f_ref /= (np.max(np.abs(f_ref)) + 1e-7)
        mic /= (np.max(np.abs(mic)) + 1e-7); mic *= 0.8 

        # Inference
        X_test = [f_ref[i:i+WINDOW_SIZE] for i in range(len(f_ref) - WINDOW_SIZE)]
        X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
        u_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        # Superposition
        rec, gen = mic[WINDOW_SIZE:], u_pred[:, -1, 0]
        res = rec + gen
        
        # Band Analysis
        def get_band_db(orig, resid, f_range):
            f, p1 = welch(orig, FS, nperseg=1024)
            _, p2 = welch(resid, FS, nperseg=1024)
            mask = (f >= f_range[0]) & (f <= f_range[1])
            return 10 * np.log10(np.sum(p1[mask]) / np.sum(p2[mask]))

        # Calculate metrics for all 5 requested bands
        b1 = get_band_db(rec, res, [50, 350])
        b2 = get_band_db(rec, res, [350, 800])
        b3 = get_band_db(rec, res, [800, 1200])
        b4 = get_band_db(rec, res, [1200, 1600])
        b5 = get_band_db(rec, res, [1600, 2000])
        total = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
        
        scenario_results.append({
            'rec': rec, 'res': res, 'total': total, 
            'bands': [b1, b2, b3, b4, b5], 'id': scn_id
        })
        print(f"Scenario {scn_id}: Total={total:.1f}dB | High-Band(1.6-2k)={b5:.1f}dB")

    # --- 3. Comparison Visualization (Locked to Scenario 45) ---
    # We find Scenario 45 in our results to ensure consistent "Blue Line"
    target_scn = [r for r in scenario_results if r['id'] == 45][0]
    rec, res = target_scn['rec'], target_scn['res']

    fig, axs = plt.subplots(2, 1, figsize=(15, 12))

    # PSD with 5-Band Highlights
    f, p_orig = welch(rec, FS, nperseg=1024)
    _, p_resid = welch(res, FS, nperseg=1024)
    
    axs[0].plot(f, 10*np.log10(p_orig+1e-12), label="Original (Scenario 45)", alpha=0.4, color='blue')
    axs[0].plot(f, 10*np.log10(p_resid+1e-12), label="v7 Residual", color='green', lw=2)
    
    # Highlight the 5 bands
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    bands = [(50, 350), (350, 800), (800, 1200), (1200, 1600), (1600, 2000)]
    for (l, h), c, val in zip(bands, colors, target_scn['bands']):
        axs[0].axvspan(l, h, color=c, alpha=0.1, label=f"{l}-{h}Hz ({val:.1f}dB)")
    
    axs[0].set_title("v7 Expert Multi-Band Blind Test: Spectral Suppression")
    axs[0].set_xlim(0, 2200); axs[0].legend(loc='upper right'); axs[0].grid(True)

    # Aggregate Bar Chart
    all_totals = [r['total'] for r in scenario_results]
    axs[1].bar([f"Scn {r['id']}" for r in scenario_results], all_totals, color='teal')
    axs[1].axhline(y=np.mean(all_totals), color='red', linestyle='--', label=f"Avg: {np.mean(all_totals):.2f} dB")
    axs[1].set_title("Spatial Robustness (dB Reduction across Head Positions)")
    axs[1].set_ylabel("dB Reduction"); axs[1].legend()

    plt.tight_layout()
    plt.savefig("v7_blind_test_report.png")
    print(f"\nReport saved for Scenario 45 comparison.")

if __name__ == "__main__":
    run_v7_blind_test()