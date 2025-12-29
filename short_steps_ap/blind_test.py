import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve, welch

# --- 1. הגדרות וקונפיגורציה ---
DATA_DIR = "car_anc_final_data"
TEST_SCENARIO = 2  # התרחיש שלא נכלל באימון
MODEL_PATH = 'anc_car_tcn_model_v2912.keras' # וודא שזה השם שבו שמרת את המודל
FS = 8000
WINDOW_SIZE = 512

# --- 2. הגדרת ה-Loss לצורך טעינת המודל ---
# חובה להגדיר את המחלקה כדי ש-Keras ידע איך לקרוא את המודל השמור
@tf.keras.utils.register_keras_serializable()
class HarmonicTargetedLoss(tf.keras.losses.Loss):
    def __init__(self, fs=8000, window_size=512, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs
        self.window_size = window_size
        weights = np.ones(window_size // 2 + 1, dtype=np.float32)
        low_bin = int(100 / (fs / window_size))
        high_bin = int(300 / (fs / window_size))
        weights[low_bin:high_bin] = 10.0 
        weights[int(350 / (fs / window_size)):] = 0.05
        self.weights = tf.constant(weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        error_fft = tf.signal.rfft(error[:, :, 0])
        weighted_error = tf.square(tf.abs(error_fft)) * self.weights
        return tf.reduce_mean(weighted_error)

# --- 3. פונקציית טעינה עם נרמול ל-0.8 ---
def load_and_normalize_test(scn_id):
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    
    # נרמול זהה לאימון
    ref /= (np.max(np.abs(ref)) + 1e-7)
    mic /= (np.max(np.abs(mic)) + 1e-7)
    mic *= 0.8 # חשוב מאוד! שמירה על הטווח הליניארי של ה-tanh
    
    return ref, mic, hs

if __name__ == "__main__":
    # א. טעינת המודל עם האובייקטים המיוחדים
    print(f"--- Loading Model: {MODEL_PATH} ---")
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'HarmonicTargetedLoss': HarmonicTargetedLoss})
    
    # ב. הכנת נתוני הטסט
    print(f"--- Preparing Data for Blind Scenario {TEST_SCENARIO} ---")
    ref, mic, hs = load_and_normalize_test(TEST_SCENARIO)
    filtered_ref = convolve(ref, hs, mode='same')
    
    X_test, y_test_windows = [], []
    for i in range(len(filtered_ref) - WINDOW_SIZE):
        X_test.append(filtered_ref[i:i+WINDOW_SIZE])
        y_test_windows.append(mic[i:i+WINDOW_SIZE])
        
    X_test = np.array(X_test).reshape(-1, WINDOW_SIZE, 1)
    y_test_windows = np.array(y_test_windows).reshape(-1, WINDOW_SIZE, 1)

    # ג. הרצת החיזוי (Inference)
    print("--- Running Inference ---")
    u_pred_full = model.predict(X_test, batch_size=64)
    
    # ד. חילוץ האותות (לקיחת הנקודה האחרונה מכל חלון לסימולציה של זמן אמת)
    sig_recorded = y_test_windows[:, -1, 0]   # הרעש המקורי באוזן
    sig_generated = u_pred_full[:, -1, 0]    # האנטי-רעש שהמודל ייצר
    sig_combined = sig_recorded + sig_generated # השארית (מה ששומעים)

    # ה. חישוב ביצועים ב-dB
    reduction = 10 * np.log10(np.mean(sig_recorded**2) / np.mean(sig_combined**2))
    print(f"\n" + "="*30)
    print(f" FINAL BLIND TEST RESULT: {reduction:.2f} dB")
    print("="*30)

    # ו. ויזואליזציה מלאה
    plt.figure(figsize=(15, 14))

    # גרף 1: ציר הזמן - אינטראקציית האותות
    plt.subplot(3, 1, 1)
    start, end = 2000, 2600 # זום על קטע קצר
    plt.plot(sig_recorded[start:end], label="Original Noise", color='blue', alpha=0.5)
    plt.plot(sig_generated[start:end], label="Anti-Noise (Model Output)", color='orange', linestyle='--')
    plt.plot(sig_combined[start:end], label="Residual Error", color='green', linewidth=2)
    plt.title(f"Time Domain: Blind Test Performance (NR: {reduction:.2f} dB)")
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)

    # גרף 2: ציר התדר - בדיקת הרמוניות ו-Waterbed Effect
    plt.subplot(3, 1, 2)
    f, psd_orig = welch(sig_recorded, FS, nperseg=1024)
    _, psd_resid = welch(sig_combined, FS, nperseg=1024)
    plt.plot(f, 10 * np.log10(psd_orig + 1e-12), label="Original Spectrum", color='blue', alpha=0.6)
    plt.plot(f, 10 * np.log10(psd_resid + 1e-12), label="Residual Spectrum", color='green')
    plt.title("Frequency Domain: PSD Analysis (Target Harmonics: 100-300Hz)"); plt.xlim(0, 500)
    plt.ylabel("dB/Hz"); plt.xlabel("Frequency [Hz]"); plt.legend(); plt.grid(True, alpha=0.3)

    # גרף 3: שגיאה מוחלטת לאורך זמן
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(sig_combined), color='red', alpha=0.4, label="Instantaneous Error")
    plt.title("Error Magnitude Over Time"); plt.ylabel("Amplitude"); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("blind_test_final_report.png")
    plt.show()