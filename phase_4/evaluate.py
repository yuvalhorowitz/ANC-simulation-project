import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import config
import scipy.signal as signal
import matplotlib.pyplot as plt

# --- FIX: Import the module where the custom function is defined ---
import tcn_constrained 

def calculate_db_reduction(original, residual):
    """Calculates overall Decibel (dB) reduction."""
    rms_orig = np.sqrt(np.mean(original**2))
    rms_resid = np.sqrt(np.mean(residual**2))
    if rms_resid < 1e-9: rms_resid = 1e-9
    return 20 * np.log10(rms_orig / rms_resid)

def plot_analysis(d_actual, u_control, residual, sr, save_dir):
    """Generates detailed engineering plots."""
    t = np.arange(len(d_actual)) / sr

    # --- FIGURE 1: TIME DOMAIN ---
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, d_actual, color='black', alpha=0.7)
    plt.title("1. Original Noise at Ear (Target)", fontsize=12)
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, u_control, color='red', alpha=0.8)
    plt.title("2. Generated Anti-Noise (Control Signal)", fontsize=12)
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    reduction = calculate_db_reduction(d_actual, residual)
    plt.plot(t, residual, color='blue', alpha=0.8)
    plt.title(f"3. Residual Noise (Result) - Reduction: {reduction:.2f} dB", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_time_domain_comparison.png"), dpi=300)
    plt.close()

    # --- FIGURE 2: PSD ---
    f, Pxx_orig = signal.welch(d_actual, fs=sr, nperseg=1024)
    _, Pxx_resid = signal.welch(residual, fs=sr, nperseg=1024)
    
    Pxx_orig_db = 10 * np.log10(Pxx_orig + 1e-12)
    Pxx_resid_db = 10 * np.log10(Pxx_resid + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.semilogx(f, Pxx_orig_db, label='Original Noise', color='black')
    plt.semilogx(f, Pxx_resid_db, label='Residual (ANC On)', color='blue')
    plt.axvspan(50, 400, color='yellow', alpha=0.2, label='Engine Band')
    
    plt.title("ANC Performance (Power Spectral Density)", fontsize=14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB/Hz)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)
    plt.xlim(20, sr/2)
    
    plt.savefig(os.path.join(save_dir, "02_frequency_performance.png"), dpi=300)
    plt.close()

    # --- FIGURE 3: SPECTROGRAM ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.specgram(d_actual, Fs=sr, NFFT=512, noverlap=256, cmap='inferno')
    plt.title("Before ANC")
    plt.ylabel("Hz")
    
    plt.subplot(1, 2, 2)
    plt.specgram(residual, Fs=sr, NFFT=512, noverlap=256, cmap='inferno')
    plt.title("After ANC")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_spectrogram_comparison.png"), dpi=300)
    plt.close()

def main():
    print("--- EVALUATING MODEL PERFORMANCE ---")
    
    # 1. Load Model with Custom Object
    model_path = os.path.join(config.MODEL_SAVE_DIR, "tcn_phase4_final.h5")
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        return
        
    print(f"Loading model: {model_path}")
    
    # --- CRITICAL FIX: Tell load_model about the custom function ---
    model = tf.keras.models.load_model(
        model_path, 
        compile=False,
        custom_objects={'get_last_step': tcn_constrained.get_last_step}
    )

    # 2. Generate Test Signals
    print("Generating Test Engine Noise...")
    duration = 5 
    t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
    
    # Test on 820 RPM
    f0 = (820 / 60) * 2 
    raw_test_noise = 0.6 * np.sin(2 * np.pi * f0 * t) + \
                     0.3 * np.sin(2 * np.pi * 2*f0 * t) + \
                     0.15 * np.sin(2 * np.pi * 3.5*f0 * t) + \
                     0.05 * np.random.normal(0, 0.1, len(t))
    raw_test_noise = raw_test_noise / np.max(np.abs(raw_test_noise))

    # Simulate Physics (Primary Path) -> Noise at Ear
    primary_ir, _ = sf.read(config.PRIMARY_PATH_FILE)
    d_target = signal.fftconvolve(raw_test_noise, primary_ir, mode='same')
    
    # 3. Prepare Input
    X_test = []
    d_actual = [] 
    
    for i in range(config.WINDOW_SIZE, len(raw_test_noise)):
        window = raw_test_noise[i-config.WINDOW_SIZE : i]
        X_test.append(window.reshape(-1, 1))
        d_actual.append(d_target[i])
        
    X_test = np.array(X_test)
    d_actual = np.array(d_actual)
    
    # 4. Inference
    print("Running Inference...")
    u_control = model.predict(X_test, batch_size=32, verbose=1)
    u_control = u_control.flatten()
    
    # 5. Cancellation
    min_len = min(len(d_actual), len(u_control))
    d_actual = d_actual[:min_len]
    u_control = u_control[:min_len]
    raw_source_trimmed = raw_test_noise[config.WINDOW_SIZE : config.WINDOW_SIZE + min_len]
    
    residual_error = d_actual + u_control
    
    # 6. Metrics & Saving
    reduction = calculate_db_reduction(d_actual, residual_error)
    print(f"\n--------------------------------")
    print(f"NOISE REDUCTION ACHIEVED: {reduction:.2f} dB")
    print(f"--------------------------------")
    
    audio_dir = os.path.join(config.BASE_DIR, "results_audio")
    plot_dir = os.path.join(config.BASE_DIR, "results_plots")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- SAVING AUDIO ---
    sf.write(os.path.join(audio_dir, "00_source_noise.wav"), raw_source_trimmed, config.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, "01_original_noise.wav"), d_actual, config.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, "02_anti_noise_control.wav"), u_control, config.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, "03_result_cancelled.wav"), residual_error, config.SAMPLE_RATE)
    
    print("Generating Engineering Plots...")
    plot_analysis(d_actual, u_control, residual_error, config.SAMPLE_RATE, plot_dir)
    
    print(f"Audio files saved to: {audio_dir}")

if __name__ == "__main__":
    main()