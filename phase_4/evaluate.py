import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import config
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa

# Import your custom modules
import tcn_constrained
import loss_functions

def extract_features_for_eval(window, sr):
    """Matches the feature extraction used in generate_dataset.py[cite: 30, 43]."""
    pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
    pitch_idx = magnitudes.argmax()
    pitch_val = pitches.flatten()[pitch_idx] if np.any(pitches) else 0.0
    envelope = np.sqrt(np.mean(window**2))
    return [pitch_val, envelope]

def calculate_db_reduction(original, residual):
    """Calculates Decibel reduction based on RMS[cite: 69, 547]."""
    rms_orig = np.sqrt(np.mean(original**2))
    rms_resid = np.sqrt(np.mean(residual**2))
    return 20 * np.log10(rms_orig / (rms_resid + 1e-9))

def main():
    print("--- EVALUATING SELECTIVE MULTI-INPUT ANC ---")
    
    # 1. Load Model with custom objects from research [cite: 338, 339]
    model_path = os.path.join(config.MODEL_SAVE_DIR, "tcn_phase4_final.h5")
    if not os.path.exists(model_path):
        print("Error: Trained model not found!")
        return

    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            'SISNRLoss': loss_functions.SISNRLoss,
            'get_last_block': tcn_constrained.get_last_block
        },
        compile=False
    )

    # 2. Generate a fresh test signal (820 RPM Engine)
    duration = 5
    t_vec = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
    f0 = (820 / 60) * 2 
    test_noise = 0.6 * np.sin(2 * np.pi * f0 * t_vec) + 0.05 * np.random.normal(0, 0.1, len(t_vec))
    test_noise /= np.max(np.abs(test_noise))

    # Simulate Primary Path P(z) [cite: 768, 822]
    p_ir, _ = sf.read(config.PRIMARY_PATH_FILE)
    d_at_ear = signal.fftconvolve(test_noise, p_ir, mode='same')
    
    # 3. Block-by-Block Processing with Feature Extraction
    u_anti_noise = np.zeros_like(d_at_ear)
    
    print(f"Processing {len(test_noise)//config.BLOCK_SIZE} blocks...")
    for i in range(config.WINDOW_SIZE, len(test_noise) - config.BLOCK_SIZE - config.LATENCY_SAMPLES, config.BLOCK_SIZE):
        # Slice current history window
        window = test_noise[i - config.WINDOW_SIZE : i]
        
        # Extract Pitch and Envelope for 'Selective' logic [cite: 134, 159]
        feats = extract_features_for_eval(window, config.SAMPLE_RATE)
        
        # Prepare inputs for the model
        in_audio = window.reshape(1, config.WINDOW_SIZE, 1)
        in_feats = np.array(feats).reshape(1, -1)
        
        # Predict 100-sample block
        pred_block = model.predict([in_audio, in_feats], verbose=0).flatten()
        
        # Place anti-noise in output buffer (accounting for latency) [cite: 441, 442]
        idx = i + config.LATENCY_SAMPLES
        u_anti_noise[idx : idx + config.BLOCK_SIZE] = pred_block

    # 4. Simulate Physical Summation [cite: 663, 753]
    s_ir, _ = sf.read(config.SECONDARY_PATH_FILE)
    y_anti_at_ear = signal.fftconvolve(u_anti_noise, s_ir, mode='same')
    residual = d_at_ear + y_anti_at_ear

    # 5. Results
    reduction = calculate_db_reduction(d_at_ear, residual)
    print(f"\n" + "="*30)
    print(f"FINAL PERFORMANCE: {reduction:.2f} dB")
    print("="*30)

    # Save audio for verification
    audio_dir = os.path.join(config.BASE_DIR, "results_audio")
    os.makedirs(audio_dir, exist_ok=True)
    sf.write(os.path.join(audio_dir, "01_original_noise.wav"), d_at_ear, config.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, "03_result_cancelled.wav"), residual, config.SAMPLE_RATE)
    print(f"Audio results saved to: {audio_dir}")

if __name__ == "__main__":
    main()