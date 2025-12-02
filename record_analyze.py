#pip install sounddevice soundfile librosa matplotlib numpy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog

def record_audio_local(duration_sec, sample_rate=16000):
    """
    Records audio from the default microphone.
    """
    print(f"\n🎙️  Recording for {duration_sec} seconds... (Speak now!)")
    
    recording = sd.rec(int(duration_sec * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1)
    sd.wait() 
    print("✅ Recording complete.")
    return recording.flatten()

def load_audio_file_ui(target_sr=16000):
    """
    Opens a file dialog to let the user select a WAV file.
    """
    print("\n📂 Opening file dialog... (Check your taskbar if it doesn't appear!)")
    
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select an Audio File",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    if file_path:
        print(f"✅ Selected: {file_path}")
        print("⏳ Loading audio file... (This may take a moment)")
        y, _ = librosa.load(file_path, sr=target_sr)
        return y
    else:
        print("❌ No file selected.")
        return None

def calculate_cutoff_frequency(y, sr, retention_percentage=95.0):
    fft_spectrum = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(y), d=1/sr)
    
    magnitude = np.abs(fft_spectrum)
    half_point = len(y) // 2
    magnitude = magnitude[:half_point]
    frequencies = frequencies[:half_point]
    
    total_energy = np.sum(magnitude)
    cumulative_energy = np.cumsum(magnitude)
    
    threshold_energy = total_energy * (retention_percentage / 100.0)
    
    if len(cumulative_energy) == 0:
        return 0
        
    cutoff_index = np.where(cumulative_energy >= threshold_energy)[0][0]
    return frequencies[cutoff_index]

def resample_audio(y, original_sr, target_sr):
    return librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)

def augment_wave(y, sr):
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    noise_amp = 0.005 * np.max(y)
    y_noisy = y_shifted + noise_amp * np.random.normal(size=len(y_shifted))
    return y_noisy

def analyze_audio(y, sr, title="Audio Analysis"):
    """
    Plots the Waveform and Spectrogram.
    """
    print(f"\n📊 Generating graphs for: {title}")
    print("⚠️  IMPORTANT: You must CLOSE the popup graph window to continue the program! ⚠️")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{title} - Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} - Spectrogram")
    
    plt.tight_layout()
    plt.show() # This line pauses the code until the window is closed

def play_audio_local(y, sr):
    print("🔊 Playing audio...")
    sd.play(y, sr)
    sd.wait()

if __name__ == "__main__":
    SAMPLE_RATE = 16000
    DURATION = 5 
    DESIRED_RETENTION = 95.0 
    
    print("--- Audio Analysis Tool ---")
    choice = input("Do you want to Record (r) or Load a file (l)? ").lower().strip()
    
    audio = None
    
    if choice == 'r':
        audio = record_audio_local(DURATION, SAMPLE_RATE)
        sf.write("recorded_original.wav", audio, SAMPLE_RATE)
        
    elif choice == 'l':
        audio = load_audio_file_ui(SAMPLE_RATE)
        
    else:
        print("Invalid choice. Please run again and choose 'r' or 'l'.")

    if audio is not None:
        # 2. Analyze Original
        # The script will PAUSE here until you close the graph
        analyze_audio(audio, SAMPLE_RATE, title="Original Signal")
        
        print("⏳ Calculating frequencies and resampling...")
        
        # 3. Calculate Cutoff
        cutoff_freq = calculate_cutoff_frequency(audio, SAMPLE_RATE, DESIRED_RETENTION)
        print(f"\n📊 Analysis Results:")
        print(f"   Desired Energy Retention: {DESIRED_RETENTION}%")
        print(f"   Calculated Cutoff Frequency: {cutoff_freq:.2f} Hz")
        
        calc_rate = int(np.ceil(2 * cutoff_freq))
        target_sample_rate = max(calc_rate, 4000) 
        
        print(f"   New Target Sample Rate: {target_sample_rate} Hz")
        
        resampled_audio = resample_audio(audio, SAMPLE_RATE, target_sample_rate)
        
        # 5. Augment
        augmented_audio = augment_wave(resampled_audio, target_sample_rate)
        
        # 6. Playback & Save Results
        print("\n--- Playback Session ---")
        
        print(f"Playing Resampled Version ({target_sample_rate} Hz)...")
        play_audio_local(resampled_audio, target_sample_rate)
        sf.write("processed_resampled.wav", resampled_audio, target_sample_rate)
        
        print("Playing Augmented Version...")
        play_audio_local(augmented_audio, target_sample_rate)
        sf.write("processed_augmented.wav", augmented_audio, target_sample_rate)

        print("\n✅ Done! Check your folder for 'processed_resampled.wav' and 'processed_augmented.wav'.")