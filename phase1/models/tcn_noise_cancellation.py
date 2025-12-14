#pip install tensorflow scikit-learn pydot graphviz
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import os

# --- 1. TCN Model Class ---

class UrbanNoiseTCN:
    def __init__(self, sequence_length=10, kernel_size=3, filters=32, dilation_depth=4):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.filters = filters
        self.dilation_depth = dilation_depth
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a TCN with causal dilated convolutions.
        """
        input_layer = layers.Input(shape=(self.sequence_length, 1))
        x = input_layer

        # Add stack of dilated convolutions
        for i in range(self.dilation_depth):
            dilation_rate = 2 ** i
            # Causal padding ensures we don't "see into the future"
            x = layers.Conv1D(filters=self.filters,
                              kernel_size=self.kernel_size,
                              padding='causal',
                              dilation_rate=dilation_rate,
                              activation='relu')(x)
            x = layers.SpatialDropout1D(0.1)(x)

        # Final convolution to map back to 1 output value (the predicted amplitude)
        # We only care about the last time step for prediction
        x = layers.Dense(16, activation='relu')(x)
        output_layer = layers.Dense(1)(x)
        
        # We take the output of the last time step
        output_layer = layers.Lambda(lambda z: z[:, -1, :])(output_layer)

        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def train(self, X, y, epochs=20, batch_size=32):
        print(f"\n🧠 Training TCN Model for {epochs} epochs...")
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        self.plot_history(history)
        return history

    def predict(self, waveform):
        """
        Predicts the anti-noise (inverted phase) for the given waveform.
        """
        # Prepare sequences
        X, _ = prepare_sequences(waveform, self.sequence_length)
        
        # Predict noise
        print("🔮 Predicting noise patterns...")
        predicted_noise = self.model.predict(X)
        
        # Invert phase to create anti-noise
        predicted_anti_noise = -1 * predicted_noise.flatten()
        
        # Pad the beginning because we lost the first 'sequence_length' samples
        padding = np.zeros(self.sequence_length)
        predicted_anti_noise = np.concatenate([padding, predicted_anti_noise])
        
        return predicted_anti_noise

    def save(self, filepath="tcn_noise_model.h5"):
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")

    def load(self, filepath="tcn_noise_model.h5"):
        if os.path.exists(filepath):
            self.model = models.load_model(filepath)
            print(f"📂 Model loaded from {filepath}")
            return True
        else:
            print(f"❌ Model file {filepath} not found.")
            return False

    def plot_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- 2. Helper Functions ---

def generate_synthetic_noise(duration_sec=10, sr=16000):
    """
    Generates a mix of low-frequency hums and white noise (Urban environment simulation).
    """
    print("🔊 Generating synthetic urban noise...")
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    # Component 1: Low frequency engine hum (50Hz - 100Hz)
    hum1 = 0.5 * np.sin(2 * np.pi * 60 * t)
    hum2 = 0.3 * np.sin(2 * np.pi * 120 * t)
    
    # Component 2: Random traffic drone (White noise with envelope)
    noise = np.random.normal(0, 0.1, len(t))
    
    # Combine
    combined_noise = hum1 + hum2 + noise
    
    # Normalize
    combined_noise = combined_noise / np.max(np.abs(combined_noise))
    return combined_noise, sr

def prepare_sequences(waveform, sequence_length):
    """
    Converts a long 1D array into (X, y) pairs for time-series training.
    X = [t_0, t_1, ..., t_9]
    y = [t_10]
    """
    X = []
    y = []
    for i in range(len(waveform) - sequence_length):
        X.append(waveform[i:i+sequence_length])
        y.append(waveform[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for TCN input (Samples, TimeSteps, Features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def load_audio_file_ui():
    """ Opens a file picker dialog """
    print("\n📂 Opening file dialog...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Select WAV File", filetypes=[("WAV files", "*.wav")])
    root.destroy()
    return file_path

def evaluate_cancellation(original, anti_noise, sr):
    """
    Combines original + anti-noise and plots the result.
    """
    # Ensure lengths match
    min_len = min(len(original), len(anti_noise))
    original = original[:min_len]
    anti_noise = anti_noise[:min_len]
    
    # Apply cancellation (Sum them up)
    # Since anti_noise is already inverted phase, we just ADD them.
    result = original + anti_noise
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(original, color='blue')
    plt.title("Original Noise")
    
    plt.subplot(3, 1, 2)
    plt.plot(anti_noise, color='orange')
    plt.title("Predicted Anti-Noise (Inverted Phase)")
    
    plt.subplot(3, 1, 3)
    plt.plot(result, color='green')
    plt.title(f"Result (Cancellation) - Residual Energy: {np.sum(result**2):.2f}")
    
    plt.tight_layout()
    plt.show()
    
    # Save
    sf.write("result_cancellation.wav", result, sr)
    print("✅ Result saved to 'result_cancellation.wav'")

# --- 3. Main Execution ---

if __name__ == "__main__":
    SEQ_LEN = 20  # Look back 20 samples to predict the next one
    
    print("--- TCN Noise Cancellation System ---")
    mode = input("Do you want to (T)rain a new model or (P)redict on a file? ").lower().strip()
    
    tcn = UrbanNoiseTCN(sequence_length=SEQ_LEN)
    
    if mode == 't':
        # 1. Generate Data
        noise_data, sr = generate_synthetic_noise(duration_sec=10)
        
        # 2. Prepare Data
        print("⚙️ Processing data sequences...")
        X, y = prepare_sequences(noise_data, SEQ_LEN)
        
        # 3. Train
        tcn.train(X, y, epochs=15) # Adjust epochs as needed
        
        # 4. Save
        tcn.save()
        
    elif mode == 'p':
        # 1. Load Model
        if tcn.load():
            # 2. Load File
            file_path = load_audio_file_ui()
            if file_path:
                print(f"📂 Loading {file_path}...")
                waveform, sr = librosa.load(file_path, sr=16000)
                
                # 3. Predict Anti-Noise
                anti_noise = tcn.predict(waveform)
                
                # 4. Evaluate
                evaluate_cancellation(waveform, anti_noise, sr)
    else:
        print("Invalid selection.")