import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Simple TCN-based ANC training script with loss tracking and visualization.
# In this version the secondery path is modeled as a fixed FIR filter (hs_rir - simple dot product not a full convolusion) applied to the control signal.
# Settings for rapid iteration
DATA_DIR = "fast_data"
SCENARIO_ID = 0
WINDOW_SIZE = 200
EPOCHS = 30  # Increased slightly to see a clearer curve
BATCH_SIZE = 32

def load_data(scn_id):
    """Load signals and ensure float32 precision."""
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs = np.load(f"{prefix}_hs.npy").astype(np.float32)
    return ref, mic, hs

def build_simple_tcn(window_size):
    """Lightweight TCN with causal padding and tanh output restraint."""
    inputs = tf.keras.Input(shape=(window_size, 1))
    
    # Feature extraction with ReLU
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Output constrained by tanh to prevent speaker blow-out
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def prepare_sequences(ref, mic, window_size):
    """Prepare sliding window sequences for TCN ingestion."""
    X, y = [], []
    for i in range(len(ref) - window_size):
        X.append(ref[i:i+window_size])
        y.append(mic[i+window_size])
    return np.array(X, dtype=np.float32).reshape(-1, window_size, 1), np.array(y, dtype=np.float32)

class ANCTrainer:
    def __init__(self, model, hs_rir):
        self.model = model
        self.hs_rir = tf.constant(hs_rir, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(self, x_batch, y_batch):
        """Perform a single Gradient Descent step using Filtered-X logic."""
        with tf.GradientTape() as tape:
            u = self.model(x_batch, training=True)
            hs_gain = tf.reduce_sum(self.hs_rir) 
            y_pred = u * hs_gain 
            loss = tf.reduce_mean(tf.square(y_batch + y_pred))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

if __name__ == "__main__":
    print("--- TCN Training with Loss Tracking ---")
    
    # 1. Setup
    ref_sig, mic_sig, hs_rir = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences(ref_sig, mic_sig, WINDOW_SIZE)
    model = build_simple_tcn(WINDOW_SIZE)
    trainer = ANCTrainer(model, hs_rir)
    
    # 2. History tracking
    loss_history = []

    # 3. Training Loop
    print(f"Training on Scenario {SCENARIO_ID} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        batch_losses = []
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            loss = trainer.train_step(x_batch, y_batch)
            batch_losses.append(loss.numpy())
        
        avg_loss = np.mean(batch_losses)
        loss_history.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:02d}: Loss (MSE) = {avg_loss:.6f}")

    # 4. Final Visualization
    print("Generating comprehensive performance plot...")
    u_pred = model.predict(X_train)
    anti_noise = (u_pred * np.sum(hs_rir)).flatten()
    residual = y_train + anti_noise
    
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Signal Comparison
    plt.subplot(3, 1, 1)
    plt.plot(y_train[1000:1500], label="Original Noise (At Ear)", alpha=0.6)
    plt.plot(residual[1000:1500], label="Residual Noise (ANC ON)", color='green', linewidth=2)
    plt.title(f"Time Domain Performance - Scenario {SCENARIO_ID}")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Control Signal (u)
    plt.subplot(3, 1, 2)
    plt.plot(u_pred[1000:1500], label="Control Signal (Predicted 'u')", color='orange')
    plt.title("Model Prediction Pattern")
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Training Loss History (MSE)
    plt.subplot(3, 1, 3)
    plt.plot(loss_history, color='red', marker='o', linestyle='-', markersize=4)
    plt.title("Training Convergence (MSE Loss vs. Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Mean Squared Error)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fast_training_result_with_loss.png")
    print("✓ Success! Detailed results saved to 'fast_training_result_with_loss.png'")