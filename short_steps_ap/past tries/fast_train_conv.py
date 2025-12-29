import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Settings
DATA_DIR = "fast_data"
SCENARIO_ID = 0
WINDOW_SIZE = 200
EPOCHS = 40 # Increased epochs as convolution learning is more complex
BATCH_SIZE = 64 # Larger batch for stable convolution gradients

def load_data(scn_id):
    """Load signals with float32 precision."""
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    ref = np.load(f"{prefix}_ref.npy").astype(np.float32)
    mic = np.load(f"{prefix}_mic.npy").astype(np.float32)
    hs_rir = np.load(f"{prefix}_hs.npy").astype(np.float32)
    return ref, mic, hs_rir

def build_simple_tcn(window_size):
    """Lightweight TCN for ANC."""
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def prepare_sequences(ref, mic, window_size):
    """Prepare sequential windows for training."""
    X, y = [], []
    for i in range(len(ref) - window_size):
        X.append(ref[i:i+window_size])
        y.append(mic[i+window_size])
    return np.array(X, dtype=np.float32).reshape(-1, window_size, 1), np.array(y, dtype=np.float32)

class ANCTrainerConv:
    def __init__(self, model, hs_rir):
        self.model = model
        # Prepare Hs RIR as a fixed convolution kernel
        # Shape for conv1d: [filter_width, in_channels, out_channels]
        self.hs_filter = tf.cast(hs_rir[::-1], tf.float32) # Flip for standard convolution
        self.hs_filter = tf.reshape(self.hs_filter, [-1, 1, 1])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    @tf.function
    def train_step(self, x_batch, y_batch):
        """
        Training step with Secondary Path (Hs) Convolution.
        This simulates the physical delay and frequency response of the speaker.
        """
        with tf.GradientTape() as tape:
            # 1. Generate anti-noise samples for the batch
            u = self.model(x_batch, training=True) # [Batch, 1]
            
            # 2. Reshape u to [1, Batch, 1] to apply convolution across the time/batch axis
            u_seq = tf.reshape(u, [1, -1, 1])
            
            # 3. Apply Secondary Path Filter (Convolution)
            # This 'smears' the effect of each u_t over the subsequent samples
            y_pred_seq = tf.nn.conv1d(u_seq, self.hs_filter, stride=1, padding='SAME')
            y_pred = tf.reshape(y_pred_seq, [-1])
            
            # 4. Minimize residual error at the ear
            loss = tf.reduce_mean(tf.square(y_batch + y_pred))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

if __name__ == "__main__":
    print("--- TCN Training with Full Secondary Path Convolution ---")
    
    # 1. Setup
    ref_sig, mic_sig, hs_rir = load_data(SCENARIO_ID)
    X_train, y_train = prepare_sequences(ref_sig, mic_sig, WINDOW_SIZE)
    model = build_simple_tcn(WINDOW_SIZE)
    trainer = ANCTrainerConv(model, hs_rir)
    
    loss_history = []

    # 2. Training Loop
    print(f"Training on Scenario {SCENARIO_ID}...")
    for epoch in range(EPOCHS):
        batch_losses = []
        # We process in large contiguous chunks to maintain convolution continuity
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            
            if len(x_batch) < BATCH_SIZE: continue # Skip partial batches
            
            loss = trainer.train_step(x_batch, y_batch)
            batch_losses.append(loss.numpy())
        
        avg_loss = np.mean(batch_losses)
        loss_history.append(avg_loss)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}: Loss = {avg_loss:.6f}")

    # 3. Evaluation with Convolution
    print("Simulating final performance...")
    u_pred = model.predict(X_train)
    # Apply physical filter for visualization
    u_tensor = tf.reshape(tf.constant(u_pred, dtype=tf.float32), [1, -1, 1])
    anti_noise_ear = tf.nn.conv1d(u_tensor, trainer.hs_filter, stride=1, padding='SAME')
    anti_noise_ear = tf.reshape(anti_noise_ear, [-1]).numpy()
    
    residual = y_train + anti_noise_ear

    # 4. Visualization
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(y_train[2000:2500], label="Noise at Ear (Original)", alpha=0.6)
    plt.plot(residual[2000:2500], label="Residual Noise (ANC ON)", color='green', linewidth=2)
    plt.title("Physical Simulation Result (Convolution Applied)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(u_pred[2000:2500], label="Model Output (u)", color='orange')
    plt.title("Control Signal at Speaker")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(loss_history, color='red')
    plt.title("Learning Curve (Convolution-Aware Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("conv_training_result.png")
    print("✓ Done. Check 'conv_training_result.png' for physical accuracy.")