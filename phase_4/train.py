import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import config
import tcn_constrained
import loss_functions

def train_model():
    print("--- STARTING MULTI-INPUT PHASE 4 TRAINING ---")
    
    # 1. Load Multi-Input Dataset
    # y now represents the target block, while X contains audio + features
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        print("Error: Dataset not found. Run the updated generate_dataset.py first!")
        return
        
    data = np.load(data_path)
    X_audio = data['X_audio']   # Raw reference signal
    X_feats = data['X_features'] # Pitch and Amplitude Envelope [cite: 19, 134]
    y_target = data['y']         # Target block for SI-SNR optimization [cite: 503]
    
    print(f"Loaded {X_audio.shape[0]} training samples.")

    # 2. Build Multi-Input Model (Integrating research insights)
    # Input 1: The TCN path for raw audio
    audio_in = layers.Input(shape=(config.WINDOW_SIZE, 1), name="audio_ref")
    tcn_output = tcn_constrained.build_tcn_backbone(audio_in)
    
    # Input 2: The Feature path (Pitch/Envelope) [cite: 135, 198]
    feat_in = layers.Input(shape=(X_feats.shape[1],), name="selective_features")
    f_dense = layers.Dense(16, activation='relu')(feat_in)
    
    # 3. Concatenate (Selective Fusion)
    # This allows the model to weight the anti-noise based on detected pitch [cite: 195, 209]
    combined = layers.Concatenate()([tcn_output, f_dense])
    
    # 4. Magnitude and Complex Phase Core (ByteDance MC-TCN Style) [cite: 338, 454]
    x = layers.Dense(256, activation='relu')(combined)
    mag_mask = layers.Dense(config.BLOCK_SIZE, activation='sigmoid', name="mag_mask")(x)
    phase_refine = layers.Dense(config.BLOCK_SIZE, activation='tanh', name="phase_refine")(x)
    
    # Combine masks to form the anti-noise block [cite: 512, 513]
    combined_output = layers.Multiply()([mag_mask, phase_refine])
    final_output = layers.Reshape((config.BLOCK_SIZE, 1))(combined_output)
    
    model = Model(inputs=[audio_in, feat_in], outputs=final_output)

    # 5. Compile with SI-SNR Loss [cite: 506]
    # SI-SNR is scale-invariant and optimizes for destructive interference [cite: 503, 219]
    loss_fn = loss_functions.SISNRLoss()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn
    )

    # 6. Training with Early Stopping to prevent overfitting [cite: 509]
    save_path = os.path.join(config.MODEL_SAVE_DIR, "tcn_phase4_final.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    model.fit(
        [X_audio, X_feats], y_target,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks
    )
    
    print(f"--- TRAINING COMPLETE. Selective Model saved to: {save_path}")

if __name__ == "__main__":
    train_model()