import os
import numpy as np
import tensorflow as tf
import config
import tcn_constrained
import loss_functions

def train_model():
    print("--- STARTING PHASE 4 TRAINING ---")
    
    # 1. Load Dataset
    data_path = os.path.join(config.PROCESSED_DATA_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        print("Error: Dataset not found. Run generate_dataset.py first!")
        return
        
    data = np.load(data_path)
    X, y = data['X'], data['y']
    print(f"Loaded {X.shape[0]} training samples.")

    # 2. Build Model
    model = tcn_constrained.build_model()
    
    # 1. Initialize the new Physics-Informed Loss class
    # This loads the secondary path file to simulate the speaker's effect during training
    loss_fn = loss_functions.SecondaryPathLoss(config.SECONDARY_PATH_FILE)
    
    # 2. Compile the model using the new loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn  # <--- Use the new loss_fn here
    )
    # 5. Training Loop
    save_path = os.path.join(config.MODEL_SAVE_DIR, "tcn_phase4_final.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')

    model.fit(
        X, y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=[checkpoint]
    )
    
    print(f"--- TRAINING COMPLETE. Model saved to: {save_path}")

if __name__ == "__main__":
    train_model()