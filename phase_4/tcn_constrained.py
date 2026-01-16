import tensorflow as tf
from tensorflow.keras import layers, Model
import config

@tf.keras.utils.register_keras_serializable()
def get_last_block(seq):
    # Instead of slicing 1 sample, we slice the last 100 samples
    # to allow the loss function to 'see' the wave's shape
    return seq[:, -100:, :] 

def build_model():
    inputs = layers.Input(shape=(config.WINDOW_SIZE, 1), name="ref_input")
    x = inputs

    # Deep Dilated Stack
    dilations = [1, 2, 4, 8, 16, 32, 64, 128]
    for d in dilations:
        prev = x
        x = layers.Conv1D(32, kernel_size=5, dilation_rate=d, padding='causal', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        if prev.shape[-1] != 32:
            prev = layers.Conv1D(32, 1)(prev)
        x = layers.Add()([prev, x])

    # --- OUTPUT BLOCK ---
    # 1. Slice the last time step features
    x = layers.Lambda(get_last_block, name="slice_block")(x)
    
    # 2. Flatten the 100x32 features so Dense can process them
    x = layers.Flatten()(x)
    
    # 3. Predict the 100 samples of anti-noise
    x = layers.Dense(100, activation='tanh', name="control_signal_flat")(x)
    
    # 4. Reshape to (100, 1) to match the target 'd_block' shape exactly
    outputs = layers.Reshape((100, 1))(x)
    
    return Model(inputs=inputs, outputs=outputs)