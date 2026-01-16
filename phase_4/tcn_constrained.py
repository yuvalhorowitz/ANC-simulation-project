import tensorflow as tf
from tensorflow.keras import layers, Model
import config

@tf.keras.utils.register_keras_serializable()
def get_last_block(seq):
    """
    Slices the last BLOCK_SIZE samples from the sequence.
    This allows the loss function to 'see' a wave block for convolution.
    [cite: 451, 893]
    """
    return seq[:, -config.BLOCK_SIZE:, :] 

def build_tcn_backbone(input_layer):
    """
    The main TCN architecture using dilated convolutions to create 
    a large receptive field for engine noise.
    [cite: 354, 493]
    """
    x = input_layer
    
    # 8-Layer Dilated Stack covering ~1021 samples (127ms @ 8kHz)
    # [cite: 539, 541]
    dilations = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for d in dilations:
        prev = x
        
        # 1D causal dilated convolutional unit [cite: 493]
        # Kernel size 5 chosen for better temporal capture
        x = layers.Conv1D(filters=32, 
                          kernel_size=5, 
                          dilation_rate=d, 
                          padding='causal', 
                          activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Residual connection to prevent vanishing gradients
        if prev.shape[-1] != 32:
            prev = layers.Conv1D(filters=32, kernel_size=1)(prev)
        x = layers.Add()([prev, x])

    # Slice the final feature block [cite: 496]
    x = layers.Lambda(get_last_block, name="slice_block")(x)
    
    # Flatten the 100x32 features so they can be fused with Pitch/Envelope data
    return layers.Flatten()(x)

def build_model():
    """
    Standalone version of the model for legacy compatibility.
    Uses a standard Dense output instead of the multi-input fusion.
    """
    # Input: Reference Mic Audio history (1000 samples)
    inputs = layers.Input(shape=(config.WINDOW_SIZE, 1), name="ref_input")
    
    # Extract temporal features
    features = build_tcn_backbone(inputs)
    
    # --- Magnitude and Complex Phase Core (MC-TCN Style) ---
    # [cite: 338, 569]
    
    # Dense layer to interpret flattened TCN features
    x = layers.Dense(256, activation='relu')(features)
    
    # 1. Magnitude Mask Core: Determines the volume of anti-noise [cite: 390, 569]
    mag_mask = layers.Dense(config.BLOCK_SIZE, activation='sigmoid', name="mag_mask")(x)
    
    # 2. Complex Phase Refinement: Adjusts the timing/phase [cite: 392, 570]
    phase_refine = layers.Dense(config.BLOCK_SIZE, activation='tanh', name="phase_correction")(x)
    
    # Combine masks to form the final anti-noise block 
    combined_output = layers.Multiply()([mag_mask, phase_refine])
    
    # Reshape to match target block (100, 1)
    outputs = layers.Reshape((config.BLOCK_SIZE, 1), name="control_signal")(combined_output)
    
    model = Model(inputs=inputs, outputs=outputs, name="Selective_MC_TCN")
    return model