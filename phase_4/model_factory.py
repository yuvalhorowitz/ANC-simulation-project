import tensorflow as tf
import tcn_constrained
import loss_functions
import config

def create_compiled_model():
    # 1. Build Architecture
    model = tcn_constrained.build_model()
    
    # 2. Define Loss
    loss_fn = loss_functions.PhysicsLoss()
    
    # 3. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model