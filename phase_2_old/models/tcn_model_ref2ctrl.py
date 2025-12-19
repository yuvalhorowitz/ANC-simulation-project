# phase2/models/tcn_model_ref2ctrl.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def build_tcn_ref2ctrl(sequence_length=200, learning_rate=0.001):
    """
    Causal TCN that maps reference mic sequence → single control sample u[t].
    """

    inputs = layers.Input(shape=(sequence_length, 1))

    # TCN Layers
    x = layers.Conv1D(filters=32, kernel_size=3, padding='causal',
                      dilation_rate=1, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding='causal',
                      dilation_rate=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding='causal',
                      dilation_rate=4, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(1)(x)  # one control sample

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=4000,
        decay_rate=0.7,
        staircase=True
    )

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
        loss='mse'
    )

    return model
