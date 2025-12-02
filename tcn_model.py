# tcn_model.py

import glob
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import Audio, display


class WaveformPredictorTCN:
    """
    TCN model that learns to output the inverted next sample of a waveform.

    For now:
      - Input:  last `sequence_length` samples of mic signal.
      - Target: -mic[t+sequence_length]  (phase inverted).
    """

    def __init__(
        self,
        train_dir="data/train",
        val_dir="data/val",
        sample_rate=16000,
        sequence_length=50,
        learning_rate=0.005,
    ):
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Collect training/validation WAV files (mic signals)
        self.train_files = sorted(glob.glob(os.path.join(train_dir, "*_mic.wav")))
        self.val_files = sorted(glob.glob(os.path.join(val_dir, "*_mic.wav")))

        if not self.train_files:
            raise ValueError(f"No training WAVs found in {train_dir}")
        if not self.val_files:
            raise ValueError(f"No validation WAVs found in {val_dir}")

        print(f"Found {len(self.train_files)} training files.")
        print(f"Found {len(self.val_files)} validation files.")

        # Choose strategy (CPU/GPU/TPU)
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            self.strategy = tf.distribute.TPUStrategy(tpu)
            print("Running on TPU")
        except ValueError:
            if tf.config.list_physical_devices("GPU"):
                self.strategy = tf.distribute.MirroredStrategy()
                print("Running on GPU")
            else:
                self.strategy = tf.distribute.get_strategy()
                print("Running on CPU")

        # Build model
        with self.strategy.scope():
            self.model = self.build_model()

        print(f"sample_rate: {self.sample_rate}")
        print(f"sequence_length: {self.sequence_length}")
        print(f"learning_rate: {self.learning_rate}")
        print()
        self.model.summary()

    # ----------------------------------------------------------------------
    # Model definition (same spirit as your original TCN)
    # ----------------------------------------------------------------------
    def build_model(self):
        input_layer = layers.Input(shape=(self.sequence_length, 1))

        # TCN-like stack of causal Conv1D layers
        x = layers.Conv1D(
            filters=32,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=1,
        )(input_layer)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(
            filters=32,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=2,
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(
            filters=64,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=4,
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        output_layer = layers.Dense(1)(x)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=5000,
            decay_rate=0.8,
            staircase=True,
        )

        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
            loss="mean_squared_error",
            metrics=[MeanAbsoluteError()],
        )
        return model

    # ----------------------------------------------------------------------
    # Data pipeline
    # ----------------------------------------------------------------------
    def _file_to_windows(self, wav_path):
        """Load one WAV and yield (X, y) sliding windows."""
        # librosa returns float32 by default
        signal, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        # Normalize to [-1, 1]
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal) + 1e-9)

        n = len(signal)
        # Basic safety: ignore too-short files
        if n <= self.sequence_length + 1:
            return

        for i in range(n - self.sequence_length - 1):
            x = signal[i : i + self.sequence_length]           # shape (L,)
            y = -signal[i + self.sequence_length]              # inverted next sample
            yield x, y

    def _dataset_generator(self, file_list):
        """Generator for tf.data.Dataset.from_generator"""
        for wav_path in file_list:
            for x, y in self._file_to_windows(wav_path):
                yield x, y

    def _count_samples(self, file_list):
        """Count total (#windows) samples for steps_per_epoch estimation."""
        total = 0
        for wav_path in file_list:
            signal, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            n = len(signal)
            if n > self.sequence_length + 1:
                total += (n - self.sequence_length - 1)
        return total

    def prepare_data(self, batch_size=32):
        """
        Build tf.data.Dataset objects for train and validation,
        plus steps_per_epoch / validation_steps.
        """
        # Count total samples (sliding windows)
        num_train_samples = self._count_samples(self.train_files)
        num_val_samples = self._count_samples(self.val_files)

        print(f"Total train windows: {num_train_samples}")
        print(f"Total val windows:   {num_val_samples}")

        steps_per_epoch = max(1, num_train_samples // batch_size)
        validation_steps = max(1, num_val_samples // batch_size)

        output_signature = (
            tf.TensorSpec(shape=(self.sequence_length,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(self.train_files),
            output_signature=output_signature,
        )
        val_ds = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(self.val_files),
            output_signature=output_signature,
        )

        # Shuffle, batch, prefetch, repeat
        train_ds = (
            train_ds.shuffle(2048)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        val_ds = (
            val_ds.shuffle(2048)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )

        return train_ds, val_ds, steps_per_epoch, validation_steps

    # ----------------------------------------------------------------------
    # Training and inference
    # ----------------------------------------------------------------------
    def train(self, epochs=5, batch_size=32):
        train_ds, val_ds, steps_per_epoch, validation_steps = self.prepare_data(
            batch_size=batch_size
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=1, restore_best_weights=True
        )

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=[early_stopping],
        )
        return history

    def predict_waveform(self, waveform):
        """
        Predict inverted samples for a raw waveform (numpy array).
        Returns a 1D array of predictions (len = len(waveform) - sequence_length - 1).
        """
        waveform = waveform.astype(np.float32)
        waveform /= np.max(np.abs(waveform) + 1e-9)

        X = []
        n = len(waveform)
        for i in range(n - self.sequence_length - 1):
            X.append(waveform[i : i + self.sequence_length])
        X = np.array(X, dtype=np.float32)
        X = X.reshape((-1, self.sequence_length, 1))

        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def test_with_wav(self, wav_path, zoom_duration=0.05):
        """
        Convenience method:
          - load a WAV,
          - run the model,
          - plot input / prediction / combined,
          - print cancellation MSE,
          - play audio.
        """
        signal, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        signal = signal.astype(np.float32)
        signal /= np.max(np.abs(signal) + 1e-9)

        pred = self.predict_waveform(signal)

        # Align prediction by padding the first sequence_length samples with 0
        pred_padded = np.pad(
            pred, (self.sequence_length + 1, 0), mode="constant", constant_values=0.0
        )

        combined = signal[: len(pred_padded)] + pred_padded

        # Compute simple MSE vs silence
        silence = np.zeros_like(combined)
        mse = mean_squared_error(silence, combined)
        print(f"Combined-to-zero validation MSE: {mse:.6e}")

        # Plot zoomed view
        n_zoom = int(self.sample_rate * zoom_duration)
        t_axis = np.arange(n_zoom) / self.sample_rate

        plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        plt.title("Input waveform")
        plt.plot(t_axis, signal[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 2)
        plt.title("Predicted inverted waveform")
        plt.plot(t_axis, pred_padded[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 3)
        plt.title("Combined (input + predicted)")
        plt.plot(t_axis, combined[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 4)
        plt.title("Residual (dB)")
        residual = combined[:n_zoom]
        residual_db = 20 * np.log10(np.abs(residual) + 1e-10)
        plt.plot(t_axis, residual_db)
        plt.ylabel("dB")
        plt.xlabel("Time [s]")

        plt.tight_layout()
        plt.show()

        # Play audio
        print("Input signal:")
        display(Audio(signal, rate=self.sample_rate))
        print("Predicted anti-noise:")
        display(Audio(pred_padded, rate=self.sample_rate))
        print("Combined (should be quieter):")
        display(Audio(combined, rate=self.sample_rate))

    # ----------------------------------------------------------------------
    # Saving / loading
    # ----------------------------------------------------------------------
    def save_model(self, path="tcn_noise_cancellation.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path="tcn_noise_cancellation.keras"):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
