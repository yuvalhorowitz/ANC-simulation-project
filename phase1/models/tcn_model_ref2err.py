# tcn_model_ref2err.py
#
# TCN that learns to predict an anti-noise signal for the error microphone
# using only the reference microphone history.
#
# Training data is expected in folders like:
#   data/train/train_scenario_000_ref.wav
#   data/train/train_scenario_000_err.wav
#   data/val/val_scenario_000_ref.wav
#   data/val/val_scenario_000_err.wav
#
# For each scenario:
#   X[i] = ref[t = i : i+sequence_length]
#   y[i] = -err[t = i+sequence_length]
#
# Author: you + ChatGPT :)

import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import MeanAbsoluteError


class WaveformPredictorRef2ErrTCN:
    def __init__(
        self,
        train_dir: str = "data/train",
        val_dir: str = "data/val",
        sample_rate: int = 16000,
        sequence_length: int = 200,
        learning_rate: float = 0.005,
    ):
        """
        TCN model that learns:
            ref history  -->  anti-noise for error mic

        Args:
            train_dir: folder with *_ref.wav / *_err.wav for training
            val_dir:   folder with *_ref.wav / *_err.wav for validation
            sample_rate: audio sample rate (Hz)
            sequence_length: number of past reference samples to use
            learning_rate: initial LR for AdamW (with decay)
        """

        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Collect reference files for train/val splits
        self.train_ref_files = sorted(
            glob.glob(os.path.join(train_dir, "*_ref.wav"))
        )
        self.val_ref_files = sorted(
            glob.glob(os.path.join(val_dir, "*_ref.wav"))
        )

        if not self.train_ref_files:
            raise ValueError(f"No training *_ref.wav files found in {train_dir}")
        if not self.val_ref_files:
            raise ValueError(f"No validation *_ref.wav files found in {val_dir}")

        print(f"Found {len(self.train_ref_files)} training scenarios.")
        print(f"Found {len(self.val_ref_files)} validation scenarios.")
        print(f"sample_rate: {self.sample_rate}")
        print(f"sequence_length: {self.sequence_length}")
        print(f"learning_rate: {self.learning_rate}")

        # Strategy: TPU > GPU > CPU
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

        with self.strategy.scope():
            self.model = self._build_model()

        print()
        self.model.summary()

    # ------------------------------------------------------------------
    # Model definition: deeper causal TCN for longer receptive field
    # ------------------------------------------------------------------
    def _build_model(self) -> tf.keras.Model:
        """
        Build a 1D TCN-like model. Input shape: (sequence_length, 1).
        """
        input_layer = layers.Input(shape=(self.sequence_length, 1))

        x = input_layer

        # Several dilated causal conv layers to increase receptive field
        for dilation in [1, 2, 4, 8, 16, 32]:
            x = layers.Conv1D(
                filters=32,
                kernel_size=3,
                padding="causal",
                activation="relu",
                dilation_rate=dilation,
            )(x)
            x = layers.BatchNormalization()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
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

    # ------------------------------------------------------------------
    # Data pipeline: windows from (ref, err)
    # ------------------------------------------------------------------
    def _scenario_to_windows(self, ref_path):
        """
        Given path like '..._ref.wav', load the matching '..._err.wav'
        and yield (X, y) windows:

            X: ref[t : t+L]
            y: -err[t+L]
        """
        err_path = ref_path.replace("_ref.wav", "_err.wav")
        if not os.path.exists(err_path):
            raise FileNotFoundError(f"Matching *_err.wav not found for {ref_path}")

        # load reference and error signals
        ref_sig, _ = librosa.load(ref_path, sr=self.sample_rate, mono=True)
        err_sig, _ = librosa.load(err_path, sr=self.sample_rate, mono=True)

        n = min(len(ref_sig), len(err_sig))
        ref_sig = ref_sig[:n].astype(np.float32)
        err_sig = err_sig[:n].astype(np.float32)

        # normalize separately
        ref_sig /= np.max(np.abs(ref_sig) + 1e-9)
        err_sig /= np.max(np.abs(err_sig) + 1e-9)

        if n <= self.sequence_length + 1:
            return

        for i in range(n - self.sequence_length - 1):
            x = ref_sig[i : i + self.sequence_length]
            y = -err_sig[i + self.sequence_length]
            yield x, y

    def _dataset_generator(self, ref_file_list):
        for ref_path in ref_file_list:
            for x, y in self._scenario_to_windows(ref_path):
                yield x, y

    def _count_samples(self, ref_file_list) -> int:
        total = 0
        for ref_path in ref_file_list:
            err_path = ref_path.replace("_ref.wav", "_err.wav")
            if not os.path.exists(err_path):
                continue
            ref_sig, _ = librosa.load(ref_path, sr=self.sample_rate, mono=True)
            err_sig, _ = librosa.load(err_path, sr=self.sample_rate, mono=True)
            n = min(len(ref_sig), len(err_sig))
            if n > self.sequence_length + 1:
                total += (n - self.sequence_length - 1)
        return total

    def prepare_data(self, batch_size: int = 32):
        """
        Build tf.data.Dataset objects (train & val) and compute
        steps_per_epoch / validation_steps.
        """
        num_train_samples = self._count_samples(self.train_ref_files)
        num_val_samples = self._count_samples(self.val_ref_files)

        print(f"Total train windows: {num_train_samples}")
        print(f"Total val windows:   {num_val_samples}")

        steps_per_epoch = max(1, num_train_samples // batch_size)
        validation_steps = max(1, num_val_samples // batch_size)

        output_signature = (
            tf.TensorSpec(shape=(self.sequence_length,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(self.train_ref_files),
            output_signature=output_signature,
        )
        val_ds = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(self.val_ref_files),
            output_signature=output_signature,
        )

        train_ds = (
            train_ds.shuffle(4096)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        val_ds = (
            val_ds.shuffle(4096)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )

        return train_ds, val_ds, steps_per_epoch, validation_steps

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, epochs: int = 5, batch_size: int = 32):
        train_ds, val_ds, steps_per_epoch, validation_steps = self.prepare_data(
            batch_size=batch_size
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
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

    # ------------------------------------------------------------------
    # Evaluation on a single scenario
    # ------------------------------------------------------------------
    def test_on_scenario(self, ref_wav_path: str, zoom_duration: float = 0.05):
        """
        Evaluate ANC performance on one scenario.

        ref_wav_path: path to a *_ref.wav file (validation or train).
        Uses the matching *_err.wav file and measures cancellation
        at the error mic: combined = err + predicted_anti_noise.
        """
        err_wav_path = ref_wav_path.replace("_ref.wav", "_err.wav")
        if not os.path.exists(err_wav_path):
            raise FileNotFoundError(f"No matching *_err.wav for {ref_wav_path}")

        ref_sig, _ = librosa.load(ref_wav_path, sr=self.sample_rate, mono=True)
        err_sig, _ = librosa.load(err_wav_path, sr=self.sample_rate, mono=True)

        n = min(len(ref_sig), len(err_sig))
        ref_sig = ref_sig[:n].astype(np.float32)
        err_sig = err_sig[:n].astype(np.float32)

        ref_sig /= np.max(np.abs(ref_sig) + 1e-9)
        err_sig /= np.max(np.abs(err_sig) + 1e-9)

        # Build input windows from reference
        X = []
        for i in range(n - self.sequence_length - 1):
            X.append(ref_sig[i : i + self.sequence_length])

        X = np.array(X, dtype=np.float32).reshape(-1, self.sequence_length, 1)

        preds = self.model.predict(X, verbose=0).flatten()
        preds_padded = np.pad(
            preds,
            (self.sequence_length + 1, 0),
            mode="constant",
            constant_values=0.0,
        )
        preds_padded = preds_padded[:n]

        # Cancellation at error mic
        combined = err_sig + preds_padded

        mse = float(np.mean(combined**2))
        print(f"Error-mic combined-to-zero MSE: {mse:.6e}")

        # Plot a zoomed-in segment
        n_zoom = min(int(self.sample_rate * zoom_duration), n)
        t_axis = np.arange(n_zoom) / self.sample_rate

        plt.figure(figsize=(12, 9))

        plt.subplot(4, 1, 1)
        plt.title("Reference mic signal")
        plt.plot(t_axis, ref_sig[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 2)
        plt.title("Error mic signal (input noise)")
        plt.plot(t_axis, err_sig[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 3)
        plt.title("Combined at error mic (err + predicted)")
        plt.plot(t_axis, combined[:n_zoom])
        plt.ylabel("Amp")

        plt.subplot(4, 1, 4)
        plt.title("Residual at error mic (dB)")
        residual_db = 20 * np.log10(np.abs(combined[:n_zoom]) + 1e-10)
        plt.plot(t_axis, residual_db)
        plt.ylabel("dB")
        plt.xlabel("Time [s]")

        plt.tight_layout()
        plt.show()

        mse_before = float(np.mean(err_sig**2))
        mse_after  = float(np.mean(combined**2))
        improvement_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

        print(f"MSE BEFORE (error mic): {mse_before:.6e}")
        print(f"MSE AFTER  (combined):  {mse_after:.6e}")
        print(f"Energy reduction:       {improvement_db:.2f} dB")

        # ------------------------------------------------------------------
    # Aggregate evaluation on multiple scenarios (no plots)
    # ------------------------------------------------------------------
    def evaluate_scenarios(self, ref_paths):
        """
        Evaluate noise reduction for a list of *_ref.wav files.
        Returns a list of dicts with per-scenario metrics and
        prints aggregate statistics.

        Each entry in the result list has:
            {
              "ref_path": ...,
              "mse_before": ...,
              "mse_after": ...,
              "delta_db": ...
            }
        """
        results = []

        for ref_wav_path in ref_paths:
            err_wav_path = ref_wav_path.replace("_ref.wav", "_err.wav")
            if not os.path.exists(err_wav_path):
                print(f"[WARN] No matching *_err.wav for {ref_wav_path}, skipping.")
                continue

            # load signals
            ref_sig, _ = librosa.load(ref_wav_path, sr=self.sample_rate, mono=True)
            err_sig, _ = librosa.load(err_wav_path, sr=self.sample_rate, mono=True)

            n = min(len(ref_sig), len(err_sig))
            ref_sig = ref_sig[:n].astype(np.float32)
            err_sig = err_sig[:n].astype(np.float32)

            ref_sig /= np.max(np.abs(ref_sig) + 1e-9)
            err_sig /= np.max(np.abs(err_sig) + 1e-9)

            # build windows from reference
            if n <= self.sequence_length + 1:
                print(f"[WARN] Signal too short for {ref_wav_path}, skipping.")
                continue

            X = []
            for i in range(n - self.sequence_length - 1):
                X.append(ref_sig[i : i + self.sequence_length])
            X = np.array(X, dtype=np.float32).reshape(-1, self.sequence_length, 1)

            preds = self.model.predict(X, verbose=0).flatten()
            preds_padded = np.pad(
                preds,
                (self.sequence_length + 1, 0),
                mode="constant",
                constant_values=0.0,
            )
            preds_padded = preds_padded[:n]

            combined = err_sig + preds_padded

            mse_before = float(np.mean(err_sig**2))
            mse_after = float(np.mean(combined**2))
            delta_db = 10.0 * np.log10((mse_before + 1e-12) / (mse_after + 1e-12))

            results.append(
                {
                    "ref_path": ref_wav_path,
                    "mse_before": mse_before,
                    "mse_after": mse_after,
                    "delta_db": delta_db,
                }
            )

        if not results:
            print("[INFO] No valid scenarios evaluated.")
            return results

        # Aggregate stats
        deltas = np.array([r["delta_db"] for r in results])
        mean_db = float(np.mean(deltas))
        std_db = float(np.std(deltas))

        print("\n=== Per-scenario results ===")
        for i, r in enumerate(results):
            print(
                f"[{i:02d}] {os.path.basename(r['ref_path'])}: "
                f"ΔE = {r['delta_db']:.2f} dB "
                f"(before={r['mse_before']:.3e}, after={r['mse_after']:.3e})"
            )

        print("\n=== Aggregate over scenarios ===")
        print(f"Average energy reduction: {mean_db:.2f} dB")
        print(f"Std of energy reduction:  {std_db:.2f} dB")

        return results

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_model(self, path: str = "tcn_ref2err.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "tcn_ref2err.keras"):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
