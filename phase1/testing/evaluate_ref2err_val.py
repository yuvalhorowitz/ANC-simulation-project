# evaluate_ref2err_val.py
#
# Evaluate a trained ref->err TCN on all validation scenarios
# and print aggregate dB improvements.

import glob
import os
import tensorflow as tf

from tcn_model_ref2err import WaveformPredictorRef2ErrTCN

def main():
    # Paths
    model_path = "tcn_ref2err.keras"
    val_dir = "data/val"

    # Collect validation *_ref.wav files
    val_ref_files = sorted(glob.glob(os.path.join(val_dir, "*_ref.wav")))
    if not val_ref_files:
        raise RuntimeError(f"No *_ref.wav files found in {val_dir}")

    print(f"[INFO] Found {len(val_ref_files)} validation scenarios.")

    # Create predictor (for config only – we'll override the model)
    predictor = WaveformPredictorRef2ErrTCN(
        train_dir="data/train",
        val_dir="data/val",
        sample_rate=16000,
        sequence_length=200,
        learning_rate=0.005,
    )

    # Load saved model and override
    print(f"[INFO] Loading trained model from {model_path}")
    predictor.model = tf.keras.models.load_model(model_path)

    # Evaluate all validation scenarios
    predictor.evaluate_scenarios(val_ref_files)


if __name__ == "__main__":
    main()
