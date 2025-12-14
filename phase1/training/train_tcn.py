# train_tcn.py

from tcn_model import WaveformPredictorTCN

if __name__ == "__main__":
    # 1. Make sure you already ran simulation_setup.py (or its generate_dataset)
    #    so that data/train/*_mic.wav and data/val/*_mic.wav exist.

    predictor = WaveformPredictorTCN(
        train_dir="data/train",
        val_dir="data/val",
        sample_rate=16000,
        sequence_length=50,
        learning_rate=0.005,
    )

    # 2. Train
    predictor.train(epochs=5, batch_size=32)

    # 3. Test on one of the validation files
    predictor.test_with_wav("data/val/val_scenario_000_mic.wav")

    # 4. Optionally save the model
    predictor.save_model("tcn_room_anc.keras")
