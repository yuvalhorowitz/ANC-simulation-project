# train_tcn_ref2err.py

from tcn_model_ref2err import WaveformPredictorRef2ErrTCN

if __name__ == "__main__":
    predictor = WaveformPredictorRef2ErrTCN(
        train_dir="data/train",
        val_dir="data/val",
        sample_rate=16000,
        sequence_length=200,
        learning_rate=0.005,
    )

    predictor.train(epochs=5, batch_size=32)

    # Test on first validation scenario
    predictor.test_on_scenario("data/val/val_scenario_000_ref.wav")

    predictor.save_model("tcn_ref2err.keras")
