from tcn_model_ref2err import WaveformPredictorRef2ErrTCN
import tensorflow as tf

# Load the trained model from disk
model = tf.keras.models.load_model("tcn_ref2err.keras")

# Create a temporary predictor object ONLY to access the test function
predictor = WaveformPredictorRef2ErrTCN(
    train_dir="data/train",
    val_dir="data/val",
    sample_rate=16000,
    sequence_length=200,
    learning_rate=0.005,
)

# Override the freshly-built model with the saved one
predictor.model = model

# Choose a scenario to test
ref_wav = "data/val/val_scenario_000_ref.wav"

# Run the test
predictor.test_on_scenario(ref_wav)
