import os

# --- PROJECT STRUCTURE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "train")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- ACOUSTIC PATH FILES ---
SECONDARY_PATH_FILE = os.path.join(PROCESSED_DATA_DIR, "secondary_path_impulse.wav")
PRIMARY_PATH_FILE = os.path.join(PROCESSED_DATA_DIR, "primary_path_impulse.wav")

# --- AUDIO PARAMETERS ---
SAMPLE_RATE = 8000
WINDOW_SIZE = 1000          # History window seen by the TCN
BLOCK_SIZE = 100            # <--- NEW: Number of samples predicted at once
HOP_SIZE = 20               # <--- NEW: Overlap for higher training density
BATCH_SIZE = 32

# --- PHYSICS CONSTRAINTS ---
LATENCY_SAMPLES = 4         # Hardware/Speaker delay
MAX_AMPLITUDE = 1.0

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 0.0005      
EPOCHS = 60                 
VALIDATION_SPLIT = 0.2

# --- LOSS WEIGHTS ---
LOW_PASS_CUTOFF = 400       # <--- NEW: Focus training on low-frequency engine drone
LAMBDA_CONTROL = 0.05       # Penalty to prevent signal clipping