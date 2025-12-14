import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from numpy import hamming
from scipy.io.wavfile import write # <-- NEW: Import the function to write WAV files

print("All libraries imported successfully!")
print(f"Using pyroomacoustics version: {pra.__version__}")

# This is a good place to start building your room
# --- 1. Create the Room ---

# Define room dimensions (Length, Width, Height) in meters
room_dim = [4, 3, 2]  # 4m long, 3m wide, 2m high

# Sampling frequency
fs = 16000

# Create the room
room = pra.ShoeBox(room_dim, fs=fs, max_order=3)

print("Step 1: Room created successfully.")

# --- 2. Create a Source Signal ---

# Create a 3-second white noise signal
duration_seconds = 3
noise_signal = np.random.randn(int(duration_seconds * fs))
# Normalize the signal to prevent clipping
noise_signal /= np.max(np.abs(noise_signal))

print("Step 2: 3-second noise signal created.")

# --- 3. Add Source and Microphone ---

# Define locations [x, y, z] in meters
source_loc = [1, 0.5, 1]     # Near one side
mic_loc = [3, 2.5, 1]        # Near the opposite corner

# Add the source TO THE ROOM, and this time, we pass in our 'noise_signal'
room.add_source(source_loc, signal=noise_signal)

# Add the microphone
room.add_microphone(mic_loc)

print("Step 3: Source (with signal) and Microphone added.")

# --- 4. Run Simulation and Save File ---

print("Step 4: Running simulation...")
# This command does the magic!
# It convolves the source signal with the RIR for you.
room.simulate()

print("Simulation finished.")

# Get the signal recorded by the first microphone
# (room.mic_array.signals is a 2D array: [mic_index, sample_index])
mic_signal = room.mic_array.signals[0, :]

# --- Save the audio to a WAV file ---
output_filename = "room_simulation.wav"

# Convert the audio from 32-bit float to 16-bit integer
# (This is the standard format for most WAV files)
mic_signal_int16 = np.int16(mic_signal * 32767)

# Write the file
write(output_filename, fs, mic_signal_int16)

print(f"Success! Audio saved to '{output_filename}'")

# --- 5. (Optional) Plot the recorded signal ---
plt.figure(figsize=(10, 4))
plt.plot(mic_signal)
plt.title("Signal Recorded at Microphone")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()