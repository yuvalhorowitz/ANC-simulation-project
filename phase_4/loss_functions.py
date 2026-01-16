import tensorflow as tf
import soundfile as sf
import scipy.signal as signal
import config
import os

class SecondaryPathLoss(tf.keras.losses.Loss):
    def __init__(self, secondary_path_file):
        super().__init__()
        # 1. Load S(z) and format for Conv1D (Kernel_Size, In_Channels, Out_Channels)
        s_z_raw, _ = sf.read(secondary_path_file)
        self.s_z = tf.constant(s_z_raw.reshape(-1, 1, 1), dtype=tf.float32)
        
        # 2. Engine Band Priority Filter (Low-Pass 400Hz)
        nyq = config.SAMPLE_RATE / 2
        cutoff = config.LOW_PASS_CUTOFF / nyq
        fir_coeff = signal.firwin(31, cutoff)
        self.lpf = tf.constant(fir_coeff.reshape(-1, 1, 1), dtype=tf.float32)

    def call(self, d_target, u_pred):
        """
        d_target: (Batch, 100, 1) - True noise at ear
        u_pred:   (Batch, 100, 1) - Model's raw output
        """
        # --- PHYSICS SIMULATION ---
        # Convolve Anti-Noise with Secondary Path (Speaker Effect)
        # padding='SAME' ensures the output stays 100 samples
        y_anti_noise = tf.nn.conv1d(u_pred, self.s_z, stride=1, padding='SAME')
        
        # --- RESIDUAL CALCULATION ---
        # Error = Noise + Anti-Noise (destructive interference)
        residual = d_target + y_anti_noise
        
        # --- FREQUENCY WEIGHTING ---
        # Apply LPF to focus training on Engine Drone
        res_filtered = tf.nn.conv1d(residual, self.lpf, stride=1, padding='SAME')
        
        # --- MULTI-OBJECTIVE LOSS ---
        # 1. Minimize Engine Noise
        mse_loss = tf.reduce_mean(tf.square(res_filtered))
        
        # 2. Penalty for 'Screaming' (Control Signal Magnitude)
        control_penalty = tf.reduce_mean(tf.square(u_pred))
        
        return mse_loss + (config.LAMBDA_CONTROL * control_penalty)