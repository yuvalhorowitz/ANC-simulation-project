import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class SISNRLoss(tf.keras.losses.Loss):
    """
    Scale-Invariant Source-to-Noise Ratio (SI-SNR) Loss.
    Optimizes the ratio between the target signal and the noise/error.
    """
    def __init__(self, name="si_snr_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # 1. Zero-mean normalization
        y_true = y_true - tf.reduce_mean(y_true, axis=1, keepdims=True)
        y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)

        # 2. Project predicted signal onto target (s_target = <y_pred, y_true> * y_true / ||y_true||^2)
        dot_product = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)
        norm_true = tf.reduce_sum(y_true**2, axis=1, keepdims=True) + 1e-8
        s_target = (dot_product * y_true) / norm_true

        # 3. Calculate noise (e_noise = y_pred - s_target)
        e_noise = y_pred - s_target

        # 4. Calculate SI-SNR: 10 * log10( ||s_target||^2 / ||e_noise||^2 )
        target_pow = tf.reduce_sum(s_target**2, axis=1) + 1e-8
        noise_pow = tf.reduce_sum(e_noise**2, axis=1) + 1e-8
        
        si_snr = 10.0 * tf.math.log(target_pow / noise_pow) / tf.math.log(10.0)
        
        # We return negative SI-SNR because the optimizer minimizes the loss
        return -tf.reduce_mean(si_snr)