import tensorflow as tf

def str2bool(v):
    """Convert a string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

class VAESampling(tf.keras.layers.Layer):
    """Variational autoencoder sampling from the distribution parameters given by the encoding.
    Adapted from the Keras example: https://keras.io/examples/generative/vae/
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def bhattacharyya_distance(mu1, mu2, om1, om2):
    """Bhattacharyya distance between two multivariate Gaussian distributions with diagonal covariance

    Parameters
    ----------
    mu1: tf.tensor (B, N)
         mean of first N-dimensional Gaussian
    mu2: tf.tensor (B, N)
         mean of second N-dimensional Gaussian
    om1: tf.tensor (B, N)
         diagonal covariance values of first Gaussian
    om2: tf.tensor (B, N)
         diagonal covariance values of second Gaussian

    Returns
    -------
    distance: tf.tensor (B)
        Bhattacharyya distance between the two input distributions
    """
    dmu = mu1 - mu2
    om = (om1 + om2) / 2
    a = tf.reduce_sum(tf.square(dmu)/om, axis=-1)
    b = tf.math.log(tf.reduce_sum(om, axis=-1)/tf.sqrt(tf.reduce_sum(om1, axis=-1)*tf.reduce_sum(om2, axis=-1)))
    return a/8 + b/2


def audio_loss(x_ref, x_smp, fft_sizes=[(2048, 1024), (1024, 512), (512, 256)], 
               p_mag=2.0, p_logmag=1.0, w_mag=1.0, w_logmag=0.0):
    """A multiscale spectrogram (MSS) loss between two time domain audio excerpts,
    heavily inspired by DDSP: https://github.com/magenta/ddsp

    Parameters
    ----------
    x_ref: tf.tensor (B, 1, L)
        single-channel time domain audio reference with length L samples
    x_smp: tf.tensor (B, 1, L)
        single-channel time domain audio output with length L samples
    fft_sizes: list of 2-tuples (int, int)
        list of pairs (N, H) where N is the fft size and H is the hop size of an STFT
    p_mag: float
        exponent of the norm comparing the magnitude spectrograms (>= 1)
    p_logmag: float
        exponent of the norm comparing the log-magnitude spectrograms (>= 1)
    w_mag: float
        weighting of the magnitude spectrogram loss in the returned total loss (default 1)
    w_logmag: float
        weighting of the log-magnitude spectrogram loss in the returned total loss (default 0)
    
    Returns
    -------
    loss: tf.tensor (B)
        combined MSS loss of spectrogram and log-spectrogram
    """
    L = L_mag = L_logmag = 0.0

    for N, H in fft_sizes:
        mag_ref = tf.abs(tf.signal.stft(x_ref, N, H, N, pad_end=True))
        mag_smp = tf.abs(tf.signal.stft(x_smp, N, H, N, pad_end=True))
            
        if w_mag > 0:
            L_mag = w_mag * tf.reduce_mean(
                tf.reduce_sum((mag_ref - mag_smp)**p_mag, axis=(-2, -1))
            )
            L += L_mag

        if w_logmag > 0:
            logmag_ref = tf.math.log(mag_ref + 1e-7)
            logmag_smp = tf.math.log(mag_smp + 1e-7)

            L_logmag = w_logmag * tf.reduce_mean(
                tf.reduce_sum(tf.abs(logmag_ref - logmag_smp)**p_logmag, axis=(-2, -1))
            )
            L += L_logmag

    return L