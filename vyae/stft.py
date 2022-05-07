import tensorflow as tf
import numpy as np


def _magnitude_to_db(x, amin=1e-5, dynamic_range=96):
    # adapted from kapre: https://github.com/keunwoochoi/kapre

    def _log10(x):
        return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

    if tf.keras.backend.ndim(x) > 1:  # we assume x is batch in this case
        max_axis = tuple(range(tf.keras.backend.ndim(x))[1:])
    else:
        max_axis = None

    log_spec = 10.0 * _log10(tf.math.maximum(x, amin))
    log_spec = tf.math.maximum(
        log_spec, tf.math.reduce_max(log_spec, axis=max_axis, keepdims=True) - dynamic_range
    )

    return log_spec

def get_stft_model(L, L_fft, L_hop, discard_dc=False, use_if=False, squeeze_channel=False, name="stft"):
    """Create a Keras Model with no learnable weights that outputs an STFT of an input signal

    Parameters
    ----------
    L: int
        length of input audio signal in samples
    L_fft: int
        fft window size of the STFT
    L_fft: int
        hop size of the STFT
    discard_dc: bool
        whether or not to remove the first Fourier coefficient of each spectrum in the STFT (default False)
    use_if: bool
        whether or not to calculate the instantaneous frequency (IF) from the phase component (default False)
    squeeze_channel: bool
        whether or not to remove the channel dimension from the output.
        If True, the model returns one spectrogram per channel (default False)
    name: str
        Human-readable name of the created Keras model (for debugging)

    Returns
    -------
    model: tf.keras.models.Model
        Keras Model with audio input (B, C, L) and spectrogram output
        (B = batch, [C = channels], M = time frames, K = frequency bins, 2 = magnitude/phase)
    """

    audio_in = tf.keras.layers.Input(shape=(1, L))
    spec = tf.signal.stft(audio_in, L_fft, L_hop, L_fft, pad_end=True)

    if discard_dc:
        spec = spec[:,:,:,1:]
    spec = spec[:,:,:,:,tf.newaxis] # add axis to concatenate magnitude and phase later
    
    mag = tf.abs(spec)
    logmag = _magnitude_to_db(mag) / 96 # guaranteed in range (-1, 1)
    phs = tf.math.angle(spec)

    if use_if:
        # calculate instantaneous frequency from phase angle
        # adapted from https://github.com/magenta/magenta/blob/main/magenta/models/gansynth
        instf = phs[:,:,1:,:,:] - phs[:,:,:-1,:,:]
        instf = tf.where(instf > np.pi, instf - 2 * np.pi, instf)
        instf = tf.where(instf < -np.pi, instf + 2 * np.pi, instf)
        # add initial phase
        instf = tf.concat([phs[:,:,0:1,:], instf], axis=-3) / np.pi # now normalized between -1 and 1

        res = tf.concat([logmag, instf], axis=-1)
    else:
        phs /= np.pi
        res = tf.concat([logmag, phs], axis=-1)

    if squeeze_channel:
        res = tf.squeeze(res, axis=1)

    return tf.keras.models.Model(audio_in, res, name=name)


def get_inv_stft_model(L, L_fft, L_hop, discard_dc=False, use_if=False, expect_channel=True, squeeze_channel=False, name="inv_stft"):
    """Create a Keras Model with no learnable weights that outputs
    a time domain signal from spectrogram input

    Parameters
    ----------
    L: int
        length of input audio signal in samples
    L_fft: int
        fft window size of the STFT
    L_fft: int
        hop size of the STFT
    discard_dc: bool
        whether or not the first Fourier coefficient of each spectrum is discarded in the STFT input (default False)
    use_if: bool
        whether or not the instantaneous frequency (IF) was calculated from the phase component (default False)
    expect_channel: bool
        whether or not to the input contains a channel dimension (default True)
    squeeze_channel: bool
        whether or not to remove the channel dimension from the output.
        If True, the model returns a multi-channel signal (default False)
    name: str
        Human-readable name of the created Keras model (for debugging)

    Returns
    -------
    model: tf.keras.models.Model
        Keras Model with spectrogram input (B = batch, [C = channels], M = time frames, K = frequency bins, 2 = magnitude/phase)
        and time domain audio output (B, [C], L)
        
    """
    M = np.ceil((L_fft + 1) / 2).astype(int) # number of frequency bins in the spectrogram
    B = np.ceil(L / L_hop).astype(int) # number of time frames in the spectrogram

    M_in = M-1 if discard_dc else M
    if expect_channel:
        spec_in = tf.keras.Input(shape=(1, B, M_in, 2))
        spec = spec_in
    else:
        spec_in = tf.keras.Input(shape=(B, M_in, 2))
        spec = tf.expand_dims(spec_in, axis=1)

    # invert magnitude/phase decomposition and scaling in preprocessing
    mag = tf.pow(10., spec[:,:,:,:,0]*9.6)
    if use_if:
        phs = tf.cumsum(spec[:,:,:,:,1] * np.pi, axis=-2)
    else:
        phs = spec[:,:,:,:,1] * np.pi

    cspec = tf.complex(mag * tf.math.cos(phs), mag * tf.math.sin(phs))
    if discard_dc:
        cspec = tf.pad(cspec, [[0, 0], [0, 0], [0, 0], [1, 0]]) # add a zero DC component
    audio_out = tf.signal.inverse_stft(cspec, L_fft, L_hop, fft_length=L_fft)
    audio_out = audio_out[:,:,:L] # crop samples at the end

    if squeeze_channel:
        audio_out = tf.squeeze(audio_out, axis=1)

    return tf.keras.models.Model(spec_in, audio_out, name=name)

def get_magspec_model(L, L_fft, L_hop, discard_dc=False, squeeze_channel=False, return_phase=False, name="magspec"):
    """Create a Keras Model with no learnable weights that outputs
    the magnitude spectrogram of an input signal

    Parameters
    ----------
    L: int
        length of input audio signal in samples
    L_fft: int
        fft window size of the STFT
    L_fft: int
        hop size of the STFT
    discard_dc: bool
        whether or not to remove the first Fourier coefficient of each spectrum in the STFT (default False)
    squeeze_channel: bool
        whether or not to remove the channel dimension from the output.
        If True, the model returns one spectrogram per channel (default False)
    return_phase: bool
        whether or not the model should return the phase of the input as a separate tf.tensor
    name: str
        Human-readable name of the created Keras model (for debugging)

    Returns
    -------
    model: tf.keras.models.Model
        Keras Model with audio input (B, C, L) and spectrogram (and optional phase) output
        (B = batch, [C = channels], M = time frames, K = frequency bins)
        normalized to the range [-1, 1]
    """
    audio_in = tf.keras.layers.Input(shape=(1, L))
    spec = tf.signal.stft(audio_in, L_fft, L_hop, L_fft, pad_end=True)

    if discard_dc:
        spec = spec[:,:,:,1:]
    spec = spec[:,:,:,:,tf.newaxis] # add axis for representing magnitude (second layer would e.g. be phase)

    if squeeze_channel:
        spec = tf.squeeze(spec, axis=1)

    mspec = tf.abs(spec)
    mspec = _magnitude_to_db(mspec) / 96 # guaranteed in range (-1, 1)

    if return_phase:
        phs = tf.math.angle(spec)
        return tf.keras.models.Model(audio_in, [mspec, phs], name=name)
    else:
        return tf.keras.models.Model(audio_in, mspec, name=name)


def get_inv_magspec_model(L, L_fft, L_hop, discard_dc=False, expect_channel=True, squeeze_channel=False, provide_phase=False,
                          phase="random", gl_iter=50, name="inv_magspec"):
    """Create a Keras Model with no learnable weights that outputs
    a time domain signal from magnitude spectrogram input

    Parameters
    ----------
    L: int
        length of input audio signal in samples
    L_fft: int
        fft window size of the STFT
    L_fft: int
        hop size of the STFT
    discard_dc: bool
        whether or not the first Fourier coefficient of each spectrum is discarded in the STFT input (default False)
    expect_channel: bool
        whether or not to the input contains a channel dimension (default True)
    squeeze_channel: bool
        whether or not to remove the channel dimension from the output.
        If True, the model returns a multi-channel signal (default False)
    provide_phase: bool
        whether or not the phase component to be used is given as a Model input (default False)
    phase: str
        if provide_phase == False, this option selects the phase reconstruction method
        (current options: "random" and "gl")
    gl_iter: int
        number of Griffin-Lim iterations if phase == "gl" (default 50)
    name: str
        Human-readable name of the created Keras model (for debugging)

    Returns
    -------
    model: tf.keras.models.Model
        Keras Model with normalized magnitude spectrogram input
        (B = batch, [C = channels], M = time frames, K = frequency bins)
        plus optional phase input (B = batch, [C = channels], M = time frames, K = frequency bins)
        and time domain audio output (B, [C], L)
        
    """
    M = np.ceil((L_fft + 1) / 2).astype(int) # number of frequency bins in the spectrogram
    B = np.ceil(L / L_hop).astype(int) # number of time frames in the spectrogram

    M_in = M-1 if discard_dc else M
    if expect_channel:
        spec_in = tf.keras.Input(shape=(1, B, M_in))
        spec = spec_in
    else:
        spec_in = tf.keras.Input(shape=(B, M_in))
        spec = tf.expand_dims(spec_in, axis=1)

    # invert magnitude/phase decomposition and scaling in preprocessing
    mag = tf.pow(10., spec*9.6)

    if provide_phase:
        if expect_channel:
            phs_in = tf.keras.Input(shape=(1, B, M_in))
            phs = phs_in
        else:
            phs_in = tf.keras.Input(shape=(B, M_in))
            phs = tf.expand_dims(phs_in, axis=1)
        cspec = tf.complex(mag * tf.math.cos(phs), mag * tf.math.sin(phs))
        if discard_dc:
            cspec = tf.pad(cspec, [[0, 0], [0, 0], [0, 0], [1, 0]]) # add a zero DC component

    else:
        if phase == "random":
            phs = tf.random.uniform(tf.shape(mag), minval=-np.pi, maxval=np.pi)
            cspec = tf.complex(mag * tf.math.cos(phs), mag * tf.math.sin(phs))
            if discard_dc:
                cspec = tf.pad(cspec, [[0, 0], [0, 0], [0, 0], [1, 0]]) # add a zero DC component
        elif phase == "gl":
            cspec = tf.cast(mag, dtype=tf.complex64)
            if discard_dc:
                cspec = tf.pad(cspec, [[0, 0], [0, 0], [0, 0], [1, 0]]) # add a zero DC component

            S = tf.identity(cspec)
            for i in range(gl_iter):
                x = tf.signal.inverse_stft(S, L_fft, L_hop, L_fft)
                E = tf.signal.stft(x, L_fft, L_hop, L_fft, pad_end=False)
                phs = E / tf.cast(tf.maximum(1e-8, tf.abs(E)), tf.complex64)
                S = cspec * phs
            cspec = S
        else:
            raise NotImplementedError("Only \"random\" and \"gl\" (Griffin-Lim) are currently implemented for phase reconstruction.")

    audio_out = tf.signal.inverse_stft(cspec, L_fft, L_hop, fft_length=L_fft)
    audio_out = audio_out[:,:,:L] # crop samples at the end

    if squeeze_channel:
        audio_out = tf.squeeze(audio_out, axis=1)

    if provide_phase:
        return tf.keras.models.Model([spec_in, phs_in], audio_out, name=name)
    else:
        return tf.keras.models.Model(spec_in, audio_out, name=name)
