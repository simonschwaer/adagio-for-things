import tensorflow as tf
import tensorflow.keras.layers as tfkl

import numpy as np
import os

from .stft import get_magspec_model, get_inv_magspec_model
from .common import bhattacharyya_distance, audio_loss, VAESampling

def create_models(args, weights=None, weights_path="results"):
    # derive some variables from the args
    M = np.ceil((args.L_fft + 1) / 2).astype(int) - 1 # number of frequency bins in the spectrogram (DC discarded)
    B = np.ceil(args.L / args.L_hop).astype(int) # number of time frames in the spectrogram
    N_e = 16 if args.dataset == "sb" else 55 # number of classes
    Z_e = 1 # dimensionality of material latent space

    audio_pre = get_magspec_model(args.L, args.L_fft, args.L_hop, 
                                  discard_dc=True, squeeze_channel=True, return_phase=True)
    audio_post = get_inv_magspec_model(args.L, args.L_fft, args.L_hop,
                                       discard_dc=True, squeeze_channel=False, expect_channel=False, provide_phase=True)

    # encoder
    spec_in = tf.keras.Input(shape=(B,M,1))
    x = tfkl.Conv2D(args.nfb*1, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(spec_in)
    x = tfkl.MaxPooling2D((1, 4), padding="same")(x)
    x = tfkl.Conv2D(args.nfb*2, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(x)
    x = tfkl.MaxPooling2D((2, 2), padding="same")(x)
    x = tfkl.Conv2D(args.nfb*3, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(x)
    x = tfkl.MaxPooling2D((2, 2), padding="same")(x)
    x = tfkl.Conv2D(args.nfb*4, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(x)
    x = tfkl.MaxPooling2D((2, 2), padding="same")(x)
    x = tfkl.Flatten()(x)
    expl = tfkl.Dense(N_e, activation="softmax")(x)
    if args.use_vae:
        z_mean = tfkl.Dense(args.Z)(x)
        z_log_var = tfkl.Dense(args.Z)(x)
        z = VAESampling()([z_mean, z_log_var])

        encoder = tf.keras.models.Model(spec_in, [expl, z, z_mean, z_log_var], name="encoder")
    else:
        impl = tfkl.Dense(args.Z)(x)
        
        encoder = tf.keras.models.Model(spec_in, [expl, impl], name="encoder")

    # decoder
    startx = int(B/2/2/2)
    starty = int(M/4/2/2/2)

    e_in = tf.keras.Input(shape=(N_e,))
    z_in = tf.keras.Input(shape=(args.Z,))
    y = tfkl.Concatenate()([e_in, z_in])
    y = tfkl.Dense(startx*starty*args.nfb*4)(y)
    y = tfkl.Reshape((startx, starty, args.nfb*4))(y)
    y = tfkl.Conv2DTranspose(args.nfb*3, (args.ks, args.ks), strides=(2, 2),
                             activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(y)
    y = tfkl.Conv2DTranspose(args.nfb*2, (args.ks, args.ks), strides=(2, 2),
                             activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(y)
    y = tfkl.Conv2DTranspose(args.nfb*1, (args.ks, args.ks), strides=(2, 2),
                             activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(y)
    y = tfkl.Conv2DTranspose(1, (args.ks, args.ks), strides=(1, 4), activation="tanh", padding="same")(y)

    decoder = tf.keras.models.Model([e_in, z_in], y, name="decoder")

    # discriminator
    spec_in_dis = tf.keras.Input(shape=(B,M,1))
    e_in_dis = tf.keras.Input(shape=(N_e,))
    d = tfkl.Conv2D(args.nfb*1, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(spec_in_dis)
    d = tfkl.MaxPooling2D((1, 4), padding="same")(d)
    d = tfkl.Conv2D(args.nfb*2, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(d)
    d = tfkl.MaxPooling2D((2, 2), padding="same")(d)
    d = tfkl.Conv2D(args.nfb*3, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(d)
    d = tfkl.MaxPooling2D((2, 2), padding="same")(d)
    d = tfkl.Conv2D(args.nfb*4, (args.ks, args.ks), activation=tfkl.LeakyReLU(alpha=0.1), padding="same")(d)
    d = tfkl.MaxPooling2D((2, 2), padding="same")(d)
    d = tfkl.Flatten()(d)
    d = tfkl.Concatenate()([e_in_dis, d])
    d = tfkl.Dense(128, activation=tfkl.LeakyReLU(alpha=0.1))(d)
    d = tfkl.Dense(1)(d)

    discriminator = tf.keras.models.Model([spec_in_dis, e_in_dis], d, name="discriminator")

    autoencoder = VYAE(
        encoder, decoder, discriminator, audio_pre, audio_post,
        w_ms=args.ms_loss, w_kl=1., w_ec=1., w_re=1., w_ii=1., w_ad=args.ad_loss, num_classes=N_e, use_vae=args.use_vae
    )
    
    if weights is not None:
        discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        autoencoder.load_weights(os.path.join(weights_path, weights, "training.ckpt"))
    
    return encoder, decoder, autoencoder, discriminator, audio_pre, audio_post

class VYAE(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, preproc, postproc,
                 w_ms=1./300, w_kl=1., w_ec=1., w_re=1., w_ii=1., w_ad=1.,
                 num_classes=16, use_input_phase=True, use_vae=True, **kwargs):
        super(VYAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.preproc = preproc
        self.postproc = postproc
        self.discriminator = discriminator

        self.L_tracker = tf.keras.metrics.Mean(name="L")       # total loss
        self.L_ms_tracker = tf.keras.metrics.Mean(name="L_ms") # multiscale spectrogram loss of the first decoding
        self.L_kl_tracker = tf.keras.metrics.Mean(name="L_kl") # KL-divergence of the latent space
    
        self.L_ec_tracker = tf.keras.metrics.Mean(name="L_ec") # loss for classifying the explicit part correctly
        self.L_re_tracker = tf.keras.metrics.Mean(name="L_re") # loss for classifying the randomized explicit part correctly
        self.L_ii_tracker = tf.keras.metrics.Mean(name="L_ii") # loss for reconstructing the implicit part with two different explicits

        self.L_ad_tracker = tf.keras.metrics.Mean(name="L_ad") # adversarial loss for deceiving the discriminator
        self.L_dis_tracker = tf.keras.metrics.Mean(name="L_dis") # loss for discriminating real and generated examples

        self.w_ms = w_ms
        self.w_kl = w_kl
        self.w_ec = w_ec
        self.w_re = w_re
        self.w_ii = w_ii
        self.w_ad = w_ad
        
        self.num_classes = num_classes
        self.use_input_phase = use_input_phase
        self.use_vae = use_vae

        self.warmed_up = False

    @property
    def metrics(self):
        return [
            self.L_tracker,
            self.L_ms_tracker,
            self.L_kl_tracker,

            self.L_ec_tracker,
            self.L_re_tracker,
            self.L_ii_tracker,

            self.L_ad_tracker,
            self.L_dis_tracker,
        ]
    
    def random_generator(self, shape):
        return tf.random.uniform(shape=shape, minval=0, maxval=self.num_classes, dtype=tf.int32)

    def cat2onehot(self, v):
        return tf.squeeze(tf.one_hot(v, self.num_classes), axis=1)

    def prob2onehot(self, v):
        return tf.one_hot(tf.argmax(v, axis=-1), self.num_classes)

    def train_step(self, data):
        inp, out = data
        x, real_e = inp
        if self.use_input_phase:
            x_m, x_p = self.preproc(x)
        else:
            x_m = self.preproc(x)

        with tf.GradientTape() as tape, tf.GradientTape() as dis_tape:
            if self.use_vae:
                e, i, i_mean, i_log_var = self.encoder(x_m)
            else:
                e, i = self.encoder(x_m)
            y_m = self.decoder([self.cat2onehot(real_e), i])
            if self.use_input_phase:
                y = self.postproc([y_m, x_p])
            else:
                y = self.postproc([y_m])
            
            # Y-AE
            rand_e = self.random_generator(tf.shape(real_e))
            l_y_m = self.decoder([self.cat2onehot(rand_e), i])
            if self.use_vae:
                l_e, _, l_i_mean, l_i_log_var = self.encoder(l_y_m) # left branch: reencoding of random gesture decoding
                _, _, r_i_mean, r_i_log_var = self.encoder(y_m) # right  branch: reencoding of original decoded signal
            else:
                l_e, l_i = self.encoder(l_y_m) # left branch: reencoding of random gesture decoding
                _, r_i = self.encoder(y_m) # right  branch: reencoding of original decoded signal
            
            L_ms = 0
            if self.w_ms != 0:
                # multiscale spectrogram loss on fullband signal
                L_ms += audio_loss(
                    x, y,
                    fft_sizes=[(2048, 1024), (1024, 256), (512, 128)], w_mag=0.0, w_logmag=1.0
                )

            L_kl = 0
            if self.use_vae and self.w_kl != 0: # and not self.warmed_up:
                L_kl = -0.5 * (1 + i_log_var - tf.square(i_mean) - tf.exp(i_log_var))
                L_kl = tf.reduce_mean(tf.reduce_sum(L_kl, axis=1))
                
            L_ec = 0
            if self.w_ec != 0: # and not self.warmed_up:
                L_ec = tf.losses.sparse_categorical_crossentropy(real_e, e, axis=-1)
                
            L_re = 0
            if self.w_re != 0:
                L_re = tf.losses.sparse_categorical_crossentropy(rand_e, l_e, axis=-1)

            L_ii = 0
            if self.w_ii != 0:
                if self.use_vae:
                    L_ii = tf.reduce_mean(
                        bhattacharyya_distance(r_i_mean, l_i_mean, tf.exp(r_i_log_var), tf.exp(l_i_log_var))
                    )
                else:
                    L_ii = tf.reduce_mean(tf.reduce_sum(
                        (i - r_i)**2, axis=-1
                    ))

            L_dis = 0
            L_ad = 0
            if self.warmed_up:
                # run discriminator on real and generated examples
                pred_x = self.discriminator([x_m, self.cat2onehot(real_e)]) # y_m
                pred_l_y = self.discriminator([l_y_m, self.cat2onehot(rand_e)])

                # calculate loss for discriminator into L_dis
                L_dis += tf.losses.binary_crossentropy(tf.ones_like(pred_x), pred_x, from_logits=True)
                L_dis += tf.losses.binary_crossentropy(tf.zeros_like(pred_l_y), pred_l_y, from_logits=True)

                # add adversarial decoder loss into L
                L_ad = tf.losses.binary_crossentropy(tf.ones_like(pred_l_y), pred_l_y, from_logits=True)

            L = self.w_ms * L_ms\
              + self.w_kl * L_kl\
              + self.w_ec * L_ec\
              + self.w_re * L_re\
              + self.w_ii * L_ii\
              + self.w_ad * L_ad

        grads = tape.gradient(L, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

        if self.warmed_up:
            dis_grads = dis_tape.gradient(L_dis, self.discriminator.trainable_weights)
            self.discriminator.optimizer.apply_gradients(zip(dis_grads, self.discriminator.trainable_weights))

        self.L_tracker.update_state(L)
        self.L_ms_tracker.update_state(L_ms)
        self.L_kl_tracker.update_state(L_kl)

        self.L_ec_tracker.update_state(L_ec)
        self.L_re_tracker.update_state(L_re)
        self.L_ii_tracker.update_state(L_ii)

        self.L_ad_tracker.update_state(L_ad)
        self.L_dis_tracker.update_state(L_dis)

        return {
            "L": self.L_tracker.result(),
            "L_ms": self.L_ms_tracker.result(),
            "L_kl": self.L_kl_tracker.result(),
            
            "L_ec": self.L_ec_tracker.result(),
            "L_re": self.L_re_tracker.result(),
            "L_ii": self.L_ii_tracker.result(),

            "L_ad": self.L_ad_tracker.result(),
            "L_dis": self.L_dis_tracker.result(),
        }
    
    def call(self, inp, training=False):
        e,i,_,_ = self.encoder(inp)
        return self.decoder([self.prob2onehot(e),i])
