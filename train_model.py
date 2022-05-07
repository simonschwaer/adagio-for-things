import tensorflow as tf
import numpy as np

import pickle
import os
import datetime
import argparse

from vyae import create_models, load_tfrecs, str2bool


print("Tensorflow version: %s" % (tf.__version__))

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, dest="batch_size", default=8)
parser.add_argument("-a", "--adverse-epochs", type=int, dest="adverse_epochs", default=70)
parser.add_argument("-w", "--warmup-epochs", type=int, dest="warmup_epochs", default=0)
parser.add_argument("-l", "--learning-rate", type=float, dest="learning_rate", default=1e-4)
parser.add_argument("-f", "--num-filters-base", type=int, dest="nfb", default=64)
parser.add_argument("-k", "--kernel-size", type=int, dest="ks", default=5)
parser.add_argument("-m", "--ms-loss", type=float, dest="ms_loss", default=1./300)
parser.add_argument("-i", "--ad-loss", type=float, dest="ad_loss", default=1.)
parser.add_argument("-d", "--dataset", type=str, dest="dataset", default="sb")
parser.add_argument("-e", "--freeze-enc", type=str2bool, dest="freeze_encoder", default=False)
parser.add_argument("-v", "--use-vae", type=str2bool, dest="use_vae", default=True)

# usually fixed settings
parser.add_argument("--sampling-rate", type=float, dest="fs", default=16000.)
parser.add_argument("--excerpt-length", type=int, dest="L", default=8192)
parser.add_argument("--fft-size", type=int, dest="L_fft", default=1024)
parser.add_argument("--hop-size", type=int, dest="L_hop", default=256)
parser.add_argument("--impl-dim", type=int, dest="Z", default=32)

args = parser.parse_args()
print("Arguments:", args)

starttime = datetime.datetime.now()

datafolder = 'data/spoonbowl' if args.dataset == "sb" else 'data/footsteps'
split = 'train' # 'test'

data = load_tfrecs(folder=datafolder, split=split, fs=args.fs, length=args.L, batch_size=args.batch_size,
                   epochs=args.warmup_epochs+args.adverse_epochs, add_channel_dim=True)

# calculate number of data samples from file name
file_counts = [int(f.split("-")[-1][:-6]) for f in os.listdir(datafolder) \
               if os.path.isfile(os.path.join(datafolder, f)) and f[-6:] == ".tfrec" and f[:len(split)] == split]
num_batches = np.ceil(sum(file_counts)/args.batch_size).astype(int)

encoder, decoder, autoencoder, discriminator, audio_pre, audio_post = create_models(args, weights=None)

encoder.summary()
decoder.summary()
discriminator.summary()

run_id = starttime.strftime('%Y%m%d_%H%M%S')
filename = os.path.splitext(os.path.basename(__file__))[0]
print("Training run ID: %s_%s" % (run_id, filename))

checkpoint_path = "./results/%s_%s/training.ckpt" % (run_id, filename)
checkpoint_dir = os.path.dirname(checkpoint_path)

# create a callback that saves the weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

try: # in try/catch block to save weights when training is interrupted with ctrl+c
    if args.warmup_epochs > 0:
        discriminator.trainable = False
        # discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        history_wrm = autoencoder.fit(
            data,
            epochs=args.warmup_epochs,
            steps_per_epoch=num_batches,
            callbacks=[cp_callback]
        )

    if args.adverse_epochs > 0:

        autoencoder.warmed_up = True
        discriminator.trainable = True
        encoder.trainable = (not args.freeze_encoder)

        discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        history_adv = autoencoder.fit(
            data,
            epochs=args.adverse_epochs,
            steps_per_epoch=num_batches,
            callbacks=[cp_callback]
        )
    with open("./results/%s_%s/info.dict" % (run_id, filename), "wb") as pickle_file:
        to_save = {
            'args': args
        }
        if args.warmup_epochs > 0:
            to_save['warmup_history'] = history_wrm.history
        if args.adverse_epochs > 0:
            to_save['adverse_history'] = history_adv.history

        pickle.dump(to_save, pickle_file)
except KeyboardInterrupt:
    save_path = "./results/%s_%s/abort.ckpt" % (run_id, filename)
    autoencoder.save_weights(save_path)
    print("\nProcess aborted, weights saved in \"" + save_path + "\"")
    exit()

endtime = datetime.datetime.now()

print("\nDone! Process took %.2f minutes..." % ((endtime - starttime).total_seconds() / 60.0))
