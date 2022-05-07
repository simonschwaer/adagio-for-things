import tensorflow as tf
import numpy as np
import os

# adapted from https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1
def _parse_batch(record_batch, fs, length, label_name):
    feature_description = {
        'audio': tf.io.FixedLenFeature([length], tf.float32),
        label_name: tf.io.FixedLenFeature([1], tf.int64),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)

    return (example['audio'], example[label_name]), example['audio']

def _add_channel_dim(inp):
    audio, material = inp

    audio = tf.expand_dims(audio, axis=1)

    return (audio, material), audio

def _return_for_classifier(inp):
    audio, material = inp

    return audio, tf.cast(material, tf.float32)

def _return_for_classifier_reverse(inp):
    audio, material = inp

    return tf.cast(material, tf.float32), audio

def load_tfrecs(folder='tfrecords', split='train', fs=16000., length=1., epochs=10, batch_size=8,
                add_channel_dim=False, for_classifier=False, reverse_classifier=False, label_name='label'):
    """Load audio data annotated with material class from tfrecord files
    Adapted from https://towardsdatascience.com/how-to-build-efficient-audio-data-pipelines-with-tensorflow-2-0-b3133474c3c1
    """
    
    if split not in ('train', 'test'):
        raise ValueError("split must be either 'train' or 'test'")

    pattern = os.path.join(folder, '{}*.tfrec'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    ds = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    ds = ds.map(lambda x: _parse_batch(x, fs, length, label_name))

    if add_channel_dim:
        ds = ds.map(lambda x, y: _add_channel_dim(x))

    if for_classifier:
        if reverse_classifier:
            ds = ds.map(lambda x, y: _return_for_classifier_reverse(x))
        else:
            ds = ds.map(lambda x, y: _return_for_classifier(x))

    if split == 'train':
        ds = ds.repeat(epochs)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)