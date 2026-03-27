import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.stack([sines, cosines], axis=-1)
    pos_encoding = tf.reshape(pos_encoding, [position, d_model])
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Verification
try:
    print("Testing PositionalEncoding with matching shape...")
    pe = PositionalEncoding(50, 128)
    inputs = tf.random.uniform((1, 50, 128))
    outputs = pe(inputs)
    print(f"Success! Output shape: {outputs.shape}")

    print("\nTesting PositionalEncoding with smaller sequence length...")
    inputs_small = tf.random.uniform((1, 10, 128))
    outputs_small = pe(inputs_small)
    print(f"Success! Output shape: {outputs_small.shape}")

    print("\nVerification complete!")
except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
