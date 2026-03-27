import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs
import tensorflow as tf
import numpy as np
import traceback

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

    angle_rads = np.zeros(angle_rads.shape)
    angle_rads[:, 0::2] = sines
    angle_rads[:, 1::2] = cosines
    pos_encoding = tf.constant(angle_rads)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    try:
        slice_idx = tf.shape(inputs)[1]
        sliced_pos = self.pos_encoding[:, :slice_idx, :]
        return inputs + sliced_pos
    except Exception as e:
        print(f"Error in PositionalEncoding.call: {e}")
        print(f"Inputs shape: {tf.shape(inputs)}")
        print(f"PosEncoding shape: {tf.shape(self.pos_encoding)}")
        raise e

# Test
try:
    print("Testing PositionalEncoding initialization...")
    pe = PositionalEncoding(50, 128)
    print(f"PosEncoding initialized with shape: {pe.pos_encoding.shape}")
    
    print("\nTesting call with matching shape...")
    inputs = tf.random.uniform((1, 50, 128))
    outputs = pe(inputs)
    print(f"Success! Output shape: {outputs.shape}")

    print("\nTesting call with smaller seq_len...")
    inputs_small = tf.random.uniform((1, 10, 128))
    outputs_small = pe(inputs_small)
    print(f"Success! Output shape: {outputs_small.shape}")

    print("\nTesting call with larger seq_len...")
    inputs_large = tf.random.uniform((1, 100, 128))
    outputs_large = pe(inputs_large)
    print(f"Output lead to: {outputs_large.shape}")

except Exception:
    traceback.print_exc()
