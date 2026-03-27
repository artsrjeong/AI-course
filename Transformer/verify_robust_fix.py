import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)
    self.add = tf.keras.layers.Add()

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
    return self.add([inputs, self.pos_encoding[:, :tf.shape(inputs)[1], :]])

  def compute_output_shape(self, input_shape):
    return input_shape

def transformer(vocab_size, d_model):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings = tf.keras.layers.Lambda(
    lambda x: x * tf.math.sqrt(tf.cast(d_model, tf.float32))
  )(embeddings)
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  return tf.keras.Model(inputs=inputs, outputs=embeddings)

try:
    print("Building model with robust PositionalEncoding...")
    model = transformer(9000, 128)
    print("Model built successfully!")
    
    print("\nTesting call with dummy data...")
    dummy_input = tf.zeros((1, 40))
    output = model(dummy_input)
    print(f"Call successful! Output shape: {output.shape}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
