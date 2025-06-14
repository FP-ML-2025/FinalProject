import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# Positional Encoding
def get_positional_encoding(seq_len, d_model):
  angle_rads = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis] / tf.pow(
      10000.0,
      (2 * (tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] // 2)) / tf.cast(d_model, tf.float32)
  )
  sines = tf.sin(angle_rads[:, 0::2])
  coses = tf.cos(angle_rads[:, 1::2])
  pos_encoding = tf.concat([sines, coses], axis=-1)
  return pos_encoding[tf.newaxis, ...]  # (1, seq_len, d_model)

# Transformer Encoder block
def transformer_encoder_block(d_model, num_heads, ff_dim, dropout=0.1):
  inputs = layers.Input(shape=(None, d_model))
  attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
  attn_output = layers.Dropout(dropout)(attn_output)
  out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
  ffn = layers.Dense(ff_dim, activation='relu')(out1)
  ffn = layers.Dense(d_model)(ffn)
  ffn_output = layers.Dropout(dropout)(ffn)
  out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
  return models.Model(inputs=inputs, outputs=out2)

# Transformer Decoder block
def transformer_decoder_block(d_model, num_heads, ff_dim, dropout=0.1):
  inputs = layers.Input(shape=(None, d_model))
  enc_outputs = layers.Input(shape=(None, d_model))
  attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
  attn1 = layers.LayerNormalization(epsilon=1e-6)(attn1 + inputs)
  attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(attn1, enc_outputs)
  attn2 = layers.Dropout(dropout)(attn2)
  out1 = layers.LayerNormalization(epsilon=1e-6)(attn2 + attn1)
  ffn = layers.Dense(ff_dim, activation='relu')(out1)
  ffn = layers.Dense(d_model)(ffn)
  ffn_output = layers.Dropout(dropout)(ffn)
  out2 = layers.LayerNormalization(epsilon=1e-6)(ffn_output + out1)
  return models.Model(inputs=[inputs, enc_outputs], outputs=out2)

# Layer to create a learnable query and expand to batch size\ nclass QueryExpand(layers.Layer):
  def __init__(self, d_model, **kwargs):
      super().__init__(**kwargs)
      self.d_model = d_model
      self.query = self.add_weight(
          shape=(1, 1, d_model),
          initializer='random_normal',
          trainable=True,
          name='learnable_query'
      )
  def call(self, enc_outputs):
      batch_size = tf.shape(enc_outputs)[0]
      # tile query to (batch_size, 1, d_model)
      return tf.tile(self.query, [batch_size, 1, 1])

# Build the full Transformer model
def build_transformer_model(
    seq_len,
    feature_dim,
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_enc_layers=2,
    num_dec_layers=1,
    num_classes=26,
    dropout=0.1
):
    enc_inputs = layers.Input(shape=(seq_len, feature_dim), name='encoder_input')
    x = layers.Dense(d_model)(enc_inputs)
    pos_enc = get_positional_encoding(seq_len, d_model)
    x = x + pos_enc

    # Encoder stack
    for _ in range(num_enc_layers):
        x = transformer_encoder_block(d_model, num_heads, ff_dim, dropout)(x)
    enc_outputs = x

    # Decoder stack with learnable query
    query_layer = QueryExpand(d_model)
    y = query_layer(enc_outputs)
    for _ in range(num_dec_layers):
        y = transformer_decoder_block(d_model, num_heads, ff_dim, dropout)([y, enc_outputs])

    # Classification head
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dropout(dropout)(y)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(y)

    model = models.Model(inputs=enc_inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Load dataset ===
X = np.load('X_T.npy')  # shape: (samples, feature_dim) or (samples, seq_len, feature_dim)
y = np.load('y_T.npy')  # shape: (samples,)

# Reshape single-frame data to sequences
if X.ndim == 2:
    seq_len = 1
    feature_dim = X.shape[1]
    X = X.reshape(-1, seq_len, feature_dim)
else:
    seq_len = X.shape[1]
    feature_dim = X.shape[2]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
num_classes = len(np.unique(y))

# Build & train
model = build_transformer_model(
    seq_len=seq_len,
    feature_dim=feature_dim,
    d_model=128,
    num_heads=8,
    ff_dim=256,
    num_enc_layers=4,
    num_dec_layers=1,
    num_classes=num_classes,
    dropout=0.1
)
model.summary()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)
model.save('sign_lang_transformer_model.h5')
