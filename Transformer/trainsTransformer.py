import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

def normalize_landmarks(vec):
    arr = vec.reshape(2, 21, 3)
    origin = arr[0, 0]
    arr -= origin
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr.flatten()

X = np.load('X_news.npy')
X = np.array([normalize_landmarks(x) for x in X])
X = X.reshape(-1, 42, 3)

y = np.load('y_news.npy')
num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        hidden_size = head_size * num_heads
        self.projection = layers.Dense(hidden_size)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(hidden_size)
        ])
        self.dropout2 = layers.Dropout(dropout) 
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        x = self.projection(inputs)
        attn_out = self.att(x, x)
        x = self.norm1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out, training=training))

input_layer = layers.Input(shape=(42, 3))
x = TransformerEncoder(head_size=32, num_heads=4, ff_dim=128, dropout=0.1)(input_layer)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
output_layer = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=input_layer, outputs=output_layer)

optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=32
)

model.save("gesture_transformer_model.h5")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

y_pred_proba = model.predict(X_val)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_true       = np.argmax(y_val,       axis=1)

acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average='macro')
rec   = recall_score(y_true, y_pred, average='macro')
f1    = f1_score(y_true, y_pred, average='macro')

print("\n— Evaluation on Test Set —")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}  (macro‑average)")
print(f"Recall   : {rec:.4f}  (macro‑average)")
print(f"F1‑Score : {f1:.4f}  (macro‑average)\n")

print("Per‑class breakdown:\n")
print(classification_report(y_true, y_pred, digits=4))
