import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers
import time

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

model = tf.keras.models.load_model(
    'gesture_transformer_model.h5',
    custom_objects={'TransformerEncoder': TransformerEncoder}
)

num_classes = model.output_shape[-1]
class_names = [chr(i) for i in range(ord('A'), ord('A') + num_classes)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def extract_landmarks(results):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            if handed.classification[0].label == 'Left':
                left = pts
            else:
                right = pts
    return np.concatenate([left.flatten(), right.flatten()])

def normalize_landmarks(vec):
    arr = vec.reshape(2, 21, 3)
    origin = arr[0, 0]
    arr -= origin
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr.flatten()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

current_letter = ""
last_confirmed = ""
letter_start_time = 0
constructed_text = ""

letter_conf_sum = 0
letter_conf_count = 0
letter_timer_start = 0

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            raw = extract_landmarks(results)
            norm = normalize_landmarks(raw)
            inp = norm.reshape(1, 42, 3)

            preds = model.predict(inp, verbose=0)[0]
            idx = np.argmax(preds)
            conf = preds[idx]
            letter = class_names[idx]

            if letter != current_letter:
                current_letter = letter
                letter_start_time = now
                letter_conf_sum = conf
                letter_conf_count = 1
                letter_timer_start = now
            else:
                letter_conf_sum += conf
                letter_conf_count += 1

                if now - letter_start_time >= 1 and letter != last_confirmed:
                    avg_conf = letter_conf_sum / letter_conf_count
                    if avg_conf >= 0.85:
                        constructed_text += letter
                        last_confirmed = letter
                        letter_duration = now - letter_timer_start
                        print(f"Added letter: {letter} | Avg accuracy: {avg_conf:.4f} | Time: {letter_duration:.2f} sec")
                    else:
                        print(f"Ignored letter: {letter} | Avg accuracy: {avg_conf:.4f}")

            cv2.putText(frame, f"{letter} ({conf:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.putText(frame, f"Text: {constructed_text}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            constructed_text += ' '
            last_confirmed = ''
        elif key == 8:
            constructed_text = constructed_text[:-1]
            last_confirmed = ''

        cv2.imshow('Semi-Transformer Sign Language Interpreter', frame)

cap.release()
cv2.destroyAllWindows()
