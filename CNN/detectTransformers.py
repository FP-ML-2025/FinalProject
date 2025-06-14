import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Define your custom TokenEmbedding layer
class TokenEmbedding(Layer):
    def _init_(self, vocab_size, d_model):
        super(TokenEmbedding, self)._init_()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        return self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

# Register custom object and load the model
custom_objects = {'TokenEmbedding': TokenEmbedding}
model = load_model('gesture_transformer_model.h5', compile=False, custom_objects=custom_objects)

# Constants
VOCAB_SIZE = 26  # A-Z
BOS_TOKEN = VOCAB_SIZE  # Start of sequence token

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def extract_landmarks(results):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm_list, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
            if handedness.classification[0].label == 'Left':
                left = coords
            else:
                right = coords
    return np.concatenate([left.flatten(), right.flatten()])

def normalize_landmarks(vec):
    arr = vec.reshape(2, 21, 3)
    origin = arr[0, 0]
    arr -= origin
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr.flatten()

# Open webcam
cap = cv2.VideoCapture(0)
with hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            raw = extract_landmarks(res)
            norm = normalize_landmarks(raw)
            inp = norm.reshape(1, 126).astype(np.float32)
            dec_input = np.array([[BOS_TOKEN]], dtype=np.int32)

            preds = model.predict([inp, dec_input], verbose=0)
            idx = int(np.argmax(preds[0, 0]))
            conf = float(preds[0, 0, idx])
            letter = chr(idx + ord('A'))

            cv2.putText(frame, f"{letter} {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Transformer Predictor', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()