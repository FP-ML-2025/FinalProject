import cv2
import numpy as np
import mediapipe as mp

# Import your custom model class
from cnnfromscratch import NeuralNetwork  # replace 'your_model_file' with your actual filename

# Load the trained custom model
model = NeuralNetwork.load("sign_language_model.h5")

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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
            inp = norm.reshape(1, -1)  # shape (1, 126)

            probs = model.forwardPropagation(inp)[0][-1]  # softmax output
            idx = np.argmax(probs)
            confidence = probs[0, idx]  # assuming shape is (1, 26)

            letter = chr(idx + ord('A'))
            text = f"{letter} ({confidence * 100:.1f}%)"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # red color


        cv2.imshow('Custom Neural Net Predictor', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
