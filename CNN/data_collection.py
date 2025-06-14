import os
import cv2
import numpy as np
import mediapipe as mp

# ——— Setup ———
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # faster on Windows
collected_data = []
labels = []

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

def extract_landmarks(results):
    left  = np.zeros((21,3), dtype=np.float32)
    right = np.zeros((21,3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                          dtype=np.float32)
            if handedness.classification[0].label == 'Left':
                left = lm
            else:
                right = lm
    return np.concatenate([left.flatten(), right.flatten()])  # 126-length vector

def normalize_landmarks(vec):
    arr = vec.reshape(2, 21, 3)
    origin = arr[0, 0]   # left wrist
    arr -= origin
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr.flatten()

current_label = 0  # starts at ‘A’

with mp_hands.Hands(max_num_hands=2,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # draw hand landmarks
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # extract & normalize
        raw  = extract_landmarks(results)
        norm = normalize_landmarks(raw)

        # overlay current letter
        letter = chr(current_label + ord('A'))
        cv2.putText(frame, f"Label = {letter}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Dataset Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        # a-z: switch label
        if ord('a') <= key <= ord('z'):
            current_label = key - ord('a')
            print(f"Switched to label: {letter}")

        # SPACE: save sample
        elif key == 32:
            collected_data.append(norm)
            labels.append(current_label)
            print(f"Saved sample for {letter} (total samples: {len(labels)})")

        # ESC: exit loop
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()

# ——— Save NumPy arrays ———
X = np.array(collected_data, dtype=np.float32)
y = np.array(labels, dtype=np.int32)
np.save('X_S.npy', X)
np.save('y_S.npy', y)
print(f"Done! Saved {X.shape[0]} samples → X.npy & y.npy")
