import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('cnn_hand_sign_model3.h5')

# Setup MediaPipe Hands
mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
hands      = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def extract_landmarks(results):
    left  = np.zeros((21,3), dtype=np.float32)
    right = np.zeros((21,3), dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm_list, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
            if handedness.classification[0].label == 'Left':
                left = coords
            else:
                right = coords
    return np.concatenate([left.flatten(), right.flatten()])

def normalize_landmarks(vec):
    arr = vec.reshape(2,21,3)
    origin = arr[0,0]
    arr -= origin
    m = np.max(np.abs(arr))
    if m>0: arr /= m
    return arr.flatten()

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


current_letter = ""
last_confirmed = ""
letter_start_time = 0
constructed_text = ""

letter_conf_sum = 0
letter_conf_count = 0
letter_timer_start = 0

with hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        now = time.time()

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            raw = extract_landmarks(res)
            norm = normalize_landmarks(raw)
            inp = norm.reshape(1, 21, 6, 1).astype(np.float32)
            preds = model.predict(inp, verbose=0)
            idx = int(np.argmax(preds[0]))
            conf = preds[0][idx]
            letter = chr(idx + ord('A'))

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


            # Display current prediction
            cv2.putText(frame, f"{letter} {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show constructed text
        cv2.putText(frame, f"Text: {constructed_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Handle keyboard input for space and delete
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            constructed_text += ' '
            last_confirmed = ''
        elif key == 8:  # BACKSPACE
            constructed_text = constructed_text[:-1]
            last_confirmed = ''

        cv2.imshow('CNN Word Builder', frame)

cap.release()
cv2.destroyAllWindows()

