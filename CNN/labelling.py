import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm  # for progress bar

# --- CONFIG ---
DATASET_DIR = 'dataset'        # root folder
MAX_HANDS    = 2               # expecting up to 2 hands
IMAGE_EXT    = ('.jpg','.png')
OUT_X = 'X.npy'
OUT_Y = 'y.npy'

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5
)

def extract_landmark_vector(image):
    """Returns a 126-d vector (left + right) or None if detection fails."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    # init zero arrays
    left = np.zeros((21,3), dtype=np.float32)
    right = np.zeros((21,3), dtype=np.float32)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None
    for lm, handness in zip(res.multi_hand_landmarks, res.multi_handedness):
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        label = handness.classification[0].label  # 'Left' or 'Right'
        if label == 'Left':
            left = coords
        else:
            right = coords
    # flatten & concat → (126,)
    return np.concatenate([left.flatten(), right.flatten()])

def normalize(X):
    """Zero-mean, unit-std normalization per feature."""
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std

# --- LOAD & PROCESS DATA ---
X, y = [], []
class_folders = sorted(os.listdir(DATASET_DIR))  # expect ['A','B',...,'Z']

for cls in class_folders:
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(cls_path): continue
    label = ord(cls.upper()) - ord('A')
    for fname in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):
        if not fname.lower().endswith(IMAGE_EXT): continue
        img = cv2.imread(os.path.join(cls_path, fname))
        vec = extract_landmark_vector(img)
        if vec is None:
            continue   # skip frames without clear two-hand detection
        X.append(vec)
        y.append(label)

X = np.vstack(X)  # shape (N_samples, 126)
y = np.array(y, dtype=np.int32)

# --- NORMALIZE & SAVE ---
X = normalize(X)
np.save(OUT_X, X)
np.save(OUT_Y, y)

print(f"Saved {X.shape[0]} samples → {OUT_X}, {OUT_Y}")


