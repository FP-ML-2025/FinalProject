import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
# 1. Load your concatenated “news” data (only X, Y, Z)
X = np.load('X_news.npy')   # shape (N, 126)
y = np.load('y_news.npy')

# 2. Remap labels 23→0, 24→1, 25→2
unique_labels = np.unique(y)                     # array([23,24,25])
label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}
y_mapped      = np.array([label_map[v] for v in y], dtype=np.int32)

# 3. One‑hot encode into 3 classes
num_classes = len(unique_labels)                 # 3
y_cat       = to_categorical(y_mapped, num_classes)

# 4. Reshape & preprocess
X = X.reshape(-1, 21, 6, 1).astype(np.float32)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_mapped, random_state=42
)

# 6. Define CNN
model = Sequential([
    InputLayer(input_shape=(21, 6, 1)),
    Conv2D(32, (3,3), activation='relu', padding='same'), MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'), MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])

learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=32
)

# 8. Save
model.save('cnn_hand_sign_models.h5')
print("Model trained and saved to cnn_hand_sign_models.h5")
# ——— 9. EVALUATION ———
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import numpy as np

# 1) Predict on X_test
y_pred_proba = model.predict(X_test)            # shape (n_samples, num_classes)
y_pred       = np.argmax(y_pred_proba, axis=1)  # discrete labels
y_true       = np.argmax(y_test,       axis=1)

# 2) Compute metrics
acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average='macro')
rec   = recall_score(y_true, y_pred, average='macro')
f1    = f1_score(y_true, y_pred, average='macro')

print("\n— Evaluation on Test Set —")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}  (macro‑average)")
print(f"Recall   : {rec:.4f}  (macro‑average)")
print(f"F1‑Score : {f1:.4f}  (macro‑average)\n")

# 3) Detailed per‑class report
print("Per‑class breakdown:\n")
print(classification_report(y_true, y_pred, digits=4))



