import numpy as np

# 1) Load existing dataset
X = np.load("X_news.npy")   # e.g. shape (1350, 126)
y = np.load("y_news.npy")   # e.g. shape (1350,)

# 2) Remove all old “S” (label 18) samples
mask_not_S = (y != 18)
X_nonS = X[mask_not_S]
y_nonS = y[mask_not_S]

# 3) Load or define your new “S” data
#    (Make sure these have the same feature‐shape as X_nonS and labels are 18.)
X_S = np.load("X_S.npy")    # e.g. shape (N_S, 126)
y_S = np.full((X_S.shape[0],), 18, dtype=y.dtype)

# 4) Re‐concatenate
X_updated = np.concatenate([X_nonS, X_S], axis=0)
y_updated = np.concatenate([y_nonS, y_S], axis=0)

# 5) (Optional) Shuffle
perm = np.random.permutation(len(y_updated))
X_updated = X_updated[perm]
y_updated = y_updated[perm]

# 6) Save back
np.save("X_news.npy", X_updated)
np.save("y_news.npy", y_updated)
print(f"Replaced all old 'S' samples. New shape: {X_updated.shape}, {y_updated.shape}")

