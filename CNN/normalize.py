import numpy as np

# 1) Load your raw data
X = np.load('X_news.npy')  # shape: (N_samples, 126)
y = np.load('y_news.npy')  # shape: (N_samples,)

# 2) Compute feature-wise min and max
X_min = X.min(axis=0, keepdims=True)   # shape: (1, 126)
X_max = X.max(axis=0, keepdims=True)   # shape: (1, 126)

# 3) Avoid division by zero for constant features
range_ = X_max - X_min
range_[range_ == 0] = 1.0

# 4) Apply min–max normalization: X_norm in [0,1]
X_norm = (X - X_min) / range_

# 5) Save the normalized data
np.save('X_news_norm.npy', X_norm)
np.save('y_news.npy', y)  # unchanged

print(f"Preprocessing complete.")
print(f"  X_news_norm.npy shape: {X_norm.shape}")
print(f"  y_news.npy shape:      {y.shape}")
