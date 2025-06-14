import numpy as np

# Load the .npy file
X = np.load('X_news.npy')  # or 'y_news.npy'
y = np.load('y_news.npy')  # or 'y_news.npy'
# Check shape and type
print("Shape:", X.shape)
print("Data type:", X.dtype)

# View the first sample
print("First sample:\n", X[0])

# Optionally: view a few more
print("Next 5 samples:\n", X[1:6])

import pandas as pd

X = np.load('X_news.npy')
df = pd.DataFrame(X)
y = pd.DataFrame(y)
# Show first 5 rows
print(df.head())
print(y)