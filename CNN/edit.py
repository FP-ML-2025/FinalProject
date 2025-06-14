import numpy as np
import string
import matplotlib.pyplot as plt

# Load the data
X = np.load("X_news.npy")
y = np.load("y_news.npy")

# Print basic info
print("X shape:", X.shape)  # Expecting (N, 126)
print("y shape:", y.shape)
print("Unique labels:", np.unique(y))

# Mapping label index (0-25) to A-Z
index_to_label = {i: letter for i, letter in enumerate(string.ascii_uppercase)}
label_to_index = {v: k for k, v in index_to_label.items()}

# Show what label 19 is
print(f"\nLabel 18 corresponds to: '{index_to_label[18]}'")
