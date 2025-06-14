
from cnn import NeuralNetwork
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h5py

X = np.load('X_news.npy')   # shape (N_samples, 126)
y = np.load('y_news.npy')   # shape (N_samples,)



ohe = OneHotEncoder(sparse_output=False, categories='auto')
y_onehot = ohe.fit_transform(y.reshape(-1, 1))


model = NeuralNetwork(input_size=126,
                   hidden_layers=[512, 256, 128, 64],
                   output_size=26)

model.train(X, y_onehot, epochs=5000, learning_rate=0.01, print_loss=True)


model.save("sign_language_model.h5")


