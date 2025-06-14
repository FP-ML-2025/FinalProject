import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h5py


class NeuralNetwork:
  def __init__(self, input_size = 126, hidden_layers = [128, 64, 32], output_size = 26):
   
    # Input size is the number of pixels in an image
    # in our cases is the number of landmarks of the hand in mediapipe
    self.input_size = input_size

    # hidden layers are the neurons inbetween input later and output layer
    self.hidden_layers = hidden_layers

    # output size are the number of alphabets
    self.output_size = output_size
    
    self.weights = []
    self.biases = []
    self.iterations = 0

  
    # here we construct the layers,that consist of
    # input layer -> hidden layer -> output layer
    layers = [input_size] + hidden_layers + [output_size]


    for i in range(len(layers) - 1):
      
      # Assign weights and biases to each layer, randomly

      W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])

      b = np.zeros((1, layers[i + 1]))
      self.weights.append(W)
      self.biases.append(b)

  def relu(self, Z):
    return np.maximum(0, Z)
  
  def softmax(self, Z):
    eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return eZ / np.sum(eZ, axis=1, keepdims=True)


  def forwardPropagation(self, x):
    # X is the neurons in the input layers
    A = x 

    # put the neutons in the input layers 
    activation = [A]

    # Z value are the equation before we put it with an activation function
    Z_Values = []

    # Forward Propagation : z = W₁·X + b₁

    for i in range(len(self.weights)):

      # in here basically since neurons are a lot, we calculate the dot product from each layer
      Z = A.dot(self.weights[i]) + self.biases[i]
      Z_Values.append(Z)

      if i < len(self.weights) - 1:
        A = self.relu(Z)
      
      else:
        A = self.softmax(Z)
      
      activation.append(A)

    return activation, Z_Values


  def CrossEntropy(self, yPred, yTrue):
    yPred = np.clip(yPred, 1e-10, 1 - 1e-10)

    loss = -np.sum(yTrue * np.log(yPred), axis=1)

    return loss
  
  def backwardPropagation(self, activations, Z_Values, yTrue):

    m = yTrue.shape[0]
    gradient_weights = [0] * len(self.weights)
    gradient_biases = [0] * len(self.biases)

    # ----- OUTPUT LAYER ----- #
    A_output = activations[-1]
    
    # ∂C/∂Z #
    dZ = (A_output - yTrue) / m # ∂C/∂A * ∂A/∂Z

    # ∂Z/∂W #
    gradient_weights[-1] = activations[-2].T.dot(dZ) 

    # ∂C/∂B #
    gradient_biases[-1] = np.sum(dZ, axis=0, keepdims=True)

    dA_prev = dZ 

    # ----- HIDDEN LAYER ----- #

    for i in reversed(range(len(self.hidden_layers))):
      # calculate ∂C/∂a

      dA = dA_prev.dot(self.weights[i+1].T)

      # calculate ∂a/∂Z

      dZ = dA * (Z_Values[i] > 0).astype(float)

      # calculate ∂Z/∂w

      gradient_weights[i] = activations[i].T.dot(dZ)

      gradient_biases[i] = np.sum(dZ, axis=0, keepdims=True)

      dA_prev = dZ
    
    return gradient_weights, gradient_biases
  
  def updateParams(self, grads_w, grads_b , lr=0.05, decay=0):
    if decay:
      learning_rate = lr / (1 + decay * self.iterations)
    else:
      learning_rate = lr
    
    for i in range(len(self.weights)):
      self.weights[i] -= learning_rate * grads_w[i]
      self.biases[i] -= learning_rate * grads_b[i]
    
    self.iterations += 1
  
  def train(self, X_train, Y_train, epochs = 10000, learning_rate = 0.01, print_loss=False):
    for epoch in range(epochs):
      activations, Z_values = self.forwardPropagation(X_train)
      yPred = activations[-1]
      loss = np.mean(self.CrossEntropy(yPred, Y_train))
      grads_w, grads_b = self.backwardPropagation(activations, Z_values, Y_train)
      decay = 0
      self.updateParams(grads_w, grads_b, lr = learning_rate, decay=decay)

      if print_loss and epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

  def predict(self, X):
    activations ,_ = self.forwardPropagation(X)
    return np.argmax(activations[-1], axis=1)
  
  
  def save(self, filename="custom_model.h5"):
    with h5py.File(filename, "w") as f:
      f.create_dataset("input_size", data=self.input_size)
      f.create_dataset("output_size", data=self.output_size)
      f.create_dataset("hidden_layers", data=self.hidden_layers)

      for i, (w, b) in enumerate(zip(self.weights, self.biases)):
        f.create_dataset(f"weights_{i}", data=w)
        f.create_dataset(f"biases_{i}", data=b)

  @classmethod
  def load(cls, filename="custom_model.h5"):
    with h5py.File(filename, "r") as f:
      input_size = int(f["input_size"][()])
      output_size = int(f["output_size"][()])
      hidden_layers = list(f["hidden_layers"][()])

      model = cls(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size)

      for i in range(len(hidden_layers) + 1):
        model.weights[i] = f[f"weights_{i}"][()]
        model.biases[i] = f[f"biases_{i}"][()]
        
    return model

