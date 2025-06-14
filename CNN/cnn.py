import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import h5py


class CNN:
    def __init__(self, input_shape=(28, 28, 1), conv_layers=None, hidden_layers=[128, 64], output_size=26):
        """
        CNN Implementation from scratch
        
        Args:
            input_shape: (height, width, channels) of input images
            conv_layers: List of dictionaries defining conv layers
                        [{'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                         {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}]
            hidden_layers: List of neurons in fully connected layers
            output_size: Number of output classes
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers or [
            {'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Initialize convolutional parameters
        self.conv_weights = []
        self.conv_biases = []
        
        # Initialize fully connected parameters
        self.fc_weights = []
        self.fc_biases = []
        
        self.iterations = 0
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for all layers"""
        
        # Initialize convolutional layers
        in_channels = self.input_shape[2]
        
        for conv_config in self.conv_layers:
            filters = conv_config['filters']
            kernel_size = conv_config['kernel_size']
            
            # He initialization for conv weights
            W = np.random.randn(filters, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            b = np.zeros((filters, 1))
            
            self.conv_weights.append(W)
            self.conv_biases.append(b)
            
            in_channels = filters
        
        # Calculate flattened size after conv layers
        flattened_size = self._calculate_flattened_size()
        
        # Initialize fully connected layers
        fc_layers = [flattened_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(fc_layers) - 1):
            W = np.random.randn(fc_layers[i], fc_layers[i+1]) * np.sqrt(2 / fc_layers[i])
            b = np.zeros((1, fc_layers[i + 1]))
            self.fc_weights.append(W)
            self.fc_biases.append(b)
    
    def _calculate_flattened_size(self):
        """Calculate the size after conv and pooling operations"""
        h, w = self.input_shape[0], self.input_shape[1]
        
        for conv_config in self.conv_layers:
            kernel_size = conv_config['kernel_size']
            stride = conv_config['stride']
            padding = conv_config['padding']
            
            # After convolution
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
            
            # After max pooling (2x2 with stride 2)
            h = h // 2
            w = w // 2
        
        return h * w * self.conv_layers[-1]['filters']
    
    def _pad_input(self, X, padding):
        """Add zero padding to input"""
        if padding == 0:
            return X
        return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    def _conv2d(self, X, W, b, stride=1, padding=0):
        """
        2D Convolution operation
        
        Args:
            X: Input tensor (batch_size, height, width, channels)
            W: Weights (filters, in_channels, kernel_height, kernel_width)
            b: Biases (filters, 1)
            stride: Stride value
            padding: Padding value
        
        Returns:
            Output tensor after convolution
        """
        batch_size, in_h, in_w, in_c = X.shape
        filters, _, kernel_h, kernel_w = W.shape
        
        # Add padding
        X_padded = self._pad_input(X, padding)
        
        # Calculate output dimensions
        out_h = (in_h + 2 * padding - kernel_h) // stride + 1
        out_w = (in_w + 2 * padding - kernel_w) // stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, filters))
        
        # Perform convolution
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kernel_h
                w_start = j * stride
                w_end = w_start + kernel_w
                
                # Extract patch
                patch = X_padded[:, h_start:h_end, w_start:w_end, :]
                
                # Convolve with each filter
                for f in range(filters):
                    output[:, i, j, f] = np.sum(patch * W[f], axis=(1, 2, 3)) + b[f]
        
        return output
    
    def _max_pool2d(self, X, pool_size=2, stride=2):
        """
        Max pooling operation
        
        Args:
            X: Input tensor (batch_size, height, width, channels)
            pool_size: Size of pooling window
            stride: Stride value
        
        Returns:
            Output tensor after max pooling
        """
        batch_size, in_h, in_w, channels = X.shape
        
        out_h = (in_h - pool_size) // stride + 1
        out_w = (in_w - pool_size) // stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                
                pool_region = X[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(pool_region, axis=(1, 2))
        
        return output
    
    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        """Softmax activation function"""
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ / np.sum(eZ, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the CNN
        
        Args:
            X: Input tensor (batch_size, height, width, channels)
        
        Returns:
            activations: List of activations from each layer
            conv_outputs: List of conv layer outputs (for backprop)
        """
        activations = [X]
        conv_outputs = []
        
        # Forward through convolutional layers
        A = X
        for i, conv_config in enumerate(self.conv_layers):
            # Convolution
            Z = self._conv2d(A, self.conv_weights[i], self.conv_biases[i], 
                           conv_config['stride'], conv_config['padding'])
            conv_outputs.append(Z)
            
            # ReLU activation
            A = self.relu(Z)
            
            # Max pooling
            A = self._max_pool2d(A)
            activations.append(A)
        
        # Flatten for fully connected layers
        batch_size = A.shape[0]
        A_flat = A.reshape(batch_size, -1)
        activations.append(A_flat)
        
        # Forward through fully connected layers
        fc_z_values = []
        for i in range(len(self.fc_weights)):
            Z = A_flat.dot(self.fc_weights[i]) + self.fc_biases[i]
            fc_z_values.append(Z)
            
            if i < len(self.fc_weights) - 1:
                A_flat = self.relu(Z)
            else:
                A_flat = self.softmax(Z)
            
            activations.append(A_flat)
        
        return activations, conv_outputs, fc_z_values
    
    def cross_entropy_loss(self, y_pred, y_true):
        """Cross-entropy loss function"""
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return loss
    
    def backward_propagation(self, activations, conv_outputs, fc_z_values, y_true):
        """
        Simplified backward propagation (mainly for FC layers)
        Note: Full CNN backprop is complex and would require extensive implementation
        """
        m = y_true.shape[0]
        
        # Gradients for FC layers
        fc_grad_weights = [0] * len(self.fc_weights)
        fc_grad_biases = [0] * len(self.fc_biases)
        
        # Output layer gradients
        A_output = activations[-1]
        dZ = (A_output - y_true) / m
        
        fc_grad_weights[-1] = activations[-3].T.dot(dZ)  # -3 because of flattened layer
        fc_grad_biases[-1] = np.sum(dZ, axis=0, keepdims=True)
        
        dA_prev = dZ
        
        # Hidden FC layers gradients
        for i in reversed(range(len(self.hidden_layers))):
            dA = dA_prev.dot(self.fc_weights[i+1].T)
            dZ = dA * (fc_z_values[i] > 0).astype(float)
            
            fc_grad_weights[i] = activations[-(4+i)].T.dot(dZ)
            fc_grad_biases[i] = np.sum(dZ, axis=0, keepdims=True)
            
            dA_prev = dZ
        
        return fc_grad_weights, fc_grad_biases
    
    def update_parameters(self, fc_grads_w, fc_grads_b, lr=0.01, decay=0):
        """Update parameters using gradients"""
        if decay:
            learning_rate = lr / (1 + decay * self.iterations)
        else:
            learning_rate = lr
        
        # Update FC layers
        for i in range(len(self.fc_weights)):
            self.fc_weights[i] -= learning_rate * fc_grads_w[i]
            self.fc_biases[i] -= learning_rate * fc_grads_b[i]
        
        self.iterations += 1
    
    def train(self, X_train, y_train, epochs=100, learning_rate=0.01, print_loss=False):
        """Train the CNN"""
        for epoch in range(epochs):
            activations, conv_outputs, fc_z_values = self.forward_propagation(X_train)
            y_pred = activations[-1]
            loss = np.mean(self.cross_entropy_loss(y_pred, y_train))
            
            fc_grads_w, fc_grads_b = self.backward_propagation(activations, conv_outputs, fc_z_values, y_train)
            self.update_parameters(fc_grads_w, fc_grads_b, lr=learning_rate)
            
            if print_loss and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        activations, _, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        activations, _, _ = self.forward_propagation(X)
        return activations[-1]


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed metrics
    """
    
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or [f"Class_{i}" for i in range(model.output_size)]
    
    def evaluate(self, X_test, y_test_encoded, y_test_labels=None):
        """
        Comprehensive evaluation of the model
        
        Args:
            X_test: Test features
            y_test_encoded: One-hot encoded test labels
            y_test_labels: Original test labels (if available)
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Convert one-hot to labels if needed
        if y_test_labels is None:
            y_true = np.argmax(y_test_encoded, axis=1)
        else:
            y_true = y_test_labels
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics for each class
        metrics = self._calculate_detailed_metrics(cm)
        
        # Overall metrics
        overall_accuracy = self._calculate_accuracy(cm)
        macro_precision = np.mean([metrics[i]['precision'] for i in range(len(self.class_names))])
        macro_recall = np.mean([metrics[i]['recall'] for i in range(len(self.class_names))])
        macro_f1 = np.mean([metrics[i]['f1_score'] for i in range(len(self.class_names))])
        
        # Weighted metrics (by support)
        total_support = np.sum([metrics[i]['support'] for i in range(len(self.class_names))])
        weighted_precision = np.sum([metrics[i]['precision'] * metrics[i]['support'] for i in range(len(self.class_names))]) / total_support
        weighted_recall = np.sum([metrics[i]['recall'] * metrics[i]['support'] for i in range(len(self.class_names))]) / total_support
        weighted_f1 = np.sum([metrics[i]['f1_score'] * metrics[i]['support'] for i in range(len(self.class_names))]) / total_support
        
        results = {
            'confusion_matrix': cm,
            'per_class_metrics': metrics,
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1_score': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1_score': weighted_f1
            },
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'true_labels': y_true
        }
        
        return results
    
    def _calculate_detailed_metrics(self, confusion_matrix):
        """
        Calculate precision, recall, F1-score for each class
        
        Formulas:
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN) 
        - F1-Score = 2 * (Precision × Recall) / (Precision + Recall)
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Where:
        - TP (True Positives): Correctly predicted positive cases
        - TN (True Negatives): Correctly predicted negative cases  
        - FP (False Positives): Incorrectly predicted as positive
        - FN (False Negatives): Incorrectly predicted as negative
        """
        num_classes = confusion_matrix.shape[0]
        metrics = {}
        
        for i in range(num_classes):
            # True Positives: diagonal element
            tp = confusion_matrix[i, i]
            
            # False Positives: sum of column i minus TP
            fp = np.sum(confusion_matrix[:, i]) - tp
            
            # False Negatives: sum of row i minus TP
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            # True Negatives: total minus TP, FP, FN
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            # Calculate metrics with zero-division handling
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Support (number of actual samples for this class)
            support = tp + fn
            
            metrics[i] = {
                'class_name': self.class_names[i],
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support
            }
        
        return metrics
    
    def _calculate_accuracy(self, confusion_matrix):
        """
        Calculate overall accuracy
        
        Formula: Accuracy = (TP₁ + TP₂ + ... + TPₙ) / Total_Samples
        Or: Accuracy = Sum of diagonal elements / Sum of all elements
        """
        return np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    def print_detailed_report(self, results):
        """Print a comprehensive evaluation report"""
        print("=" * 80)
        print("DETAILED MODEL EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\nOVERALL METRICS:")
        print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
        print(f"Macro Precision: {results['overall_metrics']['macro_precision']:.4f}")
        print(f"Macro Recall: {results['overall_metrics']['macro_recall']:.4f}")
        print(f"Macro F1-Score: {results['overall_metrics']['macro_f1_score']:.4f}")
        print(f"Weighted Precision: {results['overall_metrics']['weighted_precision']:.4f}")
        print(f"Weighted Recall: {results['overall_metrics']['weighted_recall']:.4f}")
        print(f"Weighted F1-Score: {results['overall_metrics']['weighted_f1_score']:.4f}")
        
        print(f"\nPER-CLASS METRICS:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 65)
        
        for i, metrics in results['per_class_metrics'].items():
            print(f"{metrics['class_name']:<15} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['support']:<10}")
        
        print(f"\nCONFUSION MATRIX:")
        print(results['confusion_matrix'])
        
        print(f"\nMETRIC FORMULAS EXPLANATION:")
        print("• Precision = TP / (TP + FP) - Of all positive predictions, how many were correct?")
        print("• Recall = TP / (TP + FN) - Of all actual positives, how many were correctly identified?")
        print("• F1-Score = 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean of precision and recall")
        print("• Accuracy = (TP + TN) / Total - Overall correctness of the model")
        print("\nWhere:")
        print("  TP = True Positives, TN = True Negatives")
        print("  FP = False Positives, FN = False Negatives")


# Example usage:
if __name__ == "__main__":
    # Example with dummy data
    X_train = np.random.randn(100, 28, 28, 1)  # 100 samples, 28x28 grayscale images
    y_train = np.eye(26)[np.random.randint(0, 26, 100)]  # One-hot encoded labels
    
    X_test = np.random.randn(20, 28, 28, 1)
    y_test = np.eye(26)[np.random.randint(0, 26, 20)]
    
    # Create and train CNN
    cnn = CNN(input_shape=(28, 28, 1), output_size=26)
    cnn.train(X_train, y_train, epochs=50, learning_rate=0.01, print_loss=True)
    
    # Evaluate model
    evaluator = ModelEvaluator(cnn, class_names=[chr(ord('A') + i) for i in range(26)])
    results = evaluator.evaluate(X_test, y_test)
    evaluator.print_detailed_report(results)