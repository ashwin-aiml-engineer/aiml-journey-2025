"""
Day 22: Neural Network Fundamentals (Practice)
- Perceptron, MLP, activation, forward/backprop, loss, SGD
"""
import numpy as np

# Simple Perceptron
class Perceptron:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim)
        self.b = 0.0
    def activation(self, x):
        return 1 if x > 0 else 0
    def predict(self, X):
        return [self.activation(np.dot(self.w, x) + self.b) for x in X]
    def train(self, X, y, lr=0.1, epochs=10):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.activation(np.dot(self.w, xi) + self.b)
                self.w += lr * (yi - pred) * np.array(xi)
                self.b += lr * (yi - pred)

# MLP with one hidden layer
class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim)
    def relu(self, x):
        return np.maximum(0, x)
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
    def forward(self, X):
        h = self.relu(np.dot(X, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return self.softmax(out)

# Loss functions
class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    @staticmethod
    def cross_entropy(y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# SGD optimizer
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

if __name__ == "__main__":
    print("Neural Network Fundamentals ready for practice!")
