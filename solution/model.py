
import math

class LogisticModel:
    def __init__(self, num_features: int):
        self.weights = [0.0] * num_features
        self.bias = 0.0

    def sigmoid(self, z: float) -> float:
        z = max(min(z, 20), -20)
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, features: list[float]) -> float:
        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        return self.sigmoid(z)

    def train_step(self, features: list[float], label: int, lr: float):
        pred = self.predict(features)
        err = pred - label
        for i in range(len(self.weights)):
            self.weights[i] -= lr * err * features[i]
        self.bias -= lr * err
