import numpy as np


# Constants
NROWS = 8
DELTA = 0.2
ALHPA = 0.02


class ComplexUnit:
    """A unit that takes inputs from simple units and recognizes a pattern"""
    def __init__(self):
        self.y_bar = 0.
        self.weights = np.random.rand(NROWS, NROWS, 4) * 0.1

    def run(self, inputs):
        return np.sum(np.multiply(self.weights, inputs))

    def train(self, inputs, y_t):
        # Update y_bar
        self.y_bar = self.y_bar * (1 - DELTA) + y_t * DELTA
        # Update weights
        self.weights += ALHPA * self.y_bar * (inputs - self.weights)

    def get_weights(self):
        return self.weights

    def get_y_bar(self):
        return self.y_bar


class Model:
    """The complete model containing multiple complex units"""
    def __init__(self):
        self.units = [ComplexUnit() for _ in range(4)]

    def run(self, inputs):
        values = [u.run(inputs) for u in self.units]
        k = np.argmax(values)
        return k

    def train(self, inputs):
        k = self.run(inputs)
        for i in range(len(self.units)):
            y_t = 1 if i == k else 0
            self.units[i].train(inputs, y_t)

    def get_weights(self):
        return [u.get_weights() for u in self.units]

    def get_y_bars(self):
        return [u.get_y_bar() for u in self.units]
