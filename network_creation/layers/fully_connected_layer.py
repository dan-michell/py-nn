# Implementation of a fully connected layer, inheriting the base layer class.

from .base_layer import Layer
import numpy as np

class FCLayer(Layer):
    # Input / output size = number of neurons in respective layer
    def __init__(self, input_size, output_size):
        # Assign random weights and bias based on the layer size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # Computes dE/dW, dE/dB for a given output_error = dE/dY. Returns input_error = dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# INPUTS COMING IN AS UFUNC
