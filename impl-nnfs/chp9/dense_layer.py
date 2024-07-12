import nnfs
nnfs.init()

from nnfs.datasets import spiral_data
import numpy as np
class Layer_Dense:
    def __init__(self, n_inputs,n_newrons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_newrons)
        self.biases = np.zeros((1, n_newrons))
        
    def forward(self,inputs):
        #store inputs for partial derivate calculation
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
