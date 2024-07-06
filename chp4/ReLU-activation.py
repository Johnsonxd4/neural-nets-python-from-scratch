import nnfs
nnfs.init()
from nnfs.datasets import spiral_data
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs,n_newrons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_newrons)
        self.biases = np.zeros((1, n_newrons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

X,y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
activation = Activation_ReLU()
dense1.forward(X)
activation.forward(dense1.output)
print(activation.output[:5])