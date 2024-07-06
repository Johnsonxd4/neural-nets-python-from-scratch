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

X,y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.output[:5])