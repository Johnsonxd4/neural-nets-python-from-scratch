import numpy as np
import nnfs
nnfs.init()
from nnfs.datasets import sine_data
from dense_layer import Layer_Dense
from activation_ReLU import Activation_ReLU
from linear_activation import Activation_Linear
from Loss_MeanSquaredError import Loss_MeanSquaredError
from optmizer_Adam import Optimizer_Adam
from nnfs.datasets import sine_data
from model import Model
X, y = sine_data()

model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
)

model.finalize()
model.train(X, y, epochs=10000, print_every=100)
