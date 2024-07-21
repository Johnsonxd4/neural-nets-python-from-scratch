import numpy as np
import nnfs
nnfs.init()
from nnfs.datasets import sine_data
from dense_layer import Layer_Dense
from activation_ReLU import Activation_ReLU
from linear_activation import Activation_Linear
from Loss_MeanSquaredError import Loss_meanSquaredError
from optmizer_Adam import Optimizer_Adam
X,y = sine_data()

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 1)
activation2 = Activation_Linear()
loss_function = Loss_meanSquaredError()
optimizer = Optimizer_Adam()

accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    predictions = activation2.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f} (' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.current_learning_rate}')

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

import matplotlib.pyplot as plt
X_test, y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
plt.plot(X_test, y_test)
plt.plot(X_test, activation2.output)
plt.show()