import nnfs
import numpy as np
nnfs.init()
from nnfs.datasets import spiral_data
from activation_ReLU import Activation_ReLU
from dense_layer import Layer_Dense
from activation_softmax_loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from optmizer_Adagrad import Optimizer_Adagrad
from Optimizer_RMSprop import Optimizer_RMSprop
from optmizer_Adam import Optimizer_Adam
from layer_Dropout import Layer_Dropout
from activation_sigmoid import Activation_Sigmoid
from loss_binaryCrossentropy import Loss_BinaryCrossentropy
X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1,1)
dense1 = Layer_Dense(2, 64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
optimizer = Optimizer_Adam(decay=5e-7)

for epoch in range(10001):
    #forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss  + regularization_loss
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {optimizer.current_learning_rate}')
    
    #backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100, classes=2)
y_test.reshape(-1,1)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')