import nnfs
import numpy as np
nnfs.init()
from nnfs.datasets import spiral_data
from activation_ReLU import Activation_ReLU
from dense_layer import Layer_Dense
from activation_softmax_loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizer_SGD import Optimizer_SGD
from optmizer_Adagrad import Optimizer_Adagrad
from Optimizer_RMSprop import Optimizer_RMSprop
from optmizer_Adam import Optimizer_Adam
from layer_Dropout import Layer_Dropout

X, y = spiral_data(samples=1000, classes=3)
dense1 = Layer_Dense(2, 512,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):
    #forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss  + regularization_loss
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
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
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')