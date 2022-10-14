import numpy as np

from network_creation.network import Network
from network_creation.layers.fully_connected_layer import FCLayer
from network_creation.layers.activation_layer import ActivationLayer
from network_creation.functions.activation_functions import tanh, tanh_prime
from network_creation.functions.loss_functions import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

net = Network()
net.add_layer(FCLayer(28*28, 100))                # input_shape = (1, 28*28) | output_shape = (1, 100)
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(100, 50))                   # input_shape = (1, 100) | output_shape = (1, 50)
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(50, 10))                    # input_shape = (1, 50) | output_shape=(1, 10)
net.add_layer(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.set_loss_functions(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])