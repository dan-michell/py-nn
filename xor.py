import numpy as np

from network_creation.network import Network
from network_creation.layers.fully_connected_layer import FCLayer
from network_creation.layers.activation_layer import ActivationLayer
from network_creation.functions.activation_functions import tanh, tanh_prime
from network_creation.functions.loss_functions import mse, mse_prime

# Training XOR data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Create network
network = Network()
network.add_layer(FCLayer(2,3))
network.add_layer(ActivationLayer(tanh, tanh_prime))
network.add_layer(FCLayer(3,1))
network.add_layer(ActivationLayer(tanh,tanh_prime))

# Train network
network.set_loss_functions(mse, mse_prime)
network.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

output = network.predict(x_train)
print(output)

