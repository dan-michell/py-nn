# Create and train a Neural Network!

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def set_loss_functions(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            error_display = 0
            for j in range(samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # Compute loss for display
                error_display += self.loss(y_train[j], output)

                # Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            error_display /= samples
            print(f"Epoch {i + 1}/{epochs}. Error: {error_display}")