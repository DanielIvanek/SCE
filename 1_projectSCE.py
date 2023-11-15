import math
import random
import csv

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.input_layer = [random.uniform(-1, 1) for _ in range(input_size)]
        self.hidden_layer = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.output_layer = [random.uniform(-1, 1) for _ in range(output_size)]

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - math.tanh(x)**2

    def feedforward(self, x):
        hidden_input = sum(self.input_layer[i] * x for i in range(self.input_size))
        hidden_output = self.tanh(hidden_input)

        output_input = sum(self.hidden_layer[i] * hidden_output for i in range(self.hidden_size))
        output = self.tanh(output_input)

        return output

    def train(self, x, y, epochs):
        mse = 0
        for _ in range(epochs):
            # Feedforward
            hidden_input = sum(self.input_layer[i] * x for i in range(self.input_size))
            hidden_output = self.tanh(hidden_input)

            output_input = sum(self.hidden_layer[i] * hidden_output for i in range(self.hidden_size))
            output = self.tanh(output_input)

            # Compute error
            error = y - output
            mse += error ** 2

            # Backpropagation
            d_output = error * self.tanh_derivative(output)
            d_hidden = d_output * self.hidden_layer[0] * self.tanh_derivative(hidden_output)

            # Update weights
            for i in range(self.input_size):
                self.input_layer[i] += self.learning_rate * d_hidden * x
            for i in range(self.hidden_size):
                self.hidden_layer[i] += self.learning_rate * d_output * hidden_output
            for i in range(self.output_size):
                self.output_layer[i] += self.learning_rate * d_output * hidden_output

        mse /= epochs  # Average MSE
        return mse

# Sensitivity analysis
hidden_sizes = [4, 6, 10]  # Number of neurons in the hidden layer
learning_rates = [0.1, 0.5, 0.8]  # Learning rate
epochs_values = [100, 1000, 5000]  # Number of epochs 

# Generate a range of x values
x_values = [i * 0.01 for i in range(-500, 500)]  # Values from -10 to 10

for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        for epochs in epochs_values:
            nn = NeuralNetwork(1, hidden_size, 1, learning_rate)
            final_mse = nn.train(1.0, math.tanh(1.0), epochs)  # Change here to use tanh function
            print(f"Hidden Size: {hidden_size}, Learning Rate: {learning_rate}, Epochs: {epochs}")
            print(f"Final MSE: {final_mse}")

            # Compute the network's y values
            network_y_values = [nn.feedforward(x) for x in x_values]

            # Open the CSV file for writing
            with open(f'network_output_{hidden_size}_{learning_rate}_{epochs}.csv', 'w', newline='') as csvfile:
                fieldnames = ['Input', 'Network Output']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write the header
                writer.writeheader()

                for x, y in zip(x_values, network_y_values):
                    # Write the input and output to the CSV file
                    writer.writerow({'Input': x, 'Network Output': y})
