import torch
import torch.nn as nn

# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3)  # Input layer to hidden layer
        self.relu = nn.ReLU()      # Activation function
        self.fc2 = nn.Linear(3, 1) # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the network
net = SimpleNet()

# Example input
x = torch.tensor([[1.0, 2.0, 3.0]])  # Batch of one, two features
output = net(x)
print(output)
