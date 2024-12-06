import torch.nn as nn


class HallucinationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(HallucinationModel, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Input to first hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # First hidden to second hidden layer
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Second hidden to output layer

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
