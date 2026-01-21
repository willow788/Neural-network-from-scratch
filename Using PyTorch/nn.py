#we will be training a simple feedforward neural network using PyTorch and also add backpropagation functionality.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

#defining the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    #defining the forward pass
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def backward(self, loss):
        #zero the grads
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

            
        
#function to train the neural network

def train_network(model, criterion, optimizer, data_loader, num_epochs):
    model.optimizer = optimizer
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #backward pass
            model.backward(loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

#example usage
if __name__ == "__main__":
    #hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_epochs = 5
    learning_rate = 0.001

    #dummy data
    x_train = torch.randn(100, input_size)
    y_train = torch.randn(100, output_size)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    #model, loss function and optimizer
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train the model
    trained_model = train_network(model, criterion, optimizer, data_loader, num_epochs)


"""
we are explaining the code here,
this code defines a simple feedforward neural network using pytorch.
the NeuralNetwork class defining the architecture with two hidden layers and relu activation functions.

1. the first layer takes the input size and maps it to the hidden size

2. the second layer maps the hidden size to another hidden  size, these two hidden layers help the network to learn complex patterns.

3. the final layer maps the hidden size to the output size.

the forward function defines how the data flows through the netwrok from input to output.

the backward function implements the backpropagatio algo, which computes the gradients of the loss wrt the model parameters and updates them using the optimizer.

the train_network function handles the training process, it iterates over the dataset for a specific number of epoch, each time computing the losss and calling the backward function to update the model parameters.


"""
