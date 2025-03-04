import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

# underlying model for learners
class Model(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state):
        pass


class LogReg(nn.Module):
    # simple logistic regression model
    def __init__(self, d_input: int, d_output: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(d_output, d_input))
        nn.init.kaiming_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(d_output))

    def forward(self, x):
        h = torch.matmul(self.weights, x) + self.bias
        return torch.softmax(h, dim = 1)


class MLP(Model):
    # simple N layer neural network
    def __init__(self, d_input: int, d_output: int, d_hidden: list[int]):
        # use torch list to initialize weights and biases
        super().__init__()
        self.weights = [nn.Parameter(torch.randn(d_hidden[i - 1], d_hidden[i])) for i in range(1, len(d_hidden))]
        self.biases = [nn.Parameter(torch.zeros(d_hidden[i])) for i in range(1, len(d_hidden))]
        self.weights.insert(0, nn.Parameter(torch.randn(d_input, d_hidden[0])))
        self.biases.insert(0, nn.Parameter(torch.zeros(d_hidden[0])))
        self.weights.append(nn.Parameter(torch.randn(d_output, d_hidden[-1])))
        self.biases.append(nn.Parameter(torch.zeros(d_output)))

        # initialization
        for i in range(len(self.weights)):
            nn.init.kaiming_normal_(self.weights[i])
            nn.init.zeros_(self.biases[i])
    
    def forward(self, state):
        h = torch.matmul(state, self.weights1.t()) + self.bias1
        h = torch.relu(h)
        h = torch.matmul(h, self.weights2.t()) + self.bias2
        return torch.softmax(h, dim=1)

class LSTM(Model):
    def __init__(self, d_input: int, d_output: int, d_hidden: list[int]):
        super().__init__()


if __name__ == "__main__":
    model = MLP(10, 2, [10, 10])
    print(model.weights)