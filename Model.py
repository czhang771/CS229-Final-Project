import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class Model(ABC, nn.Module):
    """Base class for all policy models (classical or neural)"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: torch.Tensor, batched: bool = False) -> torch.Tensor:
        # should return LOGITS over actions (outputs)
        pass


class LogReg(Model):
    """Simple logistic regression baseline in the spirit of CS229"""
    def __init__(self, d_input: int, d_output: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(d_input, d_output))
        nn.init.kaiming_normal_(self.weights)
        self.bias = nn.Parameter(torch.zeros(d_output))

    def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
        if batched:
            # flatten dimensions 1:
            B = x.shape[0]
            x = x.view(B, -1).float()
        else:
            x = x.view(1, -1).float()
        
        h = torch.matmul(x, self.weights) + self.bias
        # return logits over outputs
        return h


class MLP(Model):
    """Simple N layer neural network"""
    
    def __init__(self, d_input: int, d_output: int, d_hidden: list[int]):
        """Initialize with d_input input features, d_output output features, and |d_hidden| hidden layers with sizes in d_hidden"""

        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        sizes = [d_input] + d_hidden + [d_output]

        for i in range(len(sizes) - 1):
            self.weights.append(nn.Parameter(torch.randn(sizes[i], sizes[i + 1])))
            self.biases.append(nn.Parameter(torch.zeros(sizes[i + 1])))

            # initialization
            nn.init.kaiming_normal_(self.weights[i])
            nn.init.zeros_(self.biases[i])
    
    def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
        if batched:
            B = x.shape[0]
            x = x.view(B, -1)
        else:
            x = x.view(1, -1)

        h = torch.matmul(x, self.weights[0].t()) + self.biases[0]
        h = torch.relu(h)
        
        for i in range(1, len(self.weights)):
            h = torch.matmul(h, self.weights[i].t()) + self.biases[i]
            h = torch.relu(h)
        
        # return logits over outputs
        return h

class LSTMCell(nn.Module):
    """LSTM cell implementation"""
    def __init__(self, d_input: int, d_output: int, d_hidden: int):
        super().__init__()
        # LSTM layers; note we are using concatenation rather than learning separate U weights; this is equivalent
        self.W_f = nn.Parameter(torch.randn(d_input + d_hidden, d_output))
        self.W_c = nn.Parameter(torch.randn(d_input + d_hidden, d_output))
        self.W_i = nn.Parameter(torch.randn(d_input + d_hidden, d_output))
        self.W_o = nn.Parameter(torch.randn(d_input + d_hidden, d_output))
        self.b_f = nn.Parameter(torch.zeros(d_output))
        self.b_c = nn.Parameter(torch.zeros(d_output))
        self.b_i = nn.Parameter(torch.zeros(d_output))
        self.b_o = nn.Parameter(torch.zeros(d_output))
    
    def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
        if batched:
            # tentatively keeping sequence processing rather than flattening?
            B, T, D = x.shape
            c_t = torch.zeros(B, self.d_output)
            a_t = torch.zeros(B, self.d_output)
        else:
            c_t = torch.zeros(1, self.d_output)
            a_t = torch.zeros(1, self.d_output)

        for i in range(T):
            x_t = x[:, i, :]
            concat = torch.cat([x_t, a_t], dim = 1)
            f_t = torch.sigmoid(torch.matmul(concat, self.W_f) + self.b_f)
            i_t = torch.sigmoid(torch.matmul(concat, self.W_i) + self.b_i)
            o_t = torch.sigmoid(torch.matmul(concat, self.W_o) + self.b_o)
            tildec_t = torch.tanh(torch.matmul(concat, self.W_c) + self.b_c)
            c_t = f_t * c_t + i_t * tildec_t
            a_t = o_t * torch.tanh(c_t)
        
        return a_t

class LSTM(Model):
    """LSTM model implementation"""
    def __init__(self, d_input: int, d_output: int, d_hidden: list[int]):
        """Initialize with d_input input features, d_output output features, and |d_hidden| LSTM cells
          with sizes in d_hidden; note in the LSTM cell the hidden and output dimensions are the same"""
        
        super().__init__()
        self.cells = nn.ModuleList()
        # use same hidden dimension and output dimension
        self.cells.append(LSTMCell(d_input, d_hidden[0], d_hidden[0]))
        
        for i in range(1, len(d_hidden)):
            self.cells.append(LSTMCell(d_hidden[i - 1], d_hidden[i], d_hidden[i]))
    
        self.fc_out = nn.Linear(d_hidden[-1], d_output)
        nn.init.kaiming_normal_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor, batched: bool = False) -> torch.Tensor:
        for cell in self.cells:
            x = cell(x)
        
        return self.fc_out(x)

if __name__ == "__main__":
    model = LSTM(10, 2, [10, 10])
    print(model.cells)