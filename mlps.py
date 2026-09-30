import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    Simple 2 x 256 MLP used for scalar-field baselines.

    The network structure is [3, 256, 256, 1], matching the two-hidden-layer
    architecture reported in the manuscript.

    only beta parameter in the Softplus activation function can be changed
    """
    def __init__(self, beta):
        super(SimpleMLP, self).__init__()
        self.beta = beta
        self.activation = nn.Softplus(beta=self.beta) 
        #self.activation = nn.LeakyReLU()
        self.layers = nn.Sequential(
            nn.Linear(3, 256),  
            self.activation,         
            nn.Linear(256, 256), 
            self.activation,    
            nn.Linear(256, 1)   
        )

    def forward(self, x):
        return self.layers(x)


# used in implicit neural representation
class ConcatMLP(nn.Module):
    """
    MLP used by the implicit fields.

    ``n_hidden_layers`` is the total number of hidden layers. Therefore,
    ``n_hidden_layers=2`` and ``hidden_dim=256`` implement the manuscript's
    2 x 256 architecture. With ``concat=True``, the original input is joined
    to each hidden representation before the next layer.
    """
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_hidden_layers,
                 activation,
                 beta, 
                 concat):
        super(ConcatMLP, self).__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be at least 1")
        self.layers = nn.ModuleList()
        self.beta = beta
        if activation == 'Softplus':
            self.activation = nn.Softplus(beta=self.beta)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        else:
            print('Activation function not recognized. Using Softplus, ReLU, LeakyReLU, Tanh, Sigmoid, ELU.')
        self.concat = concat

        # The first layer is the first hidden layer, not an extra projection.
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if self.concat:
            hidden_input_dim = in_dim + hidden_dim
            for _ in range(n_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_input_dim, hidden_dim))
            self.output = nn.Linear(hidden_input_dim, out_dim)

        else:
            for _ in range(n_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.output = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x_target = x
        hidden = self.activation(self.layers[0](x))

        if self.concat:
            hidden = torch.cat((x_target, hidden), dim=1)
            for layer in self.layers[1:]:
                hidden = self.activation(layer(hidden))
                hidden = torch.cat((x_target, hidden), dim=1)
            return self.output(hidden)

        for layer in self.layers[1:]:
            hidden = self.activation(layer(hidden))
        return self.output(hidden)
