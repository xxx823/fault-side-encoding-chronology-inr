import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    this is a simple MLP model, the network structure is [3, 256, 256, 256, 1] which is fixed
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
            nn.Linear(256, 256), 
            self.activation,   
            nn.Linear(256, 1)   
        )

    def forward(self, x):
        return self.layers(x)


# used in implicit neural representation
class ConcatMLP(nn.Module):
    """
    concatenate the input features with the hidden layer features as an enhanced feature
    the neural network structure is flexible, where
    in_dim: the input dimension, the coodinates plus fault features
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
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
        self.num_layers = 2 + n_hidden_layers
        self.concat = concat

        # input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if self.concat:
            # hidden layers
            h_dim_concat = in_dim + hidden_dim
            for i in range(n_hidden_layers):
                self.layers.append(nn.Linear(h_dim_concat, h_dim_concat))
                #h_dim_concat *= 2  # concatenate all the former layer's features 
                h_dim_concat = in_dim + h_dim_concat # only concatenate the input with the hidden layer features
            # output layer
            self.layers.append(nn.Linear(h_dim_concat, out_dim))

        else:
            # hidden layers
            for i in range(n_hidden_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # output layer
            self.layers.append(nn.Linear(hidden_dim, out_dim))
    
    def forward(self, x):
        x_target = x  # Keep the original input features for concatenation

        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Only apply activation and concatenation to hidden layers (not the last layer)
            if i < len(self.layers) - 1:
                if self.concat:
                    # Concatenate the input with the hidden layer output
                    x = torch.cat((x_target, x), dim=1)
                x = self.activation(x)
        
        return x
    

