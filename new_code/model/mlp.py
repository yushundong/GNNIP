import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Base(nn.Module):
    def __init__(self, input_dims, layer_sizes, output_size):
        super(MLP_Base, self).__init__()
        self.input_dims = input_dims
        self.layers = nn.ModuleList()

        # Create a list of fully connected layers for each input dimension
        for i, dim in enumerate(input_dims):
            self.layers.append(self.create_fc_layers(dim, layer_sizes[i]))

        self.fc_out = nn.Linear(sum(layer_sizes), output_size)  # Output layer

    def create_fc_layers(self, input_dim, layer_sizes):
        layers = []
        prev_size = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        return nn.Sequential(*layers)

    def forward(self, *inputs):
        assert len(inputs) == len(self.input_dims), "Number of inputs should match input dimensions"
        
        outputs = []
        for i, x in enumerate(inputs):
            x = x.view(-1, self.input_dims[i])
            x = self.layers[i](x)
            outputs.append(x)

        combined = torch.cat(outputs, dim=1)
        out = self.fc_out(combined)
        return out


class MLP_ATTACK(MLP_Base):
    def __init__(self, dim_in):
        # Use the base class with the given dimension
        super(MLP_ATTACK, self).__init__(input_dims=[dim_in], layer_sizes=[[128, 32]], output_size=2)


class MLP_ATTACK_PLUS(MLP_Base):
    def __init__(self, dim_in_1, dim_in_2):
        # Use the base class with two input dimensions
        super(MLP_ATTACK_PLUS, self).__init__(input_dims=[dim_in_1, dim_in_2], 
                                              layer_sizes=[[128, 64, 16], [64, 16]], 
                                              output_size=2)


class MLP_ATTACK_PLUS2(MLP_Base):
    def __init__(self, dim_in_1, dim_in_2):
        # Use the base class with two input dimensions
        super(MLP_ATTACK_PLUS2, self).__init__(input_dims=[dim_in_1, dim_in_2], 
                                               layer_sizes=[[16, 4], [128, 64, 16]], 
                                               output_size=2)


class MLP_ATTACK_ALL(MLP_Base):
    def __init__(self, dim_in_1, dim_in_2, dim_in_3):
        # Use the base class with three input dimensions
        super(MLP_ATTACK_ALL, self).__init__(input_dims=[dim_in_1, dim_in_2, dim_in_3], 
                                             layer_sizes=[[128, 64, 16], [128, 64, 16], [4]], 
                                             output_size=2)


class MLP_Target(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP_Target, self).__init__()
        self.fc1 = nn.Linear(dim_in, 32)
        self.fc2 = nn.Linear(32, dim_out)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
