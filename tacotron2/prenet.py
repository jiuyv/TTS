import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias = True, w_init_gain = 'linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias = bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain = nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias = False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p = 0.5, training = True)
        return x