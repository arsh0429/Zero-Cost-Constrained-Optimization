import torch.nn as nn
import random

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def generate_search_space(input_dim, output_dim, depth_range=(2, 5), width_range=(16, 128), samples=50):
    space = []
    for _ in range(samples):
        depth = random.randint(*depth_range)
        width = random.randint(*width_range)
        hidden_dims = [width] * depth
        model = MLP(input_dim, hidden_dims, output_dim)
        space.append(model)
    return space
