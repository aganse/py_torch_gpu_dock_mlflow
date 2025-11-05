# import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """
    Simple 2-layer MLP for small synthetic data (keeps tests and CI fast).
    Expects input vectors of size `input_dim`.
    """
    def __init__(self, input_dim=32, hidden=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
