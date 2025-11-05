import torch
from src.model import SimpleClassifier


def test_model_forward_shapes():
    batch = 8
    input_dim = 32
    x = torch.randn(batch, input_dim)
    model = SimpleClassifier(input_dim=input_dim, hidden=16, num_classes=5)
    out = model(x)
    assert out.shape == (batch, 5)
