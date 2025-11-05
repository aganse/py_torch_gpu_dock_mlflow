import torch
from torch import nn
from torch.utils.data import DataLoader
from src.model import SimpleClassifier
from src.dataset import SyntheticDataset


def test_single_training_step():
    ds = SyntheticDataset(num_samples=8, input_dim=16, num_classes=3, seed=1)
    loader = DataLoader(ds, batch_size=4)
    model = SimpleClassifier(input_dim=16, hidden=8, num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    xb, yb = next(iter(loader))
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()

    # basic sanity: loss is finite
    assert torch.isfinite(loss).item()
