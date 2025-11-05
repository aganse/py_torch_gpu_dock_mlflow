# import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .device import get_device
from .model import SimpleClassifier
from .dataset import SyntheticDataset


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def main():
    device = get_device()
    print("Using device:", device)
    input_dim = 32
    model = SimpleClassifier(input_dim=input_dim)
    model.to(device)

    ds = SyntheticDataset(
            num_samples=64,
            input_dim=input_dim,
            num_classes=10,
            seed=42
         )
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # MLflow: use local mlruns by default unless MLflow env is set externally
    with mlflow.start_run():
        mlflow.log_param("device", str(device))
        mlflow.log_param("batch_size", 16)
        # Run a single quick epoch for smoke-test
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        mlflow.log_metric("train_loss", loss)

        # Log model
        mlflow.pytorch.log_model(model, "model")

    print("Done. train_loss:", loss)


if __name__ == "__main__":
    main()
