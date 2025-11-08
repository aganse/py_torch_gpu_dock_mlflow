import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from load_ptdata import get_data_loaders
from utils import build_model, evaluate

class TestTrainLoop(unittest.TestCase):
    def test_one_epoch_training(self):
        device = torch.device("cpu")
        train_loader, val_loader = get_data_loaders(batch_size=4)
        model = build_model("resnet18").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        inputs, targets = next(iter(train_loader))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        self.assertTrue(loss.item() > 0)

    def test_evaluation_returns_metrics(self):
        device = torch.device("cpu")
        train_loader, val_loader = get_data_loaders(batch_size=4)
        model = build_model("resnet18").to(device)
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        self.assertTrue(0 <= val_acc <= 1)
        self.assertTrue(val_loss >= 0)

if __name__ == "__main__":
    unittest.main()
