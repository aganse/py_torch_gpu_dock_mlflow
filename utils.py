import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str):
    if model_name == "resnet18":
        return models.resnet18(pretrained=False, num_classes=10)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=False, num_classes=10)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return total_loss / len(dataloader), correct / total
