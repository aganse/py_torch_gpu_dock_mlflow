import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from load_ptdata import get_data_loaders
from mlflow_callback import MLflowTorchCallback
from utils import build_model, evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)
    model = build_model(args.model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    mlflow.set_experiment(args.experiment_name)
    mlflow_callback = MLflowTorchCallback(model_name=args.model_name, register=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: train_loss = {avg_loss:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")

        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

    params = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "model_name": args.model_name}
    metrics = {"final_val_loss": val_loss, "final_val_acc": val_acc}
    mlflow_callback.log_model(model, params=params, metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--experiment_name", type=str, default="torch_gpu_experiment")
    args = parser.parse_args()
    main(args)
