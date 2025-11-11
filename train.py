import argparse
import logging
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from load_ptdata import get_data_loaders
from mlflow_callback import MLflowTorchCallback
from utils import build_model, evaluate

logger = logging.getLogger(__name__)

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI set to %s", tracking_uri)
    else:
        logger.info("MLFLOW_TRACKING_URI not set; using local `mlruns` directory.")
    mlflow.set_experiment(args.experiment_name)
    logger.info("MLflow experiment: %s", args.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Preparing data loaders (batch_size=%d)", args.batch_size)
    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)
    logger.info(
        "Data loaders ready: train=%d samples, val=%d samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    model = build_model(args.model_name).to(device)
    logger.info("Model '%s' moved to device", args.model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.register_model:
        logger.info("Model registry logging enabled.")
    else:
        logger.info("Model registry logging disabled; artifacts will not be registered.")
    mlflow_callback = MLflowTorchCallback(
        model_name=args.model_name,
        register=args.register_model,
    )

    logger.info("Starting training for %d epochs", args.epochs)
    with mlflow.start_run(run_name=f"{args.model_name}-training"):
        for epoch in range(1, args.epochs + 1):
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
            logger.info("Epoch %d: train_loss=%.4f", epoch, avg_loss)

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            logger.info("Epoch %d: val_loss=%.4f val_acc=%.4f", epoch, val_loss, val_acc)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_name": args.model_name,
        }
        metrics = {"final_val_loss": val_loss, "final_val_acc": val_acc}
        logger.info("Logging final metrics and model artifact to MLflow")
        mlflow_callback.log_model(model, params=params, metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)  # default=1 epoch for quick testing
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--experiment_name", type=str, default="torch_gpu_experiment")
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Attempt to register the trained model in the MLflow Model Registry.",
    )
    args = parser.parse_args()
    main(args)
