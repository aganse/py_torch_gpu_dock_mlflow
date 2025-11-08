import mlflow
import mlflow.pytorch
import torch

class MLflowTorchCallback:
    def __init__(self, model_name: str, register: bool = False):
        self.model_name = model_name
        self.register = register

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model", params: dict = None, metrics: dict = None):
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        mlflow.pytorch.log_model(model, name=self.model_name, artifact_path=artifact_path)

        if self.register:
            model_uri = f"models:/{self.model_name}/latest"
            mlflow.register_model(model_uri=model_uri, name=self.model_name)

        print(f"Model logged in MLflow with name={self.model_name}")
