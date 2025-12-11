import logging

import mlflow
import mlflow.pytorch
from mlflow.exceptions import MlflowException
import torch

logger = logging.getLogger(__name__)

class MLflowTorchCallback:
    def __init__(self, model_name: str, register: bool = False):
        self.model_name = model_name
        self.register = register

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model", params: dict = None, metrics: dict = None):
        if params:
            project_logged = {"epochs", "batch_size", "learning_rate", "model_name"}
            for k, v in params.items():
                if k in project_logged:
                    continue
                mlflow.log_param(k, v)
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        log_args = {"artifact_path": artifact_path}
        if self.register:
            log_args["registered_model_name"] = self.model_name

        registered_successfully = False
        try:
            mlflow.pytorch.log_model(model, **log_args)
            registered_successfully = self.register
        except MlflowException as exc:
            if self.register:
                exc_msg = getattr(exc, "message", str(exc))
                logger.warning(
                    "Model registry logging failed (%s). Falling back to artifact logging only.",
                    exc_msg,
                )
                mlflow.pytorch.log_model(model, artifact_path=artifact_path)
                registered_successfully = False
            else:
                raise

        if registered_successfully:
            logger.info("Model logged and registered in MLflow as '%s'", self.model_name)
        else:
            logger.info("Model logged in MLflow artifact path '%s'", artifact_path)
