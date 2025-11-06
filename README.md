# py_torch_gpu_dock_mlflow

> [!WARNING]  
> Development in progress on this branch; not expected to work plus fyi I'm
> using AI code generation tools (including for README contents). Don't use
> till reviewed/tested/merged into main.


This repository is a PyTorch equivalent of py_tf2_gpu_dock_mlflow. It provides:
- A minimal PyTorch training example that is compatible with both GPU and CPU hosts.
- MLflow (v3.x) model logging via mlflow.pytorch.
- Dockerfile configured by default to use a GPU-enabled base image (but the code falls back to CPU if CUDA is not available).
- Unit tests and a GitHub Actions workflow that run tests on pushes and PRs.

Quick start (CPU):
1. Build:
   docker build -t py-torch-mlflow .
2. Run (CPU):
   docker run --rm -e FORCE_CPU=1 py-torch-mlflow python -m src.train

Quick start (GPU host):
1. Build on a host that has Docker and GPU access (image defaults to CUDA-enabled base).
2. Run with GPU access:
   docker run --gpus all --rm py-torch-mlflow python -m src.train

Device selection:
- The code uses src.device.get_device() which honors the FORCE_CPU environment variable and falls back to CPU when torch.cuda.is_available() is False.

CI:
- The repository contains .github/workflows/unittests.yml which runs pytest on pushes and PRs (forced to CPU in the job environment).

Notes:
- Tests are small and deterministic. They do not attempt to log to an MLflow server. Full end-to-end runs intended to populate MLflow should be executed inside Docker on an appropriate host or in a controlled environment.
