# py_torch_gpu_dock_mlflow
![version](https://img.shields.io/badge/version-1.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/unittests/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aunittests)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/flake8/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aflake8)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/docker-build/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Adocker-build)

## Introduction
This project packages a PyTorch training workflow as an MLflow Project, with a
CUDA-enabled Docker image for reproducible GPU or CPU execution. Training runs
log metrics, parameters, and the container version to MLflow, making it
straightforward to track experiments across environments.

## Ensure MLflow Tracking Server Availability
Set `MLFLOW_TRACKING_URI` to point at a reachable MLflow Tracking Server (e.g.,
`http://<host>:5000`). The helper script `project_driver.bash` builds an env
var list that includes this URI; update it to reflect your server
configuration, or unset it to fall back to the local `mlruns` directory.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow ui  # Optional: start a local UI if you need one
```

## Running a Training
1. Clone the repository and step into it:
   ```bash
   git clone https://github.com/<your-org>/py_torch_gpu_dock_mlflow.git
   cd py_torch_gpu_dock_mlflow
   ```
2. Build the Docker image, which tags both the semantic version from
   `VERSION` and `latest`:
   ```bash
   make build
   ```
3. Launch the MLflow Project run (auto-detects GPU availability and forwards
   basic parameters):
   ```bash
   ./project_driver.bash
   ```
The run records the image tag (e.g., `1.0.0`) in MLflow, so every experiment is
tied back to the exact container version.

## Configuring Trainings
Training defaults live in `train.py` and the MLproject entry point. Adjust
common options via command-line parameters passed by MLflow:

- Edit `project_driver.bash` to change `-P` arguments such as `epochs`,
  `batch_size`, or `learning_rate`.
- If invoking `python train.py` directly, supply flags like `--epochs 5
  --batch_size 64`.
- Toggle model registry logging with `--register-model`.
- Modify the dataset plumbing or model selection in `load_ptdata.py` and
  `utils.py` as your use case evolves.

## Makefile Usage
The `makefile` wraps frequent tasks:

- `make build` – build `torch-gpu-mlflow:<VERSION>` and retag as `latest`.
- `make env` – prepare the local virtual environment used for non-container runs.
- `make run` – execute `python train.py` locally.
- `make train` – convenience combo of `make env` and `make run`.
- `make unittest` – run unit tests under `tests/`.
- `make run_mlproject` – invoke the MLflow Project driver script.

Targets assume the `VERSION` file holds the desired semantic version; update it
manually when you cut a new release, then rebuild to emit the matching
container tag.
