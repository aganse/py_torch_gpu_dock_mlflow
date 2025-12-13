# py_torch_gpu_dock_mlflow
![version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/unittests/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aunittests)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/docker-build/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Adocker-build)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/flake8/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aflake8)

Ready-to-run Python/Torch/MLflow setup to train models on GPU (or CPU if needed)
and get them running quickly, logging performance and resulting model in MLflow,
keeping everything in Docker via MLflow Projects.

> <SUP>
> :bulb: Note this repo is part of a trio that you might find useful together
> (but all are separate tools that can be used independently):
> 
> * [aganse/docker_mlflow_db](https://github.com/aganse/docker_mlflow_db):
>     ready-to-run MLflow server with PostgreSQL, AWS S3, Nginx
> 
> * [aganse/py_torch_gpu_dock_mlflow](https://github.com/aganse/py_torch_gpu_dock_mlflow):
>     ready-to-run Python/Torch/MLflow setup to train models on GPU
> 
> * [aganse/vim_mlflow](https://github.com/aganse/vim-mlflow):
>     a Vim plugin to browse the MLflow parameters and metrics instead of GUI
> </SUP>
<P>&nbsp;<P>


## Ensuring/connecting MLflow tracking server availability
Set `MLFLOW_TRACKING_URI` to point at a reachable MLflow Tracking Server (e.g.,
`http://<host>:5000`). The helper script `project_driver.bash` builds an env
var list that includes this URI; update it to reflect your server
configuration, or unset it to fall back to the local `mlruns` directory.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow ui  # Optional: start a local UI if you need one
```

## Running a training
1. Clone the repository and step into it:
   ```bash
   git clone https://github.com/aganse/py_torch_gpu_dock_mlflow.git
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

## Configuring trainings
Training defaults live in `train.py` and the MLproject entry point. Adjust
common options via command-line parameters passed by MLflow:

- Edit `project_driver.bash` to change `-P` arguments such as `epochs`,
  `batch_size`, or `learning_rate`.
- If invoking `python train.py` directly, supply flags like `--epochs 5
  --batch_size 64`.
- Toggle model registry logging with `--register-model`.
- Modify the dataset plumbing or model selection in `load_ptdata.py` and
  `utils.py` as your use case evolves.

## Makefile usage
The `makefile` wraps frequent tasks:

- `make build` – build `torch-gpu-mlflow:<VERSION>` training image and retag as `latest`.
- `make run_mlproject` – invoke the MLflow Project driver script (uses `torch-gpu-mlflow:latest` training image).
- `make unittest` – run unit tests under `tests/`.
- `make dev_env` – export variables into local virtual environment (for non-container/dev runs).
- `make dev_install` – install the python package dependencies into current environment
- `make dev_train` – execute `python train.py` locally.

The `VERSION` file holds the version in semver; update it manually when you cut
a new release, then rebuilding (make build) applies version as container tag.
