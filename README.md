# py_torch_gpu_dock_mlflow
![version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/unittests/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aunittests)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/docker-build/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Adocker-build)
[![](https://github.com/aganse/py_torch_gpu_dock_mlflow/workflows/flake8/badge.svg)](https://github.com/aganse/py_torch_gpu_dock_mlflow/actions?query=workflow%3Aflake8)

This is a ready-to-run Python/Torch/MLflow setup to train models on GPU (or CPU
if needed) and get them running quickly, logging performance and resulting model
in MLflow, keeping everything in Docker via MLflow Projects.

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


## Running a training

(Note this repo has been written and tested on Linux and MacOs.
It might not work out of the box in Windows, but using WSL is best bet there.)

1. Ensure you're set to connect your MLflow tracking server:
   Set `MLFLOW_TRACKING_URI` to point at a reachable MLflow Tracking Server (e.g.,
   `http://<host>:5000`). The helper script `project_driver.bash` sets an env
   var for URI; update that to reflect your server configuration, or unset it to
   fall back to creating/using a local `mlruns` directory.

   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   mlflow ui  # Optional: start a local UI if you need one
   ```

2. If using a GPU, verify that it is already working on your system and in Docker.
(If doing small/test/dev runs via CPU-only then you can skip this step.)
   ```bash
   > lspci | grep -i nvidia
   01:00.0 VGA compatible controller: NVIDIA Corporation TU104 [GeForce RTX 2080 SUPER] (rev a1)
   01:00.1 Audio device: NVIDIA Corporation TU104 HD Audio Controller (rev a1)
   01:00.2 USB controller: NVIDIA Corporation TU104 USB 3.1 Host Controller (rev a1)
   01:00.3 Serial bus controller [0c80]: NVIDIA Corporation TU104 USB Type-C UCSI Controller (rev a1)
   ```
   Then verify your nvidia-docker installation, e.g.:

   ```bash
   > docker run --gpus all --rm nvidia/cuda nvidia-smi
   Sun Jun  5 16:31:20 2022
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
   | 18%   26C    P8     4W / 250W |    134MiB /  7982MiB |      0%      Default |
   |                               |                      |                  N/A |
   +-------------------------------+----------------------+----------------------+

   +-----------------------------------------------------------------------------+
   | Processes:                                                                  |
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   |        ID   ID                                                   Usage      |
   |=============================================================================|
   +-----------------------------------------------------------------------------+
   ```

3. Match up to your MLflow server's expected artifact filestore location:
   The local host filesystem location set in docker_env.volumes in MLproject
   should match the ARTIFACTS env var setting used in your MLflow server if
   a host-system filesystem is being used for artifacts.  Or if your MLflow
   server uses S3 for those artifacts it should "just work".  (The MLproject
   volume mapping may still be useful/relevant to you to ensure your training
   data location is accessible to the training container.)

4. Clone this repository and step into it:
   ```bash
   git clone https://github.com/aganse/py_torch_gpu_dock_mlflow.git
   cd py_torch_gpu_dock_mlflow
   ```

5. Build the Docker image, which tags the semantic version from `VERSION`
as well as `latest`:
   ```bash
   make build
   ```
   By default the image set in MLproject is that `latest` tag, but alternately
   you might choose to set it to a specific tagged version you built.

6. Launch the MLflow Project run (auto-detects GPU availability and forwards
   basic parameters):
   ```bash
   make run_mlproject   # (equivalent to simply ./project_driver.bash)
   ```
   The run records the image tag (e.g., `1.0.0`) in MLflow, so every experiment
   is tied back to the exact container version.


## Configuring trainings
Hardwired training defaults live in `train.py` and the MLproject entry point.
Adjust common options via command-line parameters passed by MLflow:

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
- `make run_mlproject` – invoke the MLflow Project driver script (uses training image specified in MLproject).
- `make unittest` – run unit tests under `tests/`.
- `make dev_env` – export variables into local virtual environment (for non-container/dev runs).
- `make dev_install` – install the python package dependencies into current environment
- `make dev_train` – execute `python train.py` locally.

The `VERSION` file holds the version in semver; update it manually when you cut
a new release, then rebuilding (make build) applies version as container tag.


## Notes

1. For a number of reasons I specific chose not to use the --build-image arg in the
   `mlflow run` call in project_driver.bash.  In the ideal that arg is conceptually
   a nice idea - it's like running `docker build...` for the training image before
   every run, and that's useful to catch any code updates.  However:

   a.) When using --build-image, MLflow builds a fresh image and tags it as
   <project-name>:<short-commit>.  It doesn't use the image name in MLproject's
   docker_env.image because the point of --build-image is to regenerate an image
   that's tied to the current source revision.  However, if you tweak one of your
   .py files and rerun the training before git-committing that change, the image
   name (which is tagged with the latest git-commit hash) does not match the state
   of the code which is confusing/misleading.  At least with the manual approach
   the specific image name listed in MLproject clearly points to what was used
   for the training.  (Also with that image naming convention for --build-image,
   the mlflow.docker.image.uri logged in MLflow essentially duplicates the mlflow.source.git.commit and mlflow.source.git.repoURL tags).

   b.) If you want the run logging level to stay at INFO, you won't see any build
   progress at all, and it just sits there for a long time - if you want to see
   build progress with --build-image you must use DEBUG for your log level which
   likely is not what you want for all the rest of the run.

   c.) Lastly when using --build-image the run just sits there for literally
   minutes after the "No GPU detected on system; running in CPU mode" message
   before the build output line appears.  No idea what it's doing in that time!

   So approach here is doing manual build (ie run `make build`) after code update
   before running a training, and use the image name set in MLproject.  If you do
   prefer --build-image, you can simply add that arg to the `mlflow run` call in
   `project_driver.bash` with no other changes and that will work.

2. It appears that while the MLflow Projects framework is still available in
   MLflow 3.x, it has stopped being documented further and being actively expanded.
   No deprecation warnings yet, but this does seem to suggest that the Projects
   framework aspect of this work here may be of less interest than just the
   modeling and logging to MLflow itself - still important those!  But just fyi.
   Myself I'm a big fan of MLflow Projects for keeping things organized and
   easy to enable a team to deploy a bunch of training experiments quickly,
   so hopefully this framework remains for some time!

3. If I understand correctly, instead of `mlflow run .` in `project_driver.bash`
   one could do `mlflow run https://github.com/aganse/py_torch_gpu_dock_mlflow` or
   `mlflow run https://github.com/aganse/py_torch_gpu_dock_mlflow -v feature/mybranch`,
   and have locally only this `project_driver.bash` file (without rest of repo) and
   the training image.
   And for that training image, instead of `image: torch-gpu-mlflow:latest` in
   MLproject, referring to an image stored locally, one could use a uri to an online
   package location like e.g.  `image: ghcr.io/aganse/torch-gpu-mlflow:latest` to
   pull a prebuilt image saved there (say being a result of a CI build).
   Haven't done any of these things in this repo yet though.
