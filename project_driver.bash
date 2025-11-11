#!/bin/bash

# After development is frozen/slowed, we can also run trainings directly off
# the repo without cloning it locally; ie for just trying different parameters.
# Only this driver script is needed in that case, as the rest gets pulled from
# repo.  According to mlflow documentation a git commit or branch name at that
# repo uri can be specified to use like `-v abcde123` or `-v feature/mybranch`.
# I've not tried this yet though.  (Without -v and get the default branch.)
# mlflow run https://github.com/aganse/py_tf2_gpu_dock_mlflow -v abcde123 ...
# For more details see https://mlflow.org/docs/1.30.0/projects.html#running-projects

# Some possibilities to set in MLFLOW_TRACKING_URI in calling environment, or
# could set these in mlflow run args per below, but I'm just using the env var.
# -A add-host=host.docker.internal:host-gateway       \
# -A env=MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
# -A env=MLFLOW_TRACKING_URI=http://172.17.0.1:5000 \
# -A env=MLFLOW_TRACKING_URI=http://192.168.65.2:5000 \


# Set variable based on whether gpu is available on this system
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected on system."
    gpu_arg="-A gpus=all"
else
    echo "No GPU detected on system."
    gpu_arg=""
fi

mlflow run .                                            \
    ${gpu_arg}                                          \
    --experiment-name='torch_gpu_experiment'            \
    -P epochs=1                                         \
    -P batch_size=8                                     \
    -P learning_rate=1e-3                               \
    -P model_name=resnet18



    # Params set in py_tf2_gpu_dock_mlflow:
    # -b local                                            \
    # --experiment-name='Malaria Detection'               \
    # -P run_name='malaria'                               \
    # -P randomize_images=True                            \
    # -P convolutions=0                                   \
    # -P epochs=10                                        \
    # -P batch_size=128                                   \
    # -P training_samples=13779                           \
    # -P validation_samples=13779

    # --build-image  # seems broken in mlflow 2.4.1 so  \
    #                # added pre-build line in makefile for now
    # -P batch_size=10                                  \
    # -P training_samples=100                           \
    # -P validation_samples=100

