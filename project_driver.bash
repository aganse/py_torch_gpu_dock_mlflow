#!/bin/bash

# It appears that while MLflow Projects is still available in mlflow 3.x, it
# has stopped being documented further and being actively expanded.  No
# deprecation warnings yet, but this does seem to suggest that the Projects
# framework aspect of this work here is of less interest than just the
# modeling and logging to MLflow itself - still important those!  Just fyi.

# Set variable based on whether gpu is available on this system
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected on system."
    gpu_arg="-A gpus=all"
else
    echo "No GPU detected on system."
    gpu_arg=""
fi

# Concatenate envvars to set in mlflow run
varslist=""
varslist+="MLFLOW_TRACKING_URI=http://192.168.1.5:5000"
varslist+=","
varslist+="GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git"

mlflow run .                                            \
    ${gpu_arg}                                          \
    --experiment-name='torch_gpu_experiment'            \
    -A env=${varslist}                                  \
    -P epochs=20                                        \
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

