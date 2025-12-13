#!/bin/bash

# Set variable based on whether gpu is available on this system
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected on system."
    gpu_arg="-A gpus=all"
else
    echo "No GPU detected on system; running in CPU mode."
    gpu_arg=""
fi

mlflow run .                                            \
    ${gpu_arg}                                          \
    --experiment-name='torch_gpu_experiment'            \
    -A env="MLFLOW_TRACKING_URI=http://192.168.1.5:5000"\
    -P epochs=3                                         \
    -P batch_size=8                                     \
    -P learning_rate=1e-3                               \
    -P model_name=resnet18                              \
    -P log_level=INFO
