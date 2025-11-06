# Default (GPU-enabled) base image - useful if you want the same image to support GPUs when available
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

# CPU-only alternative - uncomment or use a different build arg for a smaller image
# FROM python:3.11-slim
# RUN pip install --no-cache-dir "torch==2.2.0+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy only what we need for pip install first (optional speed-up in rebuilds)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src
COPY README.md /app/README.md

# Default command runs the training example (override at runtime)
CMD ["python", "-m", "src.train"]
