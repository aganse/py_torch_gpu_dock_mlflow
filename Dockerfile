FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# using this base image instead of this:
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# so that the final image comes out 5.6GB instead of 9.6GB!

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip bash awscli \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
 && "$VIRTUAL_ENV/bin/pip" install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN "$VIRTUAL_ENV/bin/pip" install -r requirements.txt

COPY . /app

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 5000
CMD ["python", "train.py"]
