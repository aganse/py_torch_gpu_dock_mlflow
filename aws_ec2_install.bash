#!/usr/bin/env bash
sudo apt-get update && sudo apt-get install -y build-essential git awscli python3-venv python3-pip && sudo rm -rf /var/lib/apt/lists/*
python3 -m venv mlflow_env
source mlflow_env/bin/activate
pip install --upgrade pip
pip install "mlflow>=3.0.0" torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
echo "AWS EC2 install complete."
