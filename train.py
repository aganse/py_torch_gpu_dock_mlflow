import argparse
from project_driver import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--experiment_name", type=str, default="torch_gpu_experiment")
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Attempt to register the trained model in the MLflow Model Registry.",
    )
    args = parser.parse_args()
    main(args)
