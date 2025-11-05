import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Small synthetic dataset that yields random data for unit tests and smoke
    runs.  Deterministic when seed is set by caller.
    """

    def __init__(
            self,
            num_samples=128,
            input_dim=32,
            num_classes=10,
            seed=None
            ):
        if seed is not None:
            torch.manual_seed(seed)
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
