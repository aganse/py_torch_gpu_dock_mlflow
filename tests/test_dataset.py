from torch.utils.data import DataLoader
from src.dataset import SyntheticDataset


def test_dataset_and_dataloader_shapes():
    ds = SyntheticDataset(num_samples=20, input_dim=16, num_classes=3, seed=0)
    loader = DataLoader(ds, batch_size=4)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, 16)
    assert yb.shape == (4,)
