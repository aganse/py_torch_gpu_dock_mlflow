import unittest
from load_ptdata import get_data_loaders

class TestDataLoaders(unittest.TestCase):
    def test_loaders_return_tensors(self):
        train_loader, val_loader = get_data_loaders(batch_size=4)
        x, y = next(iter(train_loader))
        self.assertEqual(x.shape[0], 4)
        self.assertEqual(len(x), len(y))
        self.assertTrue(x.dtype.is_floating_point)
        self.assertTrue(y.dtype in (int, y.dtype))  # int labels

if __name__ == "__main__":
    unittest.main()
