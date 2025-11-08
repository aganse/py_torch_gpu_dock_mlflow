import unittest
import torch
from utils import build_model

class TestModel(unittest.TestCase):
    def test_build_model_output_shape(self):
        model = build_model("resnet18")
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 10)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            build_model("invalid_model")

if __name__ == "__main__":
    unittest.main()
