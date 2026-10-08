import torch
from src.data.transforms import gray_world


def test_gray_world_shape_dtype():
    x = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    y = gray_world(x)
    assert y.shape == x.shape and y.dtype == torch.uint8
