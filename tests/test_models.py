import pytest
import torch
from src.models import UNet2D, UNet3D, SegResNet3D

def test_unet2d_forward():
    model = UNet2D(in_channels=1, out_channels=4, channels=(16, 32, 64))
    # 2D slice test
    x_2d = torch.randn(2, 1, 64, 64)
    out_2d = model(x_2d)
    assert out_2d.shape == (2, 4, 64, 64), f"Expected (2, 4, 64, 64), got {out_2d.shape}"

    # 3D volume test
    x_3d = torch.randn(2, 1, 16, 64, 64)
    out_3d = model(x_3d)
    assert out_3d.shape == (2, 4, 16, 64, 64), f"Expected (2, 4, 16, 64, 64), got {out_3d.shape}"

def test_unet3d_forward():
    model = UNet3D(in_channels=1, out_channels=4, channels=(16, 32, 64))
    x_3d = torch.randn(1, 1, 32, 32, 32)
    out_3d = model(x_3d)
    assert out_3d.shape == (1, 4, 32, 32, 32), f"Expected (1, 4, 32, 32, 32), got {out_3d.shape}"

def test_segresnet_forward():
    model = SegResNet3D(in_channels=1, out_channels=4, init_filters=16)
    x_3d = torch.randn(1, 1, 32, 32, 32)
    out_3d = model(x_3d)
    assert out_3d.shape == (1, 4, 32, 32, 32), f"Expected (1, 4, 32, 32, 32), got {out_3d.shape}"

if __name__ == "__main__":
    pytest.main([__file__])
