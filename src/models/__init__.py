from .unet2d import UNet2D
from .unet3d import UNet3D
from .segresnet import SegResNet3D

def get_model(model_name: str, in_channels: int = 1, out_channels: int = 4):
    name = model_name.lower()
    if name in ["unet2d", "baseline_2d", "2d_unet"]:
        return UNet2D(in_channels=in_channels, out_channels=out_channels)
    elif name in ["unet3d", "3d_unet"]:
        return UNet3D(in_channels=in_channels, out_channels=out_channels)
    elif name in ["segresnet", "segresnet3d"]:
        return SegResNet3D(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

__all__ = ['UNet2D', 'UNet3D', 'SegResNet3D', 'get_model']
