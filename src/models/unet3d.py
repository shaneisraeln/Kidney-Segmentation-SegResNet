import torch
import torch.nn as nn
from monai.networks.nets import UNet as MONAIUNet

class UNet3D(nn.Module):
    """
    Improved 3D U-Net Architecture with Residual Units and Instance Normalization
    for Volumetric CT Kidney, Kidney Mass, and Cyst Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple = (32, 64, 128, 256, 512),
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1
    ):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.unet3d = MONAIUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="instance",
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volumetric tensor of shape (B, C, D, H, W)
        Returns:
            Logits tensor of shape (B, C_out, D, H, W)
        """
        return self.unet3d(x)
