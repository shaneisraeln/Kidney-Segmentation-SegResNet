import torch
import torch.nn as nn
from monai.networks.nets import UNet as MONAIUNet

class UNet2D(nn.Module):
    """
    2D U-Net Baseline Architecture for Slice-Wise 3D CT Segmentation.
    Operates on 2D slices (B, C_in, H, W) and reconstructs 3D volumes slice-by-slice.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple = (16, 32, 64, 128, 256),
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 2
    ):
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.unet2d = MONAIUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="batch",
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor. Can be 2D slice (B, C, H, W) or 3D volume (B, C, D, H, W).
        Returns:
            Segmentation logits tensor.
        """
        if x.dim() == 5:
            # Volumetric input (B, C, D, H, W) -> process slice-by-slice along depth dimension D
            B, C, D, H, W = x.shape
            # Permute & reshape: (B, D, C, H, W) -> (B*D, C, H, W)
            x_slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            
            # Forward through 2D UNet
            logits_2d = self.unet2d(x_slices) # (B*D, C_out, H, W)
            
            # Reshape back to 3D volume: (B, D, C_out, H, W) -> (B, C_out, D, H, W)
            _, C_out, H_out, W_out = logits_2d.shape
            logits_3d = logits_2d.view(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
            return logits_3d
        elif x.dim() == 4:
            # 2D slice input (B, C, H, W)
            return self.unet2d(x)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")
