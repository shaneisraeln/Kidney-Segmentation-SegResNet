import torch
import torch.nn as nn
from monai.networks.nets import SegResNet as MONAISegResNet

class SegResNet3D(nn.Module):
    """
    3D SegResNet Architecture for Volumetric CT Kidney and Tumor Segmentation.
    Advanced Residual Encoder-Decoder network.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        init_filters: int = 16,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        dropout_prob: float = 0.2
    ):
        super(SegResNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.net = MONAISegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volumetric tensor (B, C, D, H, W)
        Returns:
            Logits tensor (B, C_out, D, H, W)
        """
        return self.net(x)
