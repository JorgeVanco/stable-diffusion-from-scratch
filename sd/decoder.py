import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """Attention block for the Variational Autoencoder (VAE)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention block."""
        return self.attention(x)


class VAE_ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and group normalization."""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer: nn.Identity | nn.Conv2d
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        # x: (Batch_size, In_Channels, Height, Width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)
