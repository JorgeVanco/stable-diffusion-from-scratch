import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """Attention block for the Variational Autoencoder (VAE)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention block."""
        # x: (Batch_size, In_Channels, Height, Width)
        residue = x

        # (Batch_size, In_Channels, Height, Width) -> (Batch_size, In_Channels, Height, Width)
        x = self.group_norm(x)

        b, c, h, w = x.shape

        # (Batch_size, In_Channels, Height, Width) -> (Batch_size, In_Channels, Height * Width)
        x = x.view(b, c, h * w)

        # (Batch_size, In_Channels, Height * Width) -> (Batch_size, Height * Width, In_Channels)
        x = x.transpose(-1, -2)

        # (Batch_size, Height * Width, In_Channels) -> (Batch_size, Height * Width, In_Channels)
        x = self.attention(x)

        # (Batch_size, Height * Width, In_Channels) -> (Batch_size, In_Channels, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_size, In_Channels, Height * Width) -> (Batch_size, In_Channels, Height, Width)
        x = x.view(b, c, h, w)

        x += residue
        return x


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


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8,  Width / 8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 4,  Width / 4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 2,  Width / 2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height,  Width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        # x: (Batch_size, In_Channels, Height / 8, Width / 8)

        x /= 0.18215  # Normalize input

        for module in self:
            x = module(x)
        return x
