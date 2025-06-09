import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    """
    Time embedding layer that transforms a time step into a higher-dimensional space.
    This is typically used in diffusion models to condition the model on the time step.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the time embedding layer.
        Args:
            t (torch.Tensor): Input tensor representing the time step, shape (1, embed_dim).
        Returns:
            torch.Tensor: Transformed tensor with shape (1, 4 * embed_dim).
        """
        # t: (1, embed_dim)

        # (1, embed_dim) -> (1, 4 * embed_dim)
        x = self.linear_1(t)

        # (1, 4 * embed_dim) -> (1, 4 * embed_dim)
        x = F.silu(x)

        # (1, 4 * embed_dim) -> (1, 4 * embed_dim)
        x = self.linear_2(x)

        # (1, 1280)
        return x


class UNetResidualBlock(nn.Module):
    """Residual block for UNet architecture that processes features and time embeddings.
    This block applies group normalization, convolution, and a linear transformation for time embeddings.
    """

    def __init__(self, in_channels: int, out_channels: int, n_time=1280) -> None:
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.residual_layer: nn.Identity | nn.Conv2d
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        Args:
            feature (torch.Tensor): Input feature tensor with shape (Batch_size, In_Channels, Height, Width).
            t (torch.Tensor): Time embedding tensor with shape (1, 1280).
        Returns:
            torch.Tensor: Output tensor with shape (Batch_size, Out_Channels, Height, Width).
        """
        # feature: (Batch_size, In_Channels, Height, Width)
        # t: (1, 1280)

        residue = feature
        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        t = self.linear_time(t)
        merged = (
            feature + t[:, :, None, None]
        )  # Reshape to (Batch_size, Out_Channels, 1, 1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNetAttentionBlock(nn.Module):
    """Attention block for UNet architecture that applies self-attention and cross-attention.
    This block processes features and context embeddings using group normalization,
    convolution, and attention mechanisms.
    """

    def __init__(self, n_head: int, embed_dim: int, context_dim: int = 768) -> None:
        super().__init__()
        channels = embed_dim * n_head

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, context_dim, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention block.
        Args:
            x (torch.Tensor): Input feature tensor with shape (Batch_size, Channels, Height, Width).
            context (torch.Tensor): Context tensor with shape (Batch_size, Sequence_Length, Dim).
        Returns:
            torch.Tensor: Output tensor with shape (Batch_size, Channels, Height, Width).
        """
        # x: (Batch_size, Channels, Height, Width)
        # context: (Batch_size, Sequence_Length, Dim)

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h * w).transpose(
            -1, -2
        )  # (Batch_size, Height * Width, Channels)

        # Normalization + SelfAttention with skip connection
        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization + CrossAttention with skip connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # Normalization + Ge GLU activation with skip connection

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2).view(
            n, c, h, w
        )  # (Batch_size, Channels, Height, Width)

        x = self.conv_output(x)
        x += residue_long  # Skip connection
        return x  # (Batch_size, Channels, Height, Width)


class UpSample(nn.Module):
    """
    Upsampling layer that doubles the height and width of the input tensor
    using nearest neighbor interpolation, followed by a convolutional layer.
    This is typically used in UNet architectures to increase the spatial resolution
    of feature maps during the decoding phase.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling layer.
        Args:
            x (torch.Tensor): Input tensor with shape (Batch_size, Channels, Height, Width).
        Returns:
            torch.Tensor: Output tensor with shape (Batch_size, Channels, Height * 2, Width * 2).
        """
        # x: (Batch_size, Channels, Height, Width) -> (Batch_size, Channels, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # (Batch_size, Channels, Height * 2, Width * 2) -> (Batch_size, Channels, Height * 2, Width * 2)
        x = self.conv(x)
        return x


class SwitchSequential(nn.Sequential):
    """
    A sequential container that allows for different types of layers to be used
    in a single forward pass. This is useful for models that have different
    types of layers that need to be applied to the input.
    """

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture that consists of multiple encoder and decoder blocks.
    This architecture is typically used in image segmentation tasks and consists of
    a series of residual blocks and attention blocks to process the input features
    and generate output features at multiple scales.
    The architecture includes downsampling and upsampling layers to maintain the spatial resolution
    of the input while extracting features at different levels of abstraction.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 320, Height / 8, Width / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 320, Height / 16, Width / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)
                ),
                # (Batch_size, 640, Height / 16, Width / 16) -> (Batch_size, 640, Height / 32, Width / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)
                ),
                # (Batch_size, 1280, Height / 32, Width / 32) -> (Batch_size, 1280, Height / 64, Width / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(UNetResidualBlock(1280, 1280)),
                # (Batch_size, 1280, Height / 64, Width / 64) -> (Batch_size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNetResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (Batch_size, 2560, Height / 64, Width / 64) -> (Batch_size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                SwitchSequential(UNetResidualBlock(2560, 1280), UpSample(1280)),
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNetResidualBlock(1920, 1280),
                    UNetAttentionBlock(8, 160),
                    UpSample(1280),
                ),
                SwitchSequential(
                    UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNetResidualBlock(960, 640),
                    UNetAttentionBlock(8, 80),
                    UpSample(640),
                ),
                SwitchSequential(
                    UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
            ]
        )


class UNetOutputLayer(nn.Module):
    """
    Output layer for the UNet architecture that processes the final feature maps
    and produces the output tensor. This layer typically applies group normalization,
    activation, and a convolutional layer to transform the feature maps into the desired output shape.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the output layer.
        Args:
            x (torch.Tensor): Input tensor with shape (Batch_size, In_Channels, Height / 8, Width / 8).
        Returns:
            torch.Tensor: Output tensor with shape (Batch_size, Out_Channels, Height / 8, Width / 8).
        """
        # x: (Batch_size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (Batch_size, 4, Height / 8, Width / 8)
        return x


class DiffusionModel(nn.Module):
    """
    Diffusion model that combines time embeddings, UNet architecture, and an output layer.
    This model is typically used in generative tasks where the input is a latent representation
    and the output is a generated image or feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the diffusion model.
        Args:
            latent (torch.Tensor): Input latent tensor with shape (Batch_size, 4, Height / 8, Width / 8).
            context (torch.Tensor): Context tensor with shape (Batch_size, Sequence_Length, Dim).
            t (torch.Tensor): Time step tensor with shape (1, 320).
        Returns:
            torch.Tensor: Output tensor with shape (Batch_size, 4, Height / 8, Width / 8).
        """
        # latent: (Batch_size, 4, Height / 8, Width / 8)
        # context: (Batch_size, Sequence_Length, Dim)
        # t: (1, 320)

        # (1, 320) -> (1, 1280) # 4 * 320 = 1280
        t = self.time_embedding(t)

        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, t)

        # (Batch_size, 320, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        output = self.final(output)

        return output
