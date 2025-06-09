import torch
from torch import nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Self-Attention mechanism for processing sequences."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """Forward pass through the self-attention layer."""
        input_shape = x.shape
        batch_size, sequenze_length, _ = (
            input_shape  # x shape: [batch_size, seq_len, embed_dim]
        )

        intermidiate_shape = (
            batch_size,
            sequenze_length,
            self.num_heads,
            self.head_dim,
        )

        q, k, v = self.in_proj(x).chunk(
            3, dim=-1
        )  # q,k,v shape: [batch_size, seq_len, embed_dim]

        # Reshape q, k, v to separate the heads and then transpose to get dimensions:
        q = q.view(intermidiate_shape).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(intermidiate_shape).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(intermidiate_shape).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Calculate attention scores: [batch_size, num_heads, seq_len, seq_len]
        weight = q @ k.transpose(-2, -1)  # Matrix multiply q with k^T

        if causal_mask:
            # Apply causal mask to prevent attending to future tokens
            mask = torch.ones_like(weight, dtype=torch.bool).triu(
                1
            )  # Upper triangular mask
            weight.masked_fill_(mask, -torch.inf)  # Fill masked positions with -inf

        # Scale attention scores by sqrt(head_dim)
        weight /= math.sqrt(self.head_dim)

        # Apply softmax to get attention weights: [batch_size, num_heads, seq_len, seq_len]
        weight = F.softmax(weight, dim=-1)

        # Apply attention weights to values: [batch_size, num_heads, seq_len, head_dim]
        output = weight @ v

        # Reshape back to original dimensions: [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(input_shape)

        # Final projection: [batch_size, seq_len, embed_dim]
        output = self.out_proj(output)

        return output
