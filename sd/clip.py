import torch
from torch import nn
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    CLIP embedding layer that combines token and position embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, n_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, embed_dim))

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Forward pass through the CLIP embedding layer."""
        # (Batch_size, Sequence_length) -> (Batch_size, Sequence_length, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    """
    A single layer of the CLIP model, consisting of self-attention and feed-forward networks.
    """

    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.self_attention = SelfAttention(n_heads, embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear_2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP layer.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_size, Sequence_length, Dim).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_size, Sequence_length, Dim).
        """
        # (Batch_size, Sequence_length, Dim) -> (Batch_size, Sequence_length, Dim)
        residue = x

        x = self.layernorm_1(x)
        x = self.self_attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation

        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    """
    CLIP model that processes input tokens through multiple layers of self-attention and feed-forward networks.
    This model is designed to handle text input and generate embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([CLIPLayer(12, 768) for _ in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP model.

        Args:
            tokens (torch.LongTensor): Input tokens of shape (Batch_size, Sequence_length).

        Returns:
            torch.Tensor: Output features of shape (Batch_size, Sequence_length, Dim).
        """
        tokens = tokens.to(torch.long)

        # (Batch_size, Sequence_length) -> (Batch_size, Sequence_length, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_size, Sequence_length, Dim)
        state = self.layernorm(state)

        return state
