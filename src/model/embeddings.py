"""
Bharat-3B Smart-Core: Embedding Layers
=======================================
Token embeddings with RoPE (Rotary Position Embeddings) for extended context.
Designed for 128k+ token sequences with RMT memory integration.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
import numpy as np


def create_sinusoidal_positions(num_positions: int, dim: int) -> jnp.ndarray:
    """Create sinusoidal position embeddings (fallback for non-RoPE)."""
    position = jnp.arange(num_positions)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
    sincos = jnp.zeros((num_positions, dim))
    sincos = sincos.at[:, 0::2].set(jnp.sin(position * div_term))
    sincos = sincos.at[:, 1::2].set(jnp.cos(position * div_term))
    return sincos


def precompute_rope_frequencies(
    dim: int,
    max_position: int,
    theta: float = 500_000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute RoPE (Rotary Position Embedding) sin/cos frequencies.
    
    Uses extended theta (500k) for long-context support up to 128k tokens.
    This is critical for RMT segments that process long documents.
    
    Args:
        dim: Head dimension (must be even).
        max_position: Maximum sequence position.
        theta: RoPE base frequency (higher = longer context).
    
    Returns:
        Tuple of (cos_freqs, sin_freqs), each of shape (max_position, dim).
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(max_position, dtype=jnp.float32)
    # Outer product: (max_position, dim // 2)
    freqs_matrix = jnp.outer(positions, freqs)
    # Duplicate for full dim: (max_position, dim)
    cos_freqs = jnp.cos(jnp.concatenate([freqs_matrix, freqs_matrix], axis=-1))
    sin_freqs = jnp.sin(jnp.concatenate([freqs_matrix, freqs_matrix], axis=-1))
    return cos_freqs, sin_freqs


def apply_rotary_embedding(
    x: jnp.ndarray,
    cos_freqs: jnp.ndarray,
    sin_freqs: jnp.ndarray,
    position_ids: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Apply RoPE to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim).
        cos_freqs: Cosine frequencies of shape (max_pos, head_dim).
        sin_freqs: Sine frequencies of shape (max_pos, head_dim).
        position_ids: Optional position indices of shape (batch, seq_len).
    
    Returns:
        Rotated tensor of same shape as x.
    """
    seq_len = x.shape[1]

    if position_ids is not None:
        cos = cos_freqs[position_ids]  # (batch, seq_len, head_dim)
        sin = sin_freqs[position_ids]
    else:
        cos = cos_freqs[:seq_len]  # (seq_len, head_dim)
        sin = sin_freqs[:seq_len]

    # Expand dims for broadcasting with heads
    if cos.ndim == 2:
        cos = cos[jnp.newaxis, :, jnp.newaxis, :]  # (1, seq_len, 1, head_dim)
        sin = sin[jnp.newaxis, :, jnp.newaxis, :]
    elif cos.ndim == 3:
        cos = cos[:, :, jnp.newaxis, :]  # (batch, seq_len, 1, head_dim)
        sin = sin[:, :, jnp.newaxis, :]

    # Rotate pairs
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)

    return x * cos + rotated * sin


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional learned position embeddings.
    
    For Bharat-3B, we primarily use RoPE (applied in attention),
    but this provides the token embedding + optional absolute position fallback.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Embedding dimension.
        max_position: Maximum sequence length.
        use_absolute_position: Whether to add absolute position embeddings.
        initializer_range: Std for weight initialization.
        dtype: Computation dtype.
    """
    vocab_size: int = 50_257
    hidden_size: int = 2560
    max_position: int = 128_000
    use_absolute_position: bool = False
    initializer_range: float = 0.02
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Embed tokens with optional position information.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            position_ids: Position IDs of shape (batch, seq_len).
        
        Returns:
            Embeddings of shape (batch, seq_len, hidden_size).
        """
        # Token embeddings
        token_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=self.initializer_range),
            dtype=self.dtype,
            name="token_embedding",
        )
        embeddings = token_embed(input_ids)

        # Optional absolute position embeddings (not used with RoPE)
        if self.use_absolute_position:
            if position_ids is None:
                seq_len = input_ids.shape[1]
                position_ids = jnp.arange(seq_len)[jnp.newaxis, :]

            position_embed = nn.Embed(
                num_embeddings=self.max_position,
                features=self.hidden_size,
                embedding_init=nn.initializers.normal(stddev=self.initializer_range),
                dtype=self.dtype,
                name="position_embedding",
            )
            embeddings = embeddings + position_embed(position_ids)

        return embeddings


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm and used in modern architectures
    (Llama, Mistral, etc.). Critical for DEQ stability.
    
    Attributes:
        dim: Feature dimension.
        epsilon: Small constant for numerical stability.
        dtype: Computation dtype.
    """
    dim: int
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply RMS normalization."""
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.dim,),
        )
        # Compute in float32 for stability
        x_float = x.astype(jnp.float32)
        variance = jnp.mean(x_float ** 2, axis=-1, keepdims=True)
        x_normed = x_float * jax.lax.rsqrt(variance + self.epsilon)
        return (x_normed * weight).astype(self.dtype)
