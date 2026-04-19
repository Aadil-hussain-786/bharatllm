"""
Bharat-3B Smart-Core: Multi-Head Attention with GQA
====================================================
Grouped Query Attention (GQA) implementation with RoPE.
Optimized for TPU execution with JAX/Flax.

Key Features:
- GQA: 32 query heads, 8 KV heads (4:1 ratio) for memory efficiency
- RoPE: Rotary Position Embeddings for 128k context
- Flash Attention compatible (via JAX's dot_product_attention)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple

from bharat_3b_smart_core.src.model.embeddings import (
    apply_rotary_embedding,
    RMSNorm,
)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings.
    
    Uses fewer KV heads than query heads to reduce KV cache memory
    while maintaining model quality. Each KV head is shared across
    multiple query heads.
    
    DEQ Note: This layer is called repeatedly during fixed-point iteration.
    Weight sharing means the same attention weights process increasingly
    refined representations, achieving depth > 100 effective layers.
    
    Attributes:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads (GQA).
        head_dim: Dimension per attention head.
        attention_dropout: Dropout rate for attention weights.
        use_bias: Whether to use bias in projections.
        dtype: Computation dtype.
    """
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 80
    attention_dropout: float = 0.0
    use_bias: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        """Initialize projection layers."""
        dense_init = nn.initializers.normal(stddev=0.02)

        # Query projection: hidden_size -> num_heads * head_dim
        self.q_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="q_proj",
        )
        # Key projection: hidden_size -> num_kv_heads * head_dim
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="k_proj",
        )
        # Value projection: hidden_size -> num_kv_heads * head_dim
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="v_proj",
        )
        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = nn.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="o_proj",
        )

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Repeat KV heads to match query head count for GQA.
        
        Args:
            x: (batch, seq_len, num_kv_heads, head_dim)
        Returns:
            (batch, seq_len, num_attention_heads, head_dim)
        """
        if self.num_key_value_groups == 1:
            return x
        batch, seq_len, num_kv_heads, head_dim = x.shape
        x = x[:, :, :, jnp.newaxis, :]  # (B, S, KV, 1, D)
        x = jnp.broadcast_to(
            x, (batch, seq_len, num_kv_heads, self.num_key_value_groups, head_dim)
        )
        return x.reshape(batch, seq_len, self.num_attention_heads, head_dim)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_freqs: jnp.ndarray,
        sin_freqs: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        kv_cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Forward pass for grouped query attention.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            cos_freqs: RoPE cosine frequencies.
            sin_freqs: RoPE sine frequencies.
            attention_mask: Optional causal/padding mask.
            position_ids: Optional position indices for RoPE.
            deterministic: Whether in eval mode (no dropout).
            kv_cache: Optional cached key/value for autoregressive decoding.
        
        Returns:
            Tuple of (output, updated_kv_cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE to Q and K
        query = apply_rotary_embedding(query, cos_freqs, sin_freqs, position_ids)
        key = apply_rotary_embedding(key, cos_freqs, sin_freqs, position_ids)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_key, cached_value = kv_cache
            key = jnp.concatenate([cached_key, key], axis=1)
            value = jnp.concatenate([cached_value, value], axis=1)
        new_kv_cache = (key, value)

        # Repeat KV heads for GQA
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(self.dtype)
        attn_weights = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask (lower triangular)
        kv_len = key.shape[2]
        causal_mask = jnp.triu(
            jnp.full((seq_len, kv_len), -1e9, dtype=self.dtype),
            k=kv_len - seq_len + 1,
        )
        attn_weights = attn_weights + causal_mask[jnp.newaxis, jnp.newaxis, :, :]

        # Softmax
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn_weights = attn_weights.astype(self.dtype)

        # Dropout (only during training)
        if not deterministic and self.attention_dropout > 0:
            attn_weights = nn.Dropout(rate=self.attention_dropout)(
                attn_weights, deterministic=deterministic
            )

        # Weighted sum of values
        attn_output = jnp.matmul(attn_weights, value)

        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_kv_cache


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (SwiGLU variant).
    
    Uses SiLU activation with a gating mechanism for better expressiveness.
    This is the same FFN architecture used in Llama and Mistral.
    
    FFN(x) = (SiLU(xW_gate) ⊙ xW_up) @ W_down
    
    Attributes:
        hidden_size: Input/output dimension.
        intermediate_size: Inner FFN dimension.
        use_bias: Whether to use bias.
        dtype: Computation dtype.
    """
    hidden_size: int = 2560
    intermediate_size: int = 6912
    use_bias: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, hidden_size).
        Returns:
            Output of shape (batch, seq_len, hidden_size).
        """
        dense_init = nn.initializers.normal(stddev=0.02)

        gate = nn.Dense(
            self.intermediate_size,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="gate_proj",
        )(x)
        up = nn.Dense(
            self.intermediate_size,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="up_proj",
        )(x)

        # SwiGLU activation
        hidden = jax.nn.silu(gate) * up

        output = nn.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="down_proj",
        )(hidden)

        return output
