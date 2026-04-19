"""
Bharat-3B Smart-Core: Mixture of Softmaxes (MoS) Head
======================================================
The output head that replaces the standard single linear + softmax
with multiple sub-expert softmax heads combined via a gating network.

Key Insight:
    Standard softmax output is a "low-rank bottleneck" — the log-probability
    matrix has rank ≤ d_model, limiting expressiveness.
    
    MoS breaks this bottleneck by combining K softmax experts,
    giving the model rank ≤ K × d_model output expressiveness.

Architecture:
    For K=10 experts:
    1. Split hidden_states into 10 sub-representations
    2. Each expert produces its own softmax distribution
    3. A gating network computes weights for each expert
    4. Final output = weighted sum of expert distributions

Benefits:
    - Higher expressiveness than standard softmax
    - Better perplexity (10-15% improvement in prior work)
    - Captures multi-modal output distributions
    - Critical for handling Hindi/English code-switching

References:
    - Yang et al., "Breaking the Softmax Bottleneck" (ICLR 2018)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class MixtureOfSoftmaxes(nn.Module):
    """
    Mixture of Softmaxes output head with K sub-experts.
    
    P(y|x) = Σ_k π_k(x) · softmax(W_k · x + b_k)
    
    where π_k(x) are gating weights and W_k are expert-specific
    output projections.
    
    Attributes:
        vocab_size: Output vocabulary size.
        hidden_size: Input hidden dimension.
        num_experts: Number of softmax sub-experts.
        gating_type: How to compute gating weights.
        temperature: Softmax temperature for output distributions.
        dropout_rate: Dropout rate for expert outputs.
        dtype: Computation dtype.
    """
    vocab_size: int = 50_257
    hidden_size: int = 2560
    num_experts: int = 10
    gating_type: str = "learned"
    temperature: float = 1.0
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        """Initialize expert projections and gating network."""
        dense_init = nn.initializers.normal(stddev=0.02)

        # Expert projections: each expert has its own output projection
        # Instead of K separate dense layers, we use a single larger projection
        # and reshape, which is more TPU-friendly
        self.expert_projections = nn.Dense(
            self.num_experts * self.vocab_size,
            use_bias=False,
            kernel_init=dense_init,
            dtype=self.dtype,
            name="expert_projections",
        )

        # Gating network: produces mixing weights for experts
        if self.gating_type == "learned":
            self.gate = nn.Sequential([
                nn.Dense(
                    self.hidden_size // 2,
                    dtype=self.dtype,
                    name="gate_down",
                ),
                # SiLU activation for gating
                lambda x: jax.nn.silu(x),
                nn.Dense(
                    self.num_experts,
                    dtype=self.dtype,
                    name="gate_up",
                ),
            ])
        elif self.gating_type == "topk":
            self.gate = nn.Dense(
                self.num_experts,
                dtype=self.dtype,
                name="gate_topk",
            )

        # Pre-output normalization
        self.pre_output_norm = nn.RMSNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            name="pre_output_norm",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute MoS output logits.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            deterministic: Whether in eval mode (no dropout)
        
        Returns:
            logits: (batch, seq_len, vocab_size) - mixed expert logits
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Normalize before output projection
        hidden_states = self.pre_output_norm(hidden_states)

        # Compute gating weights: (batch, seq_len, num_experts)
        gate_logits = self.gate(hidden_states)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)

        # Compute expert outputs efficiently
        # Option 1: Single large projection (TPU-optimized)
        # Shape: (batch, seq_len, num_experts * vocab_size)
        all_expert_logits = self.expert_projections(hidden_states)

        # Reshape to (batch, seq_len, num_experts, vocab_size)
        all_expert_logits = all_expert_logits.reshape(
            batch_size, seq_len, self.num_experts, self.vocab_size
        )

        # Apply temperature
        if self.temperature != 1.0:
            all_expert_logits = all_expert_logits / self.temperature

        # Compute per-expert probabilities
        expert_probs = jax.nn.softmax(
            all_expert_logits.astype(jnp.float32), axis=-1
        )

        # Apply dropout to expert outputs during training
        if not deterministic and self.dropout_rate > 0:
            expert_probs = nn.Dropout(rate=self.dropout_rate)(
                expert_probs, deterministic=deterministic
            )

        # Mix experts: weighted sum of probability distributions
        # gate_weights: (batch, seq_len, num_experts, 1)
        # expert_probs: (batch, seq_len, num_experts, vocab_size)
        gate_weights_expanded = gate_weights[:, :, :, jnp.newaxis]
        mixed_probs = jnp.sum(
            gate_weights_expanded * expert_probs, axis=2
        )  # (batch, seq_len, vocab_size)

        # Convert back to logits (log probabilities) for loss computation
        # Add small epsilon to prevent log(0)
        logits = jnp.log(mixed_probs + 1e-10).astype(self.dtype)

        return logits

    def get_expert_diversity_loss(
        self,
        hidden_states: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute auxiliary loss to encourage expert diversity.
        
        Prevents mode collapse where only a few experts are used.
        Uses the coefficient of variation of gate weights as penalty.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            Diversity loss scalar.
        """
        gate_logits = self.gate(hidden_states)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)

        # Average gate weights across batch and sequence
        avg_gate = jnp.mean(gate_weights, axis=(0, 1))  # (num_experts,)

        # Coefficient of variation: std / mean
        # Lower is better (means experts are used equally)
        cv = jnp.std(avg_gate) / (jnp.mean(avg_gate) + 1e-10)

        return cv


class MoSWithSharedEmbedding(nn.Module):
    """
    MoS variant that shares the embedding matrix for output projection.
    
    Instead of a separate large projection matrix, each expert applies
    a small transform to the hidden state, then projects through the
    shared embedding matrix. This saves memory significantly.
    
    Memory Savings:
        Full MoS:   10 × 2560 × 50,257 = ~1.29B params
        Shared MoS: 10 × 2560 × 2560 + 2560 × 50,257 = ~194M params
        Savings:     ~85% reduction in output head parameters
    
    Attributes:
        vocab_size: Output vocabulary size.
        hidden_size: Input hidden dimension.
        num_experts: Number of softmax sub-experts.
        dtype: Computation dtype.
    """
    vocab_size: int = 50_257
    hidden_size: int = 2560
    num_experts: int = 10
    temperature: float = 1.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        embedding_matrix: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Compute MoS logits using shared embedding matrix.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            embedding_matrix: (vocab_size, hidden_size) - shared with input
            deterministic: Whether in eval mode
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Normalize
        hidden_states = nn.RMSNorm(
            epsilon=1e-5, dtype=self.dtype, name="mos_norm"
        )(hidden_states)

        # Expert-specific transforms (small dense layers)
        expert_transforms = []
        for i in range(self.num_experts):
            transformed = nn.Dense(
                self.hidden_size,
                use_bias=False,
                dtype=self.dtype,
                name=f"expert_transform_{i}",
            )(hidden_states)
            expert_transforms.append(transformed)

        # Stack: (batch, seq_len, num_experts, hidden_size)
        expert_hidden = jnp.stack(expert_transforms, axis=2)

        # Project through shared embedding: (batch, seq_len, num_experts, vocab_size)
        expert_logits = jnp.matmul(
            expert_hidden, embedding_matrix.T.astype(self.dtype)
        )

        if self.temperature != 1.0:
            expert_logits = expert_logits / self.temperature

        # Expert probabilities
        expert_probs = jax.nn.softmax(
            expert_logits.astype(jnp.float32), axis=-1
        )

        # Gating weights
        gate_logits = nn.Dense(
            self.num_experts, dtype=self.dtype, name="mos_gate"
        )(hidden_states)
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)

        # Mix: (batch, seq_len, vocab_size)
        mixed_probs = jnp.sum(
            gate_weights[:, :, :, jnp.newaxis] * expert_probs,
            axis=2,
        )

        logits = jnp.log(mixed_probs + 1e-10).astype(self.dtype)
        return logits
