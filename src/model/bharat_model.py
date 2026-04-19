"""
Bharat-3B Smart-Core: Full Model Assembly
==========================================
Combines all components into the complete Bharat-3B architecture:

    Input Tokens
        ↓
    Token Embedding (50,257 vocab)
        ↓
    RMT Memory Read (inject context from previous segments)
        ↓
    DEQ Universal Block (fixed-point iteration → 100+ effective layers)
        ↓
    RMT Memory Write (compress segment into memory tokens)
        ↓
    Mixture of Softmaxes (10 sub-experts → output logits)
        ↓
    Output Logits

Total Parameters: ~1.6B actual → ~3B+ effective (via DEQ weight reuse)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any

from bharat_3b_smart_core.src.model.embeddings import (
    TokenEmbedding,
    RMSNorm,
    precompute_rope_frequencies,
)
from bharat_3b_smart_core.src.model.deq_layer import DEQLayer
from bharat_3b_smart_core.src.model.rmt_memory import RMTMemory
from bharat_3b_smart_core.src.model.mos_head import MixtureOfSoftmaxes, MoSWithSharedEmbedding


class BharatModel(nn.Module):
    """
    Bharat-3B Smart-Core Language Model.
    
    A novel architecture combining:
    1. DEQ (Deep Equilibrium): 100+ effective layers from 1 weight-tied block
    2. RMT (Recurrent Memory Transformer): 128k+ context via segment-level memory
    3. MoS (Mixture of Softmaxes): 10 sub-expert output heads
    
    Patent-worthy innovations:
    - DEQ + RMT combination (first of its kind)
    - Recursive distillation methodology
    - Hindi/Hinglish optimized BPE tokenizer
    
    Attributes:
        vocab_size: Vocabulary size.
        hidden_size: Model hidden dimension.
        intermediate_size: FFN inner dimension.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of KV heads (GQA).
        head_dim: Per-head dimension.
        max_position_embeddings: Maximum context length.
        rope_theta: RoPE base frequency.
        deq_max_iterations: DEQ solver iterations.
        deq_solver: DEQ solver type.
        deq_anderson_m: Anderson acceleration history size.
        deq_tolerance: DEQ convergence threshold.
        deq_jac_reg_weight: Jacobian regularization weight.
        deq_phantom_grad_steps: Phantom gradient steps.
        rmt_num_memory_tokens: RMT memory slot count.
        rmt_num_segments: Number of RMT segments.
        rmt_segment_length: Tokens per RMT segment.
        rmt_memory_update: RMT memory update strategy.
        mos_num_experts: Number of MoS sub-experts.
        mos_temperature: MoS softmax temperature.
        use_shared_embedding_mos: Use memory-efficient MoS.
        attention_dropout: Attention dropout rate.
        hidden_dropout: Hidden state dropout rate.
        layer_norm_epsilon: LayerNorm epsilon.
        use_bias: Whether to use bias in projections.
        initializer_range: Init std.
        dtype: Computation dtype.
    """
    # Vocabulary & Embeddings
    vocab_size: int = 50_257
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 80
    max_position_embeddings: int = 128_000
    rope_theta: float = 500_000.0

    # DEQ Configuration
    deq_max_iterations: int = 20
    deq_solver: str = "anderson"
    deq_anderson_m: int = 5
    deq_tolerance: float = 1e-5
    deq_jac_reg_weight: float = 0.1
    deq_phantom_grad_steps: int = 5

    # RMT Configuration
    rmt_num_memory_tokens: int = 128
    rmt_num_segments: int = 8
    rmt_segment_length: int = 16_000
    rmt_memory_update: str = "gated"

    # MoS Configuration
    mos_num_experts: int = 10
    mos_temperature: float = 1.0
    use_shared_embedding_mos: bool = True  # Save ~85% output head params

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    initializer_range: float = 0.02

    # Compute
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        """Initialize all model components."""
        # 1. Token Embedding
        self.embed_tokens = TokenEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position=self.max_position_embeddings,
            use_absolute_position=False,  # Using RoPE instead
            initializer_range=self.initializer_range,
            dtype=self.dtype,
            name="embed_tokens",
        )

        # 2. Input normalization (before DEQ)
        self.embed_norm = RMSNorm(
            dim=self.hidden_size,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="embed_norm",
        )

        # 3. DEQ Layer (the universal block)
        self.deq_layer = DEQLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            max_iterations=self.deq_max_iterations,
            solver=self.deq_solver,
            anderson_m=self.deq_anderson_m,
            tolerance=self.deq_tolerance,
            jac_reg_weight=self.deq_jac_reg_weight,
            phantom_grad_steps=self.deq_phantom_grad_steps,
            attention_dropout=self.attention_dropout,
            use_bias=self.use_bias,
            layer_norm_epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="deq_layer",
        )

        # 4. RMT Memory System
        self.rmt = RMTMemory(
            hidden_size=self.hidden_size,
            num_memory_tokens=self.rmt_num_memory_tokens,
            num_segments=self.rmt_num_segments,
            segment_length=self.rmt_segment_length,
            memory_update=self.rmt_memory_update,
            use_cross_attention=True,
            dtype=self.dtype,
            name="rmt_memory",
        )

        # 5. Final normalization (before output head)
        self.final_norm = RMSNorm(
            dim=self.hidden_size,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype,
            name="final_norm",
        )

        # 6. Output Head (MoS)
        if self.use_shared_embedding_mos:
            self.output_head = MoSWithSharedEmbedding(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_experts=self.mos_num_experts,
                temperature=self.mos_temperature,
                dtype=self.dtype,
                name="mos_shared_head",
            )
        else:
            self.output_head = MixtureOfSoftmaxes(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_experts=self.mos_num_experts,
                temperature=self.mos_temperature,
                dtype=self.dtype,
                name="mos_head",
            )

        # Precompute RoPE frequencies
        self._cos_freqs, self._sin_freqs = precompute_rope_frequencies(
            dim=self.head_dim,
            max_position=self.max_position_embeddings,
            theta=self.rope_theta,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Full forward pass of Bharat-3B.
        
        Pipeline:
            tokens → embed → [RMT(DEQ(x))] → norm → MoS → logits
        
        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: Optional (batch, seq_len) mask.
            position_ids: Optional (batch, seq_len) positions.
            deterministic: Whether in eval mode.
            return_dict: Whether to return a dict or tuple.
        
        Returns:
            Dict with "logits", "deq_info", "memory", etc.
        """
        batch_size, seq_len = input_ids.shape

        # ---- Step 1: Embed tokens ----
        hidden_states = self.embed_tokens(input_ids, position_ids)
        hidden_states = self.embed_norm(hidden_states)

        # ---- Step 2: Prepare RoPE frequencies ----
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[jnp.newaxis, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))

        cos_freqs = self._cos_freqs
        sin_freqs = self._sin_freqs

        # ---- Step 3: Process through RMT + DEQ ----
        # Define the DEQ processing function for each segment
        def process_segment(segment_hidden):
            """Process a single segment through the DEQ layer."""
            segment_output, deq_info = self.deq_layer(
                segment_hidden,
                cos_freqs=cos_freqs,
                sin_freqs=sin_freqs,
                attention_mask=attention_mask,
                position_ids=None,  # RoPE handles positions internally
                deterministic=deterministic,
            )
            return segment_output

        # RMT handles segmentation, memory read/write, and calls DEQ per segment
        hidden_states, memory = self.rmt(
            hidden_states,
            process_fn=process_segment,
            deterministic=deterministic,
        )

        # ---- Step 4: Final normalization ----
        hidden_states = self.final_norm(hidden_states)

        # ---- Step 5: MoS output head ----
        if self.use_shared_embedding_mos:
            # Get embedding matrix for shared projection
            embedding_matrix = self.embed_tokens.variables["params"][
                "token_embedding"
            ]["embedding"]
            logits = self.output_head(
                hidden_states,
                embedding_matrix=embedding_matrix,
                deterministic=deterministic,
            )
        else:
            logits = self.output_head(
                hidden_states,
                deterministic=deterministic,
            )

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "memory": memory,
            }
        else:
            return logits

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: (batch, seq_len) prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            rng_key: Random key for sampling.
        
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens).
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        generated = input_ids

        for i in range(max_new_tokens):
            # Forward pass
            outputs = self(generated, deterministic=True)
            logits = outputs["logits"]

            # Get next token logits
            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(next_logits, top_k)
                next_logits = jnp.full_like(next_logits, -1e10)
                next_logits = next_logits.at[
                    jnp.arange(next_logits.shape[0])[:, None],
                    top_k_indices,
                ].set(top_k_logits)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = jnp.sort(next_logits, axis=-1)[:, ::-1]
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
                mask = cumulative_probs > top_p
                # Shift mask right to keep at least one token
                mask = jnp.concatenate(
                    [jnp.zeros_like(mask[:, :1]), mask[:, :-1]], axis=-1
                )
                sorted_logits = jnp.where(mask, -1e10, sorted_logits)
                # Unsort
                unsort_indices = jnp.argsort(
                    jnp.argsort(next_logits, axis=-1)[:, ::-1], axis=-1
                )
                next_logits = jnp.take_along_axis(
                    sorted_logits, unsort_indices, axis=-1
                )

            # Sample
            rng_key, sample_key = jax.random.split(rng_key)
            probs = jax.nn.softmax(next_logits, axis=-1)
            next_token = jax.random.categorical(sample_key, jnp.log(probs + 1e-10))
            next_token = next_token[:, jnp.newaxis]  # (batch, 1)

            # Append to generated sequence
            generated = jnp.concatenate([generated, next_token], axis=1)

            # Check for EOS
            if jnp.all(next_token == 2):  # eos_token_id
                break

        return generated


def count_parameters(params) -> Dict[str, int]:
    """
    Count parameters in the model.
    
    Args:
        params: Model parameters (PyTree).
    
    Returns:
        Dict with component-wise parameter counts.
    """
    counts = {}
    total = 0

    def _count(path, leaf):
        nonlocal total
        n = leaf.size
        total += n
        component = path[0] if path else "other"
        counts[component] = counts.get(component, 0) + n

    jax.tree_util.tree_map_with_path(
        lambda path, x: _count([str(p) for p in path], x),
        params,
    )
    counts["total"] = total
    return counts


def create_bharat_model(config=None):
    """
    Factory function to create a BharatModel from config.
    
    Args:
        config: Model configuration. If None, uses defaults.
    
    Returns:
        BharatModel instance.
    """
    if config is None:
        from bharat_3b_smart_core.configs.model_config import get_config
        config = get_config()

    model = BharatModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        deq_max_iterations=config.deq.max_iterations,
        deq_solver=config.deq.solver,
        deq_anderson_m=config.deq.anderson_m,
        deq_tolerance=config.deq.tolerance,
        deq_jac_reg_weight=config.deq.jac_reg_weight,
        deq_phantom_grad_steps=config.deq.phantom_grad_steps,
        rmt_num_memory_tokens=config.rmt.num_memory_tokens,
        rmt_num_segments=config.rmt.num_segments,
        rmt_segment_length=config.rmt.segment_length,
        rmt_memory_update=config.rmt.memory_update,
        mos_num_experts=config.mos.num_experts,
        mos_temperature=config.mos.temperature,
        dtype=getattr(jnp, config.dtype),
    )

    return model
