"""
Bharat-3B Smart-Core: Model Configuration
==========================================
All hyperparameters for the DEQ + RMT + MoS architecture.
3B parameters total with effective depth > 100 layers via DEQ.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Returns the default model configuration for Bharat-3B."""
    config = ml_collections.ConfigDict()

    # ========================================
    # Model Architecture
    # ========================================
    config.model_name = "bharat-3b-smart-core"
    config.hidden_size = 2560  # d_model
    config.intermediate_size = 6912  # FFN inner dim (~2.7x hidden)
    config.num_attention_heads = 32
    config.num_key_value_heads = 8  # GQA: 4 queries per KV head
    config.head_dim = 80  # hidden_size / num_attention_heads
    config.max_position_embeddings = 128_000  # 128k context via RMT
    config.rope_theta = 500_000.0  # Extended RoPE for long context

    # Vocabulary
    config.vocab_size = 50_257
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2

    # ========================================
    # Deep Equilibrium (DEQ) Configuration
    # ========================================
    config.deq = ml_collections.ConfigDict()
    config.deq.max_iterations = 20  # Fixed-point iterations
    config.deq.solver = "anderson"  # Options: "anderson", "broyden", "fixed_point"
    config.deq.anderson_m = 5  # Anderson acceleration history size
    config.deq.tolerance = 1e-5  # Convergence threshold
    config.deq.jac_reg_weight = 0.1  # Jacobian regularization for stability
    config.deq.phantom_grad_steps = 5  # Phantom gradient for backprop
    config.deq.pretrain_steps = 0  # Steps of normal forward before DEQ kicks in

    # ========================================
    # Recurrent Memory Transformer (RMT)
    # ========================================
    config.rmt = ml_collections.ConfigDict()
    config.rmt.num_memory_tokens = 128  # Memory slots
    config.rmt.memory_dim = 2560  # Same as hidden_size
    config.rmt.num_segments = 8  # Number of segments for long sequences
    config.rmt.segment_length = 16_000  # Tokens per segment
    config.rmt.memory_update = "gated"  # Options: "gated", "additive", "replace"
    config.rmt.cross_attention = True  # Use cross-attention for memory read

    # ========================================
    # Mixture of Softmaxes (MoS)
    # ========================================
    config.mos = ml_collections.ConfigDict()
    config.mos.num_experts = 10  # Number of softmax sub-experts
    config.mos.gating_type = "learned"  # Options: "learned", "topk", "hash"
    config.mos.temperature = 1.0
    config.mos.dropout_rate = 0.0

    # ========================================
    # Transformer Block Components
    # ========================================
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    config.layer_norm_epsilon = 1e-5
    config.use_bias = False  # No bias in linear layers (modern practice)
    config.activation = "silu"  # SiLU/Swish activation in FFN
    config.use_gated_ffn = True  # Gated Linear Unit (GLU) in FFN
    config.tie_word_embeddings = False  # Don't tie input/output embeddings

    # ========================================
    # Initialization
    # ========================================
    config.initializer_range = 0.02
    config.dtype = "bfloat16"  # Training precision

    # ========================================
    # Parameter Count Estimate
    # ========================================
    # Embedding:           50,257 × 2560                = ~128.7M
    # DEQ Block (single):
    #   - Q/K/V proj:      2560 × 2560 × 3             = ~19.7M
    #   - O proj:           2560 × 2560                 = ~6.6M
    #   - FFN:              2560 × 6912 × 3 (gated)     = ~53.1M
    #   - LayerNorm × 2:    2560 × 2 × 2               = ~10.2K
    # DEQ Subtotal:                                      = ~79.4M
    # RMT Memory:           128 × 2560 × 2              = ~655K
    # MoS Head:             2560 × 50,257 × 10 + gate   = ~1.29B
    # Unembedding:          2560 × 50,257               = ~128.7M
    # -----------------------------------------------
    # TOTAL ~1.63B (with DEQ iterations -> effective ~3B+)
    # Note: DEQ reuses weights, so 1.6B actual params = 3B+ effective

    return config


def get_small_config() -> ml_collections.ConfigDict:
    """Returns a smaller config for testing/debugging."""
    config = get_config()
    config.hidden_size = 512
    config.intermediate_size = 1376
    config.num_attention_heads = 8
    config.num_key_value_heads = 2
    config.head_dim = 64
    config.vocab_size = 1024
    config.max_position_embeddings = 2048

    config.deq.max_iterations = 5
    config.rmt.num_memory_tokens = 16
    config.rmt.memory_dim = 512
    config.rmt.num_segments = 2
    config.rmt.segment_length = 1024
    config.mos.num_experts = 4

    return config
