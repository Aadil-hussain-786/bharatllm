"""
Bharat-3B Smart-Core: DEQ Layer Tests
=======================================
Tests for the Deep Equilibrium Layer — the most critical component.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_anderson_acceleration_convergence():
    """Test that Anderson acceleration converges to a fixed point."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.deq_layer import anderson_acceleration

        # Simple contractive function: f(x) = 0.5x + 1
        # Fixed point: x* = 2
        def simple_func(x):
            return 0.5 * x + 1.0

        x0 = jnp.zeros(10)
        x_star, info = anderson_acceleration(
            f=simple_func,
            x0=x0,
            max_iterations=20,
            m=5,
            tolerance=1e-6,
        )

        # Should converge to 2.0
        assert jnp.allclose(x_star, 2.0, atol=1e-3), f"Expected 2.0, got {x_star}"
        print("✅ Anderson acceleration converges correctly")
    except ImportError:
        print("⚠️ JAX not available, skipping test")


def test_fixed_point_iteration():
    """Test simple fixed-point iteration."""
    try:
        import jax.numpy as jnp
        from src.model.deq_layer import fixed_point_iteration

        def simple_func(x):
            return 0.5 * x + 1.0

        x0 = jnp.zeros(10)
        x_star, info = fixed_point_iteration(
            f=simple_func,
            x0=x0,
            max_iterations=50,
            tolerance=1e-6,
        )

        assert jnp.allclose(x_star, 2.0, atol=1e-2), f"Expected 2.0, got {x_star}"
        print("✅ Fixed-point iteration converges correctly")
    except ImportError:
        print("⚠️ JAX not available, skipping test")


def test_deq_layer_shapes():
    """Test DEQ layer output shapes."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.deq_layer import DEQLayer
        from src.model.embeddings import precompute_rope_frequencies

        deq = DEQLayer(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            max_iterations=5,
            solver="fixed_point",
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        batch_size, seq_len = 2, 16
        hidden_states = jax.random.normal(
            rng, (batch_size, seq_len, 128)
        )

        cos_freqs, sin_freqs = precompute_rope_frequencies(32, 128)

        variables = deq.init(
            {"params": rng, "jac_reg": rng},
            hidden_states,
            cos_freqs,
            sin_freqs,
        )

        output, info = deq.apply(
            variables,
            hidden_states,
            cos_freqs,
            sin_freqs,
            deterministic=True,
        )

        assert output.shape == (batch_size, seq_len, 128), f"Wrong shape: {output.shape}"
        assert "num_iterations" in info
        print(f"✅ DEQ layer shapes correct: {output.shape}")
        print(f"   Iterations: {info['num_iterations']}, Residual: {info['final_residual']:.6f}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


def test_transformer_block():
    """Test the universal transformer block."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.deq_layer import TransformerBlock
        from src.model.embeddings import precompute_rope_frequencies

        block = TransformerBlock(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 16, 128))
        cos_freqs, sin_freqs = precompute_rope_frequencies(32, 128)

        variables = block.init(rng, x, cos_freqs, sin_freqs)
        output = block.apply(variables, x, cos_freqs, sin_freqs)

        assert output.shape == x.shape
        print(f"✅ Transformer block shapes correct: {output.shape}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


if __name__ == "__main__":
    print("=" * 50)
    print("DEQ Layer Tests")
    print("=" * 50)
    test_anderson_acceleration_convergence()
    test_fixed_point_iteration()
    test_transformer_block()
    test_deq_layer_shapes()
    print("\n✨ All DEQ tests completed!")
