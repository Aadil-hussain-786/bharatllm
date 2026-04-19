"""
Bharat-3B Smart-Core: RMT Memory Tests
=========================================
Tests for the Recurrent Memory Transformer system.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_memory_read_shapes():
    """Test memory read module output shapes."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.rmt_memory import MemoryReadModule

        reader = MemoryReadModule(
            hidden_size=128,
            num_memory_tokens=16,
            num_heads=4,
            use_cross_attention=True,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        hidden = jax.random.normal(rng, (2, 32, 128))

        # Test with no memory (first segment)
        variables = reader.init(rng, hidden, memory=None)
        output = reader.apply(variables, hidden, memory=None)
        assert output.shape == hidden.shape
        print(f"✅ Memory read (no prior memory): {output.shape}")

        # Test with memory from previous segment
        memory = jax.random.normal(rng, (2, 16, 128))
        output = reader.apply(variables, hidden, memory=memory)
        assert output.shape == hidden.shape
        print(f"✅ Memory read (with memory): {output.shape}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


def test_memory_write_shapes():
    """Test memory write module output shapes."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.rmt_memory import MemoryWriteModule

        writer = MemoryWriteModule(
            hidden_size=128,
            num_memory_tokens=16,
            update_type="gated",
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        hidden = jax.random.normal(rng, (2, 32, 128))

        # Test writing without old memory
        variables = writer.init(rng, hidden, old_memory=None)
        memory = writer.apply(variables, hidden, old_memory=None)
        assert memory.shape == (2, 16, 128)
        print(f"✅ Memory write (new): {memory.shape}")

        # Test writing with old memory (gated update)
        old_memory = jax.random.normal(rng, (2, 16, 128))
        memory = writer.apply(variables, hidden, old_memory=old_memory)
        assert memory.shape == (2, 16, 128)
        print(f"✅ Memory write (gated update): {memory.shape}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


def test_rmt_segmentation():
    """Test RMT input segmentation."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.rmt_memory import RMTMemory

        rmt = RMTMemory(
            hidden_size=128,
            num_memory_tokens=8,
            num_segments=4,
            segment_length=16,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        # 48 tokens -> should split into 3 segments of 16
        hidden = jax.random.normal(rng, (2, 48, 128))

        segments = rmt.segment_input(hidden)
        assert len(segments) == 3  # 48 / 16 = 3
        assert segments[0].shape == (2, 16, 128)
        print(f"✅ Segmentation: {len(segments)} segments of {segments[0].shape}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


def test_rmt_full_forward():
    """Test full RMT forward pass with memory propagation."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.rmt_memory import RMTMemory

        rmt = RMTMemory(
            hidden_size=128,
            num_memory_tokens=8,
            num_segments=4,
            segment_length=16,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(0)
        hidden = jax.random.normal(rng, (2, 32, 128))

        # Simple identity processing function
        def process_fn(x):
            return x * 0.99  # Slight contraction

        variables = rmt.init(rng, hidden, process_fn)
        output, memory = rmt.apply(variables, hidden, process_fn)

        assert output.shape == hidden.shape
        assert memory.shape == (2, 8, 128)
        print(f"✅ Full RMT forward: output={output.shape}, memory={memory.shape}")
    except ImportError:
        print("⚠️ JAX/Flax not available, skipping test")


if __name__ == "__main__":
    print("=" * 50)
    print("RMT Memory Tests")
    print("=" * 50)
    test_memory_read_shapes()
    test_memory_write_shapes()
    test_rmt_segmentation()
    test_rmt_full_forward()
    print("\n✨ All RMT tests completed!")
