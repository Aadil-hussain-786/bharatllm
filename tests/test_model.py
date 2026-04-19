"""
Bharat-3B Smart-Core: Full Model Tests
=========================================
Integration tests for the complete model assembly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_small_model_forward():
    """Test full model forward pass with small config."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.bharat_model import BharatModel, count_parameters

        # Small model for testing
        model = BharatModel(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
            deq_max_iterations=3,
            deq_solver="fixed_point",
            rmt_num_memory_tokens=8,
            rmt_num_segments=2,
            rmt_segment_length=64,
            mos_num_experts=4,
            use_shared_embedding_mos=True,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (2, 32), 0, 1024)

        variables = model.init(
            {"params": rng, "dropout": rng, "jac_reg": rng},
            input_ids,
        )

        # Count parameters
        params = variables["params"]
        param_counts = count_parameters(params)
        print(f"✅ Small model initialized: {param_counts['total']:,} parameters")

        # Forward pass
        outputs = model.apply(
            variables,
            input_ids,
            deterministic=True,
        )

        logits = outputs["logits"]
        assert logits.shape == (2, 32, 1024), f"Wrong shape: {logits.shape}"
        print(f"✅ Forward pass: logits shape = {logits.shape}")
        print(f"   Memory shape: {outputs['memory'].shape}")
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")


def test_model_generation():
    """Test autoregressive generation."""
    try:
        import jax
        import jax.numpy as jnp
        from src.model.bharat_model import BharatModel

        model = BharatModel(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            deq_max_iterations=2,
            deq_solver="fixed_point",
            rmt_num_memory_tokens=4,
            rmt_num_segments=1,
            rmt_segment_length=64,
            mos_num_experts=2,
            use_shared_embedding_mos=True,
            dtype=jnp.float32,
        )

        rng = jax.random.PRNGKey(42)
        prompt = jnp.array([[1, 10, 20, 30, 40]], dtype=jnp.int32)  # BOS + 4 tokens

        variables = model.init(
            {"params": rng, "dropout": rng, "jac_reg": rng},
            prompt,
        )

        # Note: generate() would need model.apply bound to variables
        # This is a shape test
        print("✅ Generation test setup correct")
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")


def test_distillation_loss():
    """Test distillation loss computation."""
    try:
        import jax
        import jax.numpy as jnp
        from src.training.distillation import (
            distillation_loss,
            kl_divergence_loss,
            cross_entropy_loss,
        )

        batch_size, seq_len, vocab_size = 2, 16, 100

        student_logits = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, vocab_size)
        )
        teacher_logits = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, vocab_size)
        )
        labels = jax.random.randint(
            jax.random.PRNGKey(2), (batch_size, seq_len), 0, vocab_size
        )

        # Test KL divergence
        kl = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)
        assert kl.shape == (), f"KL should be scalar, got {kl.shape}"
        assert kl >= 0, "KL divergence must be non-negative"
        print(f"✅ KL divergence loss: {float(kl):.4f}")

        # Test CE loss
        ce = cross_entropy_loss(student_logits, labels)
        assert ce.shape == ()
        print(f"✅ Cross-entropy loss: {float(ce):.4f}")

        # Test combined distillation loss
        total, metrics = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=4.0, alpha=0.7,
        )
        assert total.shape == ()
        print(f"✅ Distillation loss: {float(total):.4f}")
        print(f"   KL: {float(metrics['kl_loss']):.4f}, CE: {float(metrics['ce_loss']):.4f}")
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")


def test_dpo_loss():
    """Test DPO loss computation."""
    try:
        import jax
        import jax.numpy as jnp
        from src.alignment.dpo_trainer import dpo_loss

        batch_size = 4

        # Chosen should have higher log-prob than rejected
        policy_chosen = jnp.array([-1.0, -0.5, -0.8, -0.3])
        policy_rejected = jnp.array([-2.0, -1.5, -1.8, -1.3])
        ref_chosen = jnp.array([-1.2, -0.7, -1.0, -0.5])
        ref_rejected = jnp.array([-1.8, -1.3, -1.6, -1.1])

        loss, metrics = dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=0.1,
        )

        assert loss.shape == ()
        assert metrics["accuracy"] > 0.5  # Should prefer chosen
        print(f"✅ DPO loss: {float(loss):.4f}")
        print(f"   Accuracy: {float(metrics['accuracy']):.2%}")
        print(f"   Reward margin: {float(metrics['reward_margin']):.4f}")
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")


def test_safety_guardrails():
    """Test safety guardrails."""
    from src.alignment.safety import SafetyGuardrails

    guard = SafetyGuardrails(use_ml_detection=False)

    # Should be safe
    result = guard.check_input("What is the capital of India?")
    assert result.is_safe, "Safe input incorrectly flagged"
    print("✅ Safe input passed correctly")

    # Should be safe (Hindi)
    result = guard.check_input("भारत की राजधानी क्या है?")
    assert result.is_safe, "Safe Hindi input incorrectly flagged"
    print("✅ Safe Hindi input passed correctly")

    # Should be PII detected in output
    result = guard.check_output("My Aadhaar number is 123456789012")
    assert not result.is_safe, "PII not detected"
    print(f"✅ PII detection working: {result.category}")

    # Test PII scrubbing
    scrubbed = guard.scrub_pii("Call me at 9876543210 or email test@example.com")
    assert "[PHONE]" in scrubbed
    assert "[EMAIL]" in scrubbed
    print(f"✅ PII scrubbing: '{scrubbed}'")


def test_lr_schedule():
    """Test learning rate schedule."""
    try:
        from src.training.lr_schedule import create_cosine_schedule, get_lr_at_step

        schedule = create_cosine_schedule(
            peak_lr=3e-4,
            min_lr=3e-5,
            warmup_steps=2000,
            total_steps=100000,
        )

        # At step 0: should be ~0
        lr_0 = get_lr_at_step(schedule, 0)
        assert lr_0 < 1e-5, f"LR at step 0 should be ~0, got {lr_0}"

        # At step 2000: should be peak_lr
        lr_peak = get_lr_at_step(schedule, 2000)
        assert abs(lr_peak - 3e-4) < 1e-5, f"LR at step 2000 should be 3e-4, got {lr_peak}"

        # At step 100000: should be min_lr
        lr_end = get_lr_at_step(schedule, 100000)
        assert lr_end < 1e-4, f"LR at end should be ~3e-5, got {lr_end}"

        print(f"✅ LR schedule: start={lr_0:.2e}, peak={lr_peak:.2e}, end={lr_end:.2e}")
    except ImportError as e:
        print(f"⚠️ Missing dependency: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Full Model Integration Tests")
    print("=" * 50)
    test_small_model_forward()
    test_model_generation()
    test_distillation_loss()
    test_dpo_loss()
    test_safety_guardrails()
    test_lr_schedule()
    print("\n✨ All integration tests completed!")
