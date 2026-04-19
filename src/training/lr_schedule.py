"""
Bharat-3B Smart-Core: Learning Rate Scheduler
===============================================
Implements the critical warm-up + cosine decay schedule.

The 2,000-step warm-up is essential for DEQ stability:
    - DEQ's fixed-point iteration is sensitive to early gradients
    - A slow warm-up prevents initial loss spikes
    - If a spike occurs, we restart from the last good checkpoint

Schedule:
    Steps 0-2000:       Linear warm-up (0 → peak_lr)
    Steps 2000-end:     Cosine decay (peak_lr → min_lr)
"""

import jax.numpy as jnp
import optax
from typing import Optional


def create_cosine_schedule(
    peak_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 2000,
    total_steps: int = 1_000_000,
) -> optax.Schedule:
    """
    Create warm-up + cosine decay learning rate schedule.
    
    Args:
        peak_lr: Maximum learning rate after warm-up.
        min_lr: Minimum learning rate at end of cosine decay.
        warmup_steps: Number of linear warm-up steps.
        total_steps: Total training steps.
    
    Returns:
        Optax schedule function.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr / peak_lr,  # Cosine decay minimum ratio
    )

    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )

    return schedule


def create_linear_schedule(
    peak_lr: float = 3e-4,
    min_lr: float = 0.0,
    warmup_steps: int = 2000,
    total_steps: int = 1_000_000,
) -> optax.Schedule:
    """Create warm-up + linear decay schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )

    linear_fn = optax.linear_schedule(
        init_value=peak_lr,
        end_value=min_lr,
        transition_steps=total_steps - warmup_steps,
    )

    return optax.join_schedules(
        schedules=[warmup_fn, linear_fn],
        boundaries=[warmup_steps],
    )


def create_optimizer(
    schedule: optax.Schedule,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create the full optimizer chain for Bharat-3B training.
    
    Chain:
        1. Gradient clipping (max_grad_norm)
        2. AdamW with weight decay
        3. Learning rate schedule
    
    Args:
        schedule: Learning rate schedule.
        weight_decay: Weight decay coefficient.
        beta1: Adam beta1.
        beta2: Adam beta2.
        epsilon: Adam epsilon.
        max_grad_norm: Maximum gradient norm for clipping.
    
    Returns:
        Optax optimizer.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=beta1,
            b2=beta2,
            eps=epsilon,
            weight_decay=weight_decay,
        ),
    )

    return optimizer


def get_lr_at_step(schedule: optax.Schedule, step: int) -> float:
    """Get learning rate at a specific step (for logging)."""
    return float(schedule(step))
