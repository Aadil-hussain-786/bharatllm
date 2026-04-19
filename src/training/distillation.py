"""
Bharat-3B Smart-Core: Recursive Distillation
==============================================
The secret sauce: learning from Teacher models during pre-training.

Recursive Distillation Loop:
    For each training batch:
    1. Student (Bharat-3B) generates logits for the batch
    2. Teacher models generate "Soft Labels" (probability distributions)
    3. Loss = α * KL_div(student, teacher_soft) + (1-α) * CE(student, hard_labels)
    4. Student learns BOTH the correct answer AND the teacher's confidence

Why "Recursive"?
    - As the student improves, its own outputs become more teacher-like
    - Periodically, the student's improved outputs are used to re-generate
      soft labels for harder examples
    - This creates a virtuous cycle of improvement

Key Hyperparameters:
    - Temperature (τ=4.0): Higher = softer distributions = more knowledge transfer
    - Alpha (α=0.7): Weight towards distillation loss (vs hard label CE)
    - Consensus mode: Use average of teachers or majority vote

References:
    - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
    - Furlanello et al., "Born Again Neural Networks" (2018)
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def kl_divergence_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    temperature: float = 4.0,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute KL divergence loss between student and teacher distributions.
    
    The temperature parameter controls how "soft" the distributions are:
    - T=1: Standard softmax (hard)
    - T=4: Very soft (transfers more "dark knowledge")
    - T>10: Nearly uniform (too soft, less useful)
    
    Loss is scaled by T² to maintain gradient magnitude across temperatures.
    
    Args:
        student_logits: (batch, seq_len, vocab_size) from student model.
        teacher_logits: (batch, seq_len, vocab_size) from teacher model.
        temperature: Softmax temperature for distillation.
        mask: Optional (batch, seq_len) mask for padding tokens.
    
    Returns:
        Scalar KL divergence loss.
    """
    # Apply temperature scaling
    student_soft = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_soft = jax.nn.softmax(teacher_logits / temperature, axis=-1)

    # KL divergence: sum over vocab, mean over positions
    kl = jnp.sum(teacher_soft * (jnp.log(teacher_soft + 1e-10) - student_soft), axis=-1)

    # Apply mask if provided
    if mask is not None:
        kl = kl * mask
        kl_loss = jnp.sum(kl) / (jnp.sum(mask) + 1e-10)
    else:
        kl_loss = jnp.mean(kl)

    # Scale by T² to preserve gradient magnitude
    kl_loss = kl_loss * (temperature ** 2)

    return kl_loss


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Standard cross-entropy loss with hard labels.
    
    Args:
        logits: (batch, seq_len, vocab_size) model predictions.
        labels: (batch, seq_len) ground truth token IDs.
        mask: Optional (batch, seq_len) mask.
    
    Returns:
        Scalar CE loss.
    """
    vocab_size = logits.shape[-1]

    # One-hot encode labels
    one_hot = jax.nn.one_hot(labels, vocab_size)

    # Cross entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(one_hot * log_probs, axis=-1)

    if mask is not None:
        ce = ce * mask
        return jnp.sum(ce) / (jnp.sum(mask) + 1e-10)

    return jnp.mean(ce)


def distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 4.0,
    alpha: float = 0.7,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined distillation loss.
    
    L = α * KL_div(student_T, teacher_T) + (1-α) * CE(student, labels)
    
    where:
        student_T = softmax(student_logits / T)
        teacher_T = softmax(teacher_logits / T)
    
    Args:
        student_logits: Student model predictions.
        teacher_logits: Teacher soft labels.
        labels: Hard ground truth labels.
        temperature: Distillation temperature.
        alpha: Weight for distillation loss.
        mask: Optional padding mask.
    
    Returns:
        Tuple of (total_loss, loss_dict).
    """
    # Distillation loss (soft labels)
    kl_loss = kl_divergence_loss(
        student_logits, teacher_logits, temperature, mask
    )

    # Hard label loss
    ce_loss = cross_entropy_loss(student_logits, labels, mask)

    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    loss_dict = {
        "total_loss": total_loss,
        "kl_loss": kl_loss,
        "ce_loss": ce_loss,
        "alpha": alpha,
        "temperature": temperature,
    }

    return total_loss, loss_dict


class SoftLabelCache:
    """
    Cache for teacher soft labels.
    
    Pre-computing and caching teacher logits avoids querying
    expensive teacher APIs during every training step.
    
    For 1T tokens at fp16: ~2TB of soft labels
    We cache the top-k logits only to reduce storage.
    """

    def __init__(
        self,
        cache_dir: str = "gs://bharat-3b-data/soft_labels/",
        top_k: int = 100,
    ):
        self.cache_dir = cache_dir
        self.top_k = top_k

    def cache_soft_labels(
        self,
        batch_ids: jnp.ndarray,
        teacher_logits: jnp.ndarray,
    ) -> None:
        """
        Cache top-k teacher logits for a batch.
        
        Args:
            batch_ids: Unique batch identifiers.
            teacher_logits: Full teacher logits (batch, seq, vocab).
        """
        # Keep only top-k logits (sparse storage)
        top_k_vals, top_k_indices = jax.lax.top_k(teacher_logits, self.top_k)

        # TODO: Save to GCS in production
        # For now, keep in memory
        self._cache = {
            "values": top_k_vals,
            "indices": top_k_indices,
        }

    def load_soft_labels(
        self,
        batch_ids: jnp.ndarray,
        vocab_size: int,
    ) -> jnp.ndarray:
        """
        Load cached soft labels for a batch.
        
        Reconstructs full logits from sparse top-k representation.
        Non-top-k positions are filled with a large negative value.
        
        Args:
            batch_ids: Batch identifiers.
            vocab_size: Full vocabulary size.
        
        Returns:
            Reconstructed teacher logits (batch, seq, vocab).
        """
        top_k_vals = self._cache["values"]
        top_k_indices = self._cache["indices"]

        batch_size, seq_len, k = top_k_vals.shape

        # Initialize with large negative (will become ~0 after softmax)
        full_logits = jnp.full(
            (batch_size, seq_len, vocab_size), -1e10,
            dtype=top_k_vals.dtype,
        )

        # Scatter top-k values back
        batch_idx = jnp.arange(batch_size)[:, None, None]
        seq_idx = jnp.arange(seq_len)[None, :, None]

        full_logits = full_logits.at[batch_idx, seq_idx, top_k_indices].set(top_k_vals)

        return full_logits


class RecursiveDistillationTrainer:
    """
    Recursive Distillation Training Loop.
    
    Orchestrates the student-teacher training process:
    1. Generate or load teacher soft labels for batch
    2. Compute combined distillation + CE loss
    3. Update student parameters
    4. Periodically refresh soft labels with improved student
    
    Usage:
        trainer = RecursiveDistillationTrainer(
            student_model=bharat_model,
            optimizer=optimizer,
            temperature=4.0,
            alpha=0.7,
        )
        for batch in dataloader:
            metrics = trainer.train_step(state, batch, teacher_logits)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        refresh_interval: int = 10_000,
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.refresh_interval = refresh_interval
        self.soft_label_cache = SoftLabelCache()

    def compute_loss(
        self,
        student_logits: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        labels: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deq_info: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute the full training loss.
        
        Includes:
        - Distillation loss (KL divergence + CE)
        - DEQ Jacobian regularization (if available)
        - MoS diversity loss (if available)
        
        Args:
            student_logits: Student model output.
            teacher_logits: Teacher soft labels.
            labels: Hard ground truth.
            mask: Optional padding mask.
            deq_info: Optional DEQ convergence info.
        
        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Core distillation loss
        dist_loss, loss_dict = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=self.temperature,
            alpha=self.alpha,
            mask=mask,
        )

        total_loss = dist_loss

        # Add DEQ Jacobian regularization
        if deq_info and "jac_reg" in deq_info:
            total_loss = total_loss + deq_info["jac_reg"]
            loss_dict["jac_reg"] = deq_info["jac_reg"]

        loss_dict["total_loss"] = total_loss

        # Add DEQ convergence info
        if deq_info:
            loss_dict["deq_iterations"] = deq_info.get("num_iterations", 0)
            loss_dict["deq_residual"] = deq_info.get("final_residual", 0)
            loss_dict["effective_depth"] = deq_info.get("effective_depth", 0)

        return total_loss, loss_dict

    @staticmethod
    def create_train_step(
        model_apply_fn: Callable,
        optimizer: optax.GradientTransformation,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ) -> Callable:
        """
        Create a JIT-compiled training step function.
        
        Args:
            model_apply_fn: Model's apply function.
            optimizer: Optax optimizer.
            temperature: Distillation temperature.
            alpha: Distillation weight.
        
        Returns:
            JIT-compiled train_step function.
        """

        @jax.jit
        def train_step(
            params,
            opt_state,
            batch,
            teacher_logits,
            rng_key,
        ):
            """Single training step."""

            def loss_fn(params):
                # Forward pass
                outputs = model_apply_fn(
                    {"params": params},
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    deterministic=False,
                    rngs={"dropout": rng_key, "jac_reg": rng_key},
                )

                student_logits = outputs["logits"]

                # Shift logits and labels for next-token prediction
                shift_logits = student_logits[:, :-1, :]
                shift_labels = batch["input_ids"][:, 1:]
                shift_teacher = teacher_logits[:, :-1, :]

                # Compute mask (ignore padding)
                mask = (shift_labels != 0).astype(jnp.float32)

                # Distillation loss
                total_loss, loss_dict = distillation_loss(
                    student_logits=shift_logits,
                    teacher_logits=shift_teacher,
                    labels=shift_labels,
                    temperature=temperature,
                    alpha=alpha,
                    mask=mask,
                )

                return total_loss, loss_dict

            # Compute gradients
            (loss, loss_dict), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)

            # Update parameters
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Compute gradient norm for monitoring
            grad_norm = jnp.sqrt(
                sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
            )
            loss_dict["grad_norm"] = grad_norm

            return new_params, new_opt_state, loss_dict

        return train_step
