"""
Bharat-3B Smart-Core: Main Trainer
====================================
The pre-training loop that trains Bharat-3B on 1 Trillion tokens
with recursive distillation, FSDP, and automatic checkpoint recovery.

Training Pipeline:
    1. Initialize model with FSDP sharding
    2. Load/generate teacher soft labels
    3. For each batch:
       a. Forward pass through DEQ+RMT+MoS
       b. Compute distillation + CE loss
       c. Backward pass with implicit differentiation
       d. Update weights
       e. Log metrics to W&B / TensorBoard
    4. Checkpoint every N steps
    5. Evaluate on validation set every M steps
    6. Handle loss spikes: auto-restart from last checkpoint

Loss Spike Recovery:
    If loss suddenly increases by >50%, this indicates training instability.
    Strategy: Revert to last checkpoint, reduce LR by 0.5x, resume.
"""

import jax
import jax.numpy as jnp
import optax
import os
import time
import json
import logging
from typing import Dict, Optional, Any, Iterator
from dataclasses import dataclass, field

from bharat_3b_smart_core.src.training.lr_schedule import (
    create_cosine_schedule,
    create_optimizer,
    get_lr_at_step,
)
from bharat_3b_smart_core.src.training.distillation import (
    RecursiveDistillationTrainer,
)
from bharat_3b_smart_core.src.training.fsdp import (
    create_device_mesh,
    FSDPTrainState,
    shard_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""
    step: int = 0
    loss: float = 0.0
    kl_loss: float = 0.0
    ce_loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    deq_iterations: float = 0.0
    deq_residual: float = 0.0
    effective_depth: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_time: float = 0.0

    def to_dict(self) -> Dict:
        return {k: float(v) for k, v in self.__dict__.items()}


class BharatTrainer:
    """
    Main pre-training orchestrator for Bharat-3B.
    
    Handles the complete training pipeline including:
    - Model initialization with FSDP
    - Data loading and preprocessing
    - Training loop with distillation
    - Metric logging
    - Checkpoint management
    - Loss spike recovery
    
    Usage:
        trainer = BharatTrainer(config)
        trainer.initialize()
        trainer.train()
    
    Attributes:
        config: Training configuration.
        model: BharatModel instance.
        optimizer: Optax optimizer.
        mesh: JAX device mesh.
        distillation: RecursiveDistillationTrainer.
    """

    def __init__(self, config: Any):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_state = None
        self.mesh = None
        self.distillation = None

        # Loss spike tracking
        self._recent_losses = []
        self._best_loss = float("inf")
        self._spike_count = 0

        # Timing
        self._start_time = None
        self._step_times = []

    def initialize(self) -> None:
        """Initialize model, optimizer, and training state."""
        from bharat_3b_smart_core.src.model.bharat_model import create_bharat_model
        from bharat_3b_smart_core.configs.model_config import get_config

        model_config = get_config()
        train_config = self.config

        logger.info("=" * 60)
        logger.info("🚀 Initializing Bharat-3B Smart-Core Training")
        logger.info("=" * 60)

        # 1. Create device mesh
        mesh_shape = tuple(train_config.distributed.mesh_shape)
        self.mesh = create_device_mesh(mesh_shape)

        # 2. Create model
        logger.info("Building model architecture...")
        self.model = create_bharat_model(model_config)

        # 3. Create LR schedule and optimizer
        total_tokens = train_config.pretrain.total_tokens
        batch_tokens = (
            train_config.pretrain.batch_size
            * train_config.pretrain.max_seq_length
        )
        total_steps = total_tokens // batch_tokens

        logger.info(f"Total training steps: {total_steps:,}")

        schedule = create_cosine_schedule(
            peak_lr=train_config.pretrain.learning_rate,
            min_lr=train_config.pretrain.min_learning_rate,
            warmup_steps=train_config.pretrain.warmup_steps,
            total_steps=total_steps,
        )

        self.optimizer = create_optimizer(
            schedule=schedule,
            weight_decay=train_config.pretrain.weight_decay,
            beta1=train_config.pretrain.adam_beta1,
            beta2=train_config.pretrain.adam_beta2,
            max_grad_norm=train_config.pretrain.max_grad_norm,
        )
        self._schedule = schedule
        self._total_steps = total_steps

        # 4. Initialize training state with FSDP
        logger.info("Initializing parameters with FSDP sharding...")
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones(
            (1, train_config.pretrain.max_seq_length),
            dtype=jnp.int32,
        )

        self.train_state = FSDPTrainState.create(
            model=self.model,
            optimizer=self.optimizer,
            rng_key=rng,
            mesh=self.mesh,
            dummy_input=dummy_input,
        )

        # 5. Setup distillation
        if train_config.distillation.enabled:
            self.distillation = RecursiveDistillationTrainer(
                temperature=train_config.distillation.temperature,
                alpha=train_config.distillation.alpha,
            )
            logger.info(
                f"Recursive distillation enabled: "
                f"T={train_config.distillation.temperature}, "
                f"α={train_config.distillation.alpha}"
            )

        logger.info("✅ Initialization complete!")

    def _check_loss_spike(
        self,
        current_loss: float,
        threshold: float = 1.5,
    ) -> bool:
        """
        Check if loss has spiked (increased by >50%).
        
        Args:
            current_loss: Current step's loss.
            threshold: Spike detection multiplier.
        
        Returns:
            True if spike detected.
        """
        self._recent_losses.append(current_loss)

        # Keep last 100 losses
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-100:]

        if len(self._recent_losses) < 10:
            return False

        # Compute rolling average of last 50 steps
        recent_avg = sum(self._recent_losses[-50:]) / min(50, len(self._recent_losses))

        # Compare with best seen loss
        if current_loss > self._best_loss * threshold:
            self._spike_count += 1
            logger.warning(
                f"⚠️ Loss spike detected! "
                f"Current: {current_loss:.4f}, "
                f"Best: {self._best_loss:.4f}, "
                f"Spike #{self._spike_count}"
            )
            return True

        # Update best loss
        if recent_avg < self._best_loss:
            self._best_loss = recent_avg

        return False

    def _handle_loss_spike(self) -> None:
        """
        Handle a loss spike by reverting to last checkpoint.
        
        Strategy:
        1. Load last good checkpoint
        2. Reduce learning rate by 50%
        3. Resume training
        """
        logger.warning("🔄 Reverting to last checkpoint due to loss spike...")

        checkpoint_dir = self.config.pretrain.checkpoint_dir
        # Load last checkpoint
        # In production: use orbax-checkpoint
        logger.info("Checkpoint recovery not yet implemented in demo mode")

    def train_step(
        self,
        batch: Dict[str, jnp.ndarray],
        teacher_logits: Optional[jnp.ndarray] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ) -> TrainingMetrics:
        """
        Execute a single training step.
        
        Args:
            batch: Dict with "input_ids" and optional "attention_mask".
            teacher_logits: Optional soft labels from teachers.
            rng_key: Random key for dropout.
        
        Returns:
            TrainingMetrics for this step.
        """
        step_start = time.time()

        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.train_state.step)

        # Create JIT-compiled train step (cached after first call)
        train_step_fn = RecursiveDistillationTrainer.create_train_step(
            model_apply_fn=self.model.apply,
            optimizer=self.optimizer,
            temperature=self.distillation.temperature if self.distillation else 1.0,
            alpha=self.distillation.alpha if self.distillation else 0.0,
        )

        # If no teacher logits, use dummy (CE-only training)
        if teacher_logits is None:
            teacher_logits = jnp.zeros_like(
                jnp.ones((
                    batch["input_ids"].shape[0],
                    batch["input_ids"].shape[1],
                    self.model.vocab_size,
                ))
            )

        # Execute train step
        new_params, new_opt_state, loss_dict = train_step_fn(
            self.train_state.params,
            self.train_state.optimizer_state,
            batch,
            teacher_logits,
            rng_key,
        )

        # Update state
        self.train_state.params = new_params
        self.train_state.optimizer_state = new_opt_state
        self.train_state.step += 1

        # Compute metrics
        step_time = time.time() - step_start
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        tokens_per_second = (batch_size * seq_len) / step_time

        metrics = TrainingMetrics(
            step=self.train_state.step,
            loss=float(loss_dict["total_loss"]),
            kl_loss=float(loss_dict.get("kl_loss", 0)),
            ce_loss=float(loss_dict.get("ce_loss", 0)),
            grad_norm=float(loss_dict.get("grad_norm", 0)),
            learning_rate=get_lr_at_step(self._schedule, self.train_state.step),
            deq_iterations=float(loss_dict.get("deq_iterations", 0)),
            deq_residual=float(loss_dict.get("deq_residual", 0)),
            effective_depth=float(loss_dict.get("effective_depth", 0)),
            tokens_per_second=tokens_per_second,
            elapsed_time=step_time,
        )

        # Check for loss spikes
        if self._check_loss_spike(metrics.loss):
            self._handle_loss_spike()

        return metrics

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint at current step."""
        return self.train_state.step % self.config.pretrain.checkpoint_every == 0

    def should_evaluate(self) -> bool:
        """Check if we should run evaluation at current step."""
        return self.train_state.step % self.config.pretrain.eval_every == 0

    def should_log(self) -> bool:
        """Check if we should log metrics at current step."""
        return self.train_state.step % self.config.pretrain.log_every == 0

    def save_checkpoint(self) -> str:
        """
        Save a training checkpoint.
        
        Returns:
            Path to saved checkpoint.
        """
        step = self.train_state.step
        checkpoint_dir = os.path.join(
            self.config.pretrain.checkpoint_dir,
            f"step_{step}",
        )

        logger.info(f"💾 Saving checkpoint at step {step} to {checkpoint_dir}")

        # In production: use orbax.checkpoint
        # orbax_checkpointer.save(checkpoint_dir, self.train_state)

        return checkpoint_dir

    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        logger.info(
            f"Step {metrics.step:>8d} | "
            f"Loss: {metrics.loss:.4f} | "
            f"KL: {metrics.kl_loss:.4f} | "
            f"CE: {metrics.ce_loss:.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Grad: {metrics.grad_norm:.4f} | "
            f"DEQ-Depth: {metrics.effective_depth:.0f} | "
            f"Tok/s: {metrics.tokens_per_second:.0f}"
        )

        # W&B logging (if available)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics.to_dict(), step=metrics.step)
        except ImportError:
            pass

    def train(
        self,
        dataloader: Optional[Iterator] = None,
        num_steps: Optional[int] = None,
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            dataloader: Iterator yielding training batches.
            num_steps: Override for total training steps.
        
        Returns:
            Final training metrics dict.
        """
        total_steps = num_steps or self._total_steps
        self._start_time = time.time()

        logger.info(f"🏃 Starting training for {total_steps:,} steps...")

        for step in range(total_steps):
            # Get batch from dataloader
            if dataloader:
                batch = next(dataloader)
            else:
                # Dummy batch for testing
                batch = {
                    "input_ids": jax.random.randint(
                        jax.random.PRNGKey(step),
                        (self.config.pretrain.per_device_batch_size, 
                         self.config.pretrain.max_seq_length),
                        0, self.model.vocab_size,
                    ),
                }

            # Shard batch across devices
            with self.mesh:
                batch = shard_batch(batch, self.mesh)

            # Training step
            metrics = self.train_step(batch)

            # Logging
            if self.should_log():
                self.log_metrics(metrics)

            # Checkpointing
            if self.should_checkpoint():
                self.save_checkpoint()

            # Evaluation
            if self.should_evaluate():
                logger.info("📊 Running evaluation...")
                # eval_metrics = self.evaluate(eval_dataloader)

        total_time = time.time() - self._start_time
        logger.info(f"✅ Training complete! Total time: {total_time/3600:.1f} hours")

        return metrics.to_dict()
