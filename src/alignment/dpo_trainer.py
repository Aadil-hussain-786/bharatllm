"""
Bharat-3B Smart-Core: Direct Preference Optimization (DPO)
============================================================
Phase 4, Step 4.2: Align model preferences without a reward model.

Why DPO over RLHF?
    - RLHF requires training a separate reward model (expensive)
    - DPO directly optimizes the policy from preference data
    - Simpler, more stable, and empirically matches RLHF quality

DPO Process:
    1. Model generates 2 responses for each prompt
    2. Teacher (Gemini 1.5 Pro) judges which is "better"
    3. DPO loss pushes model to prefer the "chosen" response
       and avoid the "rejected" response

DPO Loss:
    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    
    where:
        π = current policy (being trained)
        π_ref = reference policy (SFT checkpoint, frozen)
        y_w = chosen (winning) response
        y_l = rejected (losing) response
        β = KL penalty coefficient

References:
    - Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
"""

import jax
import jax.numpy as jnp
import optax
import logging
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single DPO preference pair."""
    prompt: str
    chosen: str  # Preferred response (y_w)
    rejected: str  # Rejected response (y_l)
    teacher_judge: str = "gemini-1.5-pro"
    confidence: float = 1.0


def dpo_loss(
    policy_chosen_logps: jnp.ndarray,
    policy_rejected_logps: jnp.ndarray,
    ref_chosen_logps: jnp.ndarray,
    ref_rejected_logps: jnp.ndarray,
    beta: float = 0.1,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute DPO loss.
    
    L = -log σ(β * ((log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))))
    
    Args:
        policy_chosen_logps: Log-prob of chosen response under policy.
        policy_rejected_logps: Log-prob of rejected response under policy.
        ref_chosen_logps: Log-prob of chosen under reference (SFT).
        ref_rejected_logps: Log-prob of rejected under reference.
        beta: KL Divergence penalty coefficient.
    
    Returns:
        Tuple of (scalar loss, metrics dict).
    """
    # Log-ratio differences
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # DPO logits
    logits = beta * (chosen_logratios - rejected_logratios)

    # Loss: negative log-sigmoid
    loss = -jax.nn.log_sigmoid(logits).mean()

    # Metrics
    chosen_rewards = beta * chosen_logratios.mean()
    rejected_rewards = beta * rejected_logratios.mean()
    reward_margin = (chosen_rewards - rejected_rewards)
    accuracy = (logits > 0).astype(jnp.float32).mean()

    metrics = {
        "loss": loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_margin": reward_margin,
        "accuracy": accuracy,
        "logits_mean": logits.mean(),
    }

    return loss, metrics


def compute_log_probs(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute per-sequence log probabilities.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        mask: (batch, seq_len) - 1 for valid tokens, 0 for padding
    
    Returns:
        Per-sequence log probabilities (batch,)
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:]

    # Per-token log probs
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_log_probs = jnp.take_along_axis(
        log_probs,
        shift_labels[:, :, jnp.newaxis],
        axis=-1,
    ).squeeze(-1)

    # Mask and sum per sequence
    masked_log_probs = token_log_probs * shift_mask
    seq_log_probs = jnp.sum(masked_log_probs, axis=-1) / (
        jnp.sum(shift_mask, axis=-1) + 1e-10
    )

    return seq_log_probs


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.
    
    Stage 4.2 of the Bharat-3B pipeline.
    Uses the SFT checkpoint as the frozen reference model and
    trains the policy to prefer chosen over rejected responses.
    
    Teacher Judge:
        Gemini 1.5 Pro acts as the judge, deciding which of two
        model-generated responses is "better" for each prompt.
    
    Usage:
        dpo = DPOTrainer(
            policy_model=model,
            ref_model=frozen_sft_model,
            beta=0.1,
        )
        dpo.train(preference_pairs)
    """

    def __init__(
        self,
        policy_model: Any,
        ref_model: Any,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=learning_rate,
                weight_decay=0.0,  # No weight decay for DPO
            ),
        )

    async def generate_preference_pairs(
        self,
        prompts: List[str],
        tokenizer: Any,
        teacher_judge: str = "gemini-1.5-pro",
    ) -> List[PreferencePair]:
        """
        Generate preference pairs using the model and teacher judge.
        
        For each prompt:
        1. Generate 2 responses with different temperatures
        2. Ask teacher to judge which is better
        3. Create a PreferencePair
        
        Args:
            prompts: List of prompts.
            tokenizer: Tokenizer for encoding/decoding.
            teacher_judge: Teacher model for judging.
        
        Returns:
            List of PreferencePair objects.
        """
        from bharat_3b_smart_core.src.data.synthetic_engine import GeminiTeacher

        judge = GeminiTeacher()
        pairs = []

        for prompt in prompts:
            # Generate 2 responses (with different temperatures/seeds)
            # In production: use model.generate() with different configs
            response_a = f"[Response A for: {prompt[:50]}...]"
            response_b = f"[Response B for: {prompt[:50]}...]"

            # Ask teacher to judge
            judge_prompt = (
                f"Compare these two responses to the question: '{prompt}'\n\n"
                f"Response A: {response_a}\n\n"
                f"Response B: {response_b}\n\n"
                f"Which response is better? Reply with just 'A' or 'B' "
                f"and a brief explanation."
            )

            judgment = await judge.generate(judge_prompt, max_tokens=100)

            if "A" in judgment.upper()[:10]:
                chosen, rejected = response_a, response_b
            else:
                chosen, rejected = response_b, response_a

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                teacher_judge=teacher_judge,
            ))

        logger.info(f"Generated {len(pairs)} preference pairs")
        return pairs

    @staticmethod
    def create_dpo_train_step(
        policy_apply_fn,
        ref_apply_fn,
        optimizer,
        beta: float = 0.1,
    ):
        """Create JIT-compiled DPO training step."""

        @jax.jit
        def train_step(
            policy_params,
            ref_params,
            opt_state,
            chosen_batch,
            rejected_batch,
        ):
            def loss_fn(params):
                # Policy log-probs for chosen and rejected
                chosen_out = policy_apply_fn(
                    {"params": params},
                    chosen_batch["input_ids"],
                    deterministic=True,
                )
                rejected_out = policy_apply_fn(
                    {"params": params},
                    rejected_batch["input_ids"],
                    deterministic=True,
                )

                policy_chosen_logps = compute_log_probs(
                    chosen_out["logits"],
                    chosen_batch["input_ids"],
                    chosen_batch["attention_mask"],
                )
                policy_rejected_logps = compute_log_probs(
                    rejected_out["logits"],
                    rejected_batch["input_ids"],
                    rejected_batch["attention_mask"],
                )

                # Reference log-probs (no gradient)
                ref_chosen_out = ref_apply_fn(
                    {"params": ref_params},
                    chosen_batch["input_ids"],
                    deterministic=True,
                )
                ref_rejected_out = ref_apply_fn(
                    {"params": ref_params},
                    rejected_batch["input_ids"],
                    deterministic=True,
                )

                ref_chosen_logps = compute_log_probs(
                    ref_chosen_out["logits"],
                    chosen_batch["input_ids"],
                    chosen_batch["attention_mask"],
                )
                ref_rejected_logps = compute_log_probs(
                    ref_rejected_out["logits"],
                    rejected_batch["input_ids"],
                    rejected_batch["attention_mask"],
                )

                loss, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    jax.lax.stop_gradient(ref_chosen_logps),
                    jax.lax.stop_gradient(ref_rejected_logps),
                    beta=beta,
                )

                return loss, metrics

            (loss, metrics), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(policy_params)

            updates, new_opt_state = optimizer.update(
                grads, opt_state, policy_params
            )
            new_params = optax.apply_updates(policy_params, updates)

            return new_params, new_opt_state, metrics

        return train_step
