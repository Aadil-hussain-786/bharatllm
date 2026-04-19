"""
Bharat-3B Smart-Core: Supervised Fine-Tuning (SFT)
====================================================
Phase 4, Step 4.1: Turn the pre-trained "predictor" into a "chat assistant".

SFT trains the model on 50k high-quality instruction-response pairs
in Hindi, English, and Hinglish. This is the critical step that gives
the model its conversational ability.

Data Format:
    {"instruction": "...", "response": "...", "language": "hindi/english/hinglish"}

Training Details:
    - 50k examples, 3 epochs
    - Learning rate: 2e-5 (much lower than pre-training)
    - Max sequence length: 8192 tokens
    - Loss only on response tokens (not instruction)
"""

import jax
import jax.numpy as jnp
import optax
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Chat template for multi-turn conversations
CHAT_TEMPLATE = """<bos>### System:
You are Bharat-3B, an AI assistant created by Bharat AI Labs. You are helpful, harmless, and honest. You can communicate fluently in Hindi, English, and Hinglish.

### User:
{instruction}

### Assistant:
{response}<eos>"""

MULTI_TURN_TEMPLATE = """<bos>### System:
You are Bharat-3B, an AI assistant created by Bharat AI Labs.

{turns}### Assistant:
{response}<eos>"""


@dataclass
class SFTExample:
    """A single SFT training example."""
    instruction: str
    response: str
    language: str = "english"
    system_prompt: Optional[str] = None


def format_sft_example(
    example: SFTExample,
    tokenizer: Any,
    max_length: int = 8192,
) -> Dict[str, jnp.ndarray]:
    """
    Format a single SFT example into model input tensors.
    
    Critical: Loss is computed ONLY on the response part.
    The instruction tokens are masked out in the loss computation
    to prevent the model from learning to "generate" instructions.
    
    Args:
        example: SFT training example.
        tokenizer: Bharat tokenizer instance.
        max_length: Maximum sequence length.
    
    Returns:
        Dict with "input_ids", "attention_mask", "labels".
    """
    formatted = CHAT_TEMPLATE.format(
        instruction=example.instruction,
        response=example.response,
    )

    # Tokenize
    token_ids = tokenizer.encode(formatted, add_special_tokens=False, max_length=max_length)

    # Find where the response starts (after "### Assistant:\n")
    instruction_part = CHAT_TEMPLATE.split("{response}")[0].format(
        instruction=example.instruction,
    )
    instruction_tokens = tokenizer.encode(
        instruction_part, add_special_tokens=False
    )
    response_start = len(instruction_tokens)

    # Create labels: -100 for instruction tokens (ignored in loss)
    labels = [-100] * response_start + token_ids[response_start:]

    # Pad/truncate to max_length
    pad_id = tokenizer.pad_token_id
    if len(token_ids) < max_length:
        padding = max_length - len(token_ids)
        token_ids += [pad_id] * padding
        labels += [-100] * padding
    else:
        token_ids = token_ids[:max_length]
        labels = labels[:max_length]

    # Attention mask
    attention_mask = [1 if t != pad_id else 0 for t in token_ids]

    return {
        "input_ids": jnp.array(token_ids, dtype=jnp.int32),
        "attention_mask": jnp.array(attention_mask, dtype=jnp.int32),
        "labels": jnp.array(labels, dtype=jnp.int32),
    }


def sft_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute SFT cross-entropy loss with label masking.
    
    Labels with value -100 are ignored in the loss computation.
    
    Args:
        logits: (batch, seq_len, vocab_size) model output.
        labels: (batch, seq_len) with -100 for masked positions.
    
    Returns:
        Scalar loss value.
    """
    vocab_size = logits.shape[-1]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Create mask: 1 where label != -100
    mask = (shift_labels != -100).astype(jnp.float32)

    # Replace -100 with 0 for one-hot encoding
    safe_labels = jnp.where(shift_labels == -100, 0, shift_labels)

    # Cross entropy
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    one_hot = jax.nn.one_hot(safe_labels, vocab_size)
    ce = -jnp.sum(one_hot * log_probs, axis=-1)

    # Apply mask
    masked_loss = ce * mask
    loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-10)

    return loss


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer.
    
    Takes a pre-trained Bharat-3B checkpoint and fine-tunes it
    on 50k instruction-response pairs to create a chat model.
    
    Usage:
        trainer = SFTTrainer(model, config)
        trainer.train(train_dataset)
        trainer.save("checkpoints/sft/")
    """

    def __init__(
        self,
        model: Any,
        config: Any,
    ):
        self.model = model
        self.config = config

        # SFT uses much lower LR than pre-training
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=optax.linear_schedule(
                    init_value=0.0,
                    end_value=config.sft.learning_rate,
                    transition_steps=int(config.sft.num_examples * config.sft.warmup_ratio),
                ),
                weight_decay=0.01,
            ),
        )

    @staticmethod
    def create_sft_train_step(
        model_apply_fn,
        optimizer,
    ):
        """Create JIT-compiled SFT training step."""

        @jax.jit
        def train_step(params, opt_state, batch, rng_key):
            def loss_fn(params):
                outputs = model_apply_fn(
                    {"params": params},
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    deterministic=False,
                    rngs={"dropout": rng_key},
                )
                return sft_loss(outputs["logits"], batch["labels"])

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, {"loss": loss}

        return train_step

    def train(
        self,
        train_examples: List[SFTExample],
        tokenizer: Any,
        num_epochs: int = 3,
    ):
        """
        Run SFT training.
        
        Args:
            train_examples: List of SFT training examples.
            tokenizer: Bharat tokenizer.
            num_epochs: Number of training epochs.
        """
        logger.info(f"Starting SFT with {len(train_examples)} examples, {num_epochs} epochs")

        # Initialize optimizer state
        # In production: load pre-trained checkpoint first
        # params = load_checkpoint(config.pretrain.checkpoint_dir)
        # opt_state = self.optimizer.init(params)

        train_step_fn = self.create_sft_train_step(
            self.model.apply, self.optimizer
        )

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, example in enumerate(train_examples):
                batch = format_sft_example(example, tokenizer)
                # Add batch dimension
                batch = {k: v[jnp.newaxis] for k, v in batch.items()}

                rng = jax.random.PRNGKey(epoch * len(train_examples) + i)
                # params, opt_state, metrics = train_step_fn(params, opt_state, batch, rng)
                # epoch_loss += metrics["loss"]

            avg_loss = epoch_loss / max(len(train_examples), 1)
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
