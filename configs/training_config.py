"""
Bharat-3B Smart-Core: Training Configuration
=============================================
Covers pre-training, distillation, SFT, and DPO settings.
"""

import ml_collections


def get_training_config() -> ml_collections.ConfigDict:
    """Returns the default training configuration."""
    config = ml_collections.ConfigDict()

    # ========================================
    # Phase 3: Pre-training Settings
    # ========================================
    config.pretrain = ml_collections.ConfigDict()
    config.pretrain.total_tokens = 1_000_000_000_000  # 1 Trillion tokens
    config.pretrain.batch_size = 256  # Global batch size
    config.pretrain.per_device_batch_size = 32  # Per TPU core
    config.pretrain.gradient_accumulation_steps = 1
    config.pretrain.max_seq_length = 4096  # Per-segment length during pretraining
    config.pretrain.num_epochs = 1  # Single pass over data

    # Learning Rate Schedule
    config.pretrain.learning_rate = 3e-4  # Peak LR
    config.pretrain.min_learning_rate = 3e-5  # Min LR (cosine decay target)
    config.pretrain.warmup_steps = 2000  # LR warm-up steps
    config.pretrain.lr_schedule = "cosine"  # Options: "cosine", "linear", "constant"
    config.pretrain.weight_decay = 0.1
    config.pretrain.max_grad_norm = 1.0  # Gradient clipping

    # Optimizer
    config.pretrain.optimizer = "adamw"
    config.pretrain.adam_beta1 = 0.9
    config.pretrain.adam_beta2 = 0.95
    config.pretrain.adam_epsilon = 1e-8

    # Checkpointing
    config.pretrain.checkpoint_every = 1000  # Save every N steps
    config.pretrain.eval_every = 500
    config.pretrain.log_every = 10
    config.pretrain.checkpoint_dir = "gs://bharat-3b-checkpoints/pretrain/"
    config.pretrain.max_checkpoints = 5  # Keep last N checkpoints

    # ========================================
    # Recursive Distillation Settings
    # ========================================
    config.distillation = ml_collections.ConfigDict()
    config.distillation.enabled = True
    config.distillation.temperature = 4.0  # Softmax temperature for soft labels
    config.distillation.alpha = 0.7  # Weight for distillation loss (vs CE loss)
    config.distillation.teacher_models = [
        "gemini-1.5-pro",
        "llama-3.1-405b",
        "gpt-4",
    ]
    config.distillation.consensus_threshold = 2  # Min teachers that must agree
    config.distillation.soft_label_cache = "gs://bharat-3b-data/soft_labels/"

    # ========================================
    # Phase 4: SFT Settings
    # ========================================
    config.sft = ml_collections.ConfigDict()
    config.sft.num_examples = 50_000  # High-quality instruction pairs
    config.sft.batch_size = 32
    config.sft.learning_rate = 2e-5
    config.sft.num_epochs = 3
    config.sft.max_seq_length = 8192
    config.sft.warmup_ratio = 0.03
    config.sft.languages = ["hindi", "english", "hinglish"]
    config.sft.checkpoint_dir = "gs://bharat-3b-checkpoints/sft/"

    # ========================================
    # Phase 4: DPO Settings
    # ========================================
    config.dpo = ml_collections.ConfigDict()
    config.dpo.beta = 0.1  # KL divergence penalty
    config.dpo.batch_size = 16
    config.dpo.learning_rate = 5e-7
    config.dpo.num_epochs = 1
    config.dpo.max_seq_length = 4096
    config.dpo.reference_model = "sft"  # Use SFT checkpoint as reference
    config.dpo.teacher_judge = "gemini-1.5-pro"  # Teacher that ranks preferences
    config.dpo.checkpoint_dir = "gs://bharat-3b-checkpoints/dpo/"

    # ========================================
    # FSDP / Distributed Training
    # ========================================
    config.distributed = ml_collections.ConfigDict()
    config.distributed.strategy = "fsdp"  # Fully Sharded Data Parallel
    config.distributed.num_tpu_cores = 8  # TPU v3-8
    config.distributed.shard_axes = {
        "embed": ("mp", None),
        "attention": ("mp", None, None, None),
        "ffn": (None, "mp"),
    }
    config.distributed.mesh_shape = (1, 8)  # (data_parallel, model_parallel)
    config.distributed.mixed_precision = True
    config.distributed.dtype = "bfloat16"

    # ========================================
    # Compute Budget
    # ========================================
    config.compute = ml_collections.ConfigDict()
    config.compute.tpu_type = "v3-8"
    config.compute.hbm_per_core_gb = 16  # 16GB per core × 8 = 128GB total
    config.compute.estimated_flops = 1.2e20  # Total training FLOPs
    config.compute.target_mfu = 0.45  # Model FLOP Utilization target

    return config
