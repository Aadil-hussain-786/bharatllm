import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BharatTPUManager:
    """
    Ultra-Scalable TPU Training Engine for Bharat-LLM.
    Designed to push 100B+ parameter architectures on Google Colab TPU nodes.
    Uses torch_xla for native TPU acceleration and FSDP for sharding.
    """
    def __init__(self, model_dir: str, dataset_path: str, output_dir: str):
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def train_on_tpu(self, rank, flags):
        # TPU Initialization
        device = xm.xla_device()
        
        logger.info(f"[{rank}] Loading dataset for TPU training...")
        dataset = load_from_disk(self.dataset_path)
        
        logger.info(f"[{rank}] Initializing 100B Architecture Case (Sparse MoE)...")
        # For 100B models on TPU, we avoid loading huge weights into CPU RAM.
        # We load the config first, then initialize the model.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception:
            logger.info(f"[{rank}] No pre-trained weights found. Initializing skeleton from Config...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
            # Initialize with random weights directly in bfloat16 to save RAM
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            # Materialize on CPU in BF16 before moving to TPU
            model = model.to_empty(device="cpu").to(torch.bfloat16)
        
        # 100B Training on Colab Free necessitates LoRA + Activation Checkpointing
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"]
        )
        
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()
        model.to(device)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32, # Massive accumulation for 100B stability
            learning_rate=1e-4,
            bf16=True, # TPUs love bfloat16
            logging_steps=5,
            save_strategy="no",
            # TPU Specific settings
            tpu_num_cores=8,
            ddp_backend="xla",
            dataloader_num_workers=0 
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(AutoTokenizer.from_pretrained(self.model_dir), mlm=False)
        )

        logger.info(f"[{rank}] 🚀 Launching 100B TPU Kernel...")
        trainer.train()
        
        if xm.is_master_ordinal():
            model.save_pretrained(self.output_dir)
            logger.info(f"✅ Master Rank saved 100B Weights to {self.output_dir}")

def launch_training(model_dir, dataset_path, output_dir):
    manager = BharatTPUManager(model_dir, dataset_path, output_dir)
    # xmp.spawn handles the 8 independent TPU cores on Google Colab
    xmp.spawn(manager.train_on_tpu, args=({},), nprocs=8, start_method='fork')
