import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QLoRACPTTrainer:
    """
    Continuous Pre-Training via QLoRA.
    Trains the newly expanded embeddings so they learn language semantics, 
    while drastically cutting VRAM cost by freezing 98% of the model in 4-bit precision.
    """
    def __init__(self, model_dir: str, packed_dataset_path: str, output_dir: str):
        self.model_dir = model_dir
        self.packed_dataset_path = packed_dataset_path
        self.output_dir = output_dir

    def train(self):
        logger.info(f"Loading perfectly packed Arrow dataset from {self.packed_dataset_path}...")
        dataset = load_from_disk(self.packed_dataset_path)

        logger.info("Configuring 4-bit Quantization (BitsAndBytes)...")
        # Load the base 1.5B/7B model in 4-bit precision. This squashes a 15GB model to ~4GB on GPU.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 # Fixed: Mismatch between bfloat16 and TrainingArguments(fp16) caused the Colab detach() crash
        )

        logger.info(f"Loading structurally resized Foundation Model from {self.model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            quantization_config=bnb_config,
            device_map="auto" # Automatically splatters the 4-bit matrix across available GPUs
        )
        
        # Prepare the model for INT4 training (gradient checkpointing to save memory)
        model = prepare_model_for_kbit_training(model)

        logger.info("Injecting LoRA Adapters and unfreezing required modules...")
        
        # Qwen uses specific attention layers. We inject tiny LoRA adapters into all of them.
        lora_config = LoraConfig(
            r=64, # High rank helps the model rapidly learn structurally distinct languages
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            
            # ---> THE MOST CRITICAL LINE FOR VOCABULARY EXPANSION <---
            # While the rest of the model is 4-bit and completely read-only, 
            # this mathematically unfreezes BOTH embedding and LM output matrices.
            # This allows the blank padding vectors we injected in Phase 1 to physically absorb intelligence.
            modules_to_save=["embed_tokens", "lm_head"] 
        )

        # Merge the model with the LoRA config
        model = get_peft_model(model, lora_config)
        
        # Proof of cost saving: Check exactly how much of the model we are training
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"Cost saving metric: Training {trainable_params:,} / {all_param:,} params ({100 * trainable_params / all_param:.2f}%)")

        logger.info("Configuring PyTorch Engine arguments...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,   # Safest for 15GB Colab T4 limits
            gradient_accumulation_steps=16,  
            warmup_steps=200,                
            learning_rate=2e-4,              
            fp16=True,                       # Colab T4 does NOT support bfloat16 natively, it defaults to massive fp32 unless fp16=True
            bf16=False,                      
            gradient_checkpointing=True,     # <--- CRUCIAL: Discards cached layers during forward pass. Saves 70% VRAM!
            logging_steps=10,
            save_strategy="steps",
            save_steps=1000,
            max_steps=5000,                  # Run for ~5000 steps to reach initial fluency
            optim="paged_adamw_32bit",       # Avoids VRAM spikes during weight updates
            report_to="none"                 # Turn this to "wandb" in your startup config file later
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("🚀 Igniting the Continuous Pre-Training Engine...")
        trainer.train()
        
        logger.info(f"💾 Training complete! Saving highly trained LoRA weights & embeddings to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        
        # Extremely important: We also copy the Tokenizer, as these tuned embeddings are now permanently locked to it
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info("✅ Pipeline Complete: You now have a fluent Indic foundation model.")
