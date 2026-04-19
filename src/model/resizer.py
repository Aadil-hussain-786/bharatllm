import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelResizer:
    """
    Expands the LLM's internal mathematical embedding and output layers 
    to physically accommodate the newly injected Indic tokens.
    Without this step, the model will throw an index out-of-bounds crash
    when it sees an ID belonging to our new language tokens.
    """
    def __init__(self, base_model_id: str, expanded_tokenizer_path: str, output_model_path: str):
        self.base_model_id = base_model_id
        self.expanded_tokenizer_path = expanded_tokenizer_path
        self.output_model_path = output_model_path
        os.makedirs(self.output_model_path, exist_ok=True)

    def resize_and_save(self):
        logger.info(f"Loading newly expanded Tokenizer from {self.expanded_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.expanded_tokenizer_path)
        new_vocab_size = len(tokenizer)

        logger.info(f"Loading Foundation weights: {self.base_model_id} (in bfloat16 to save RAM)...")
        try:
            # We use bfloat16 to keep the RAM footprint small while injecting new layers
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            logger.info(f"Original Embedding Tensors: {model.get_input_embeddings().weight.shape}")
            logger.info(f"Reszing mathematically to match new Vocab Size: {new_vocab_size}")
            
            # ---> THE CRITICAL FLIPPING SWITCH <---
            # This manipulates PyTorch to append blank vectors to Qwen's LM Head & Embeddings
            model.resize_token_embeddings(new_vocab_size)
            
            logger.info(f"Expanded Embedding Tensors: {model.get_input_embeddings().weight.shape}")
            
            logger.info(f"Writing expanded Foundation Model to disk at: {self.output_model_path}...")
            model.save_pretrained(self.output_model_path)
            
            # The Tokenizer must also live inside the final model dir
            tokenizer.save_pretrained(self.output_model_path)
            
            logger.info("✅ Resizing Complete! The blank token spaces are ready to be filled with knowledge in Phase 2.")
            return self.output_model_path
            
        except Exception as e:
            logger.error(f"Failed to resize mathematical model layers: {e}")
            raise
