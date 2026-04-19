import os
import json
from transformers import AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TokenizerMerger:
    """
    Responsible for merging the base Tokenizer (e.g. Qwen) with our customized BPE token set,
    and generating metrics for MLOps tracking.
    """
    
    def __init__(self, base_model_id: str, output_dir: str):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_and_save(self, sp_vocab_path: str):
        logger.info(f"Loading Base Tokenizer: {self.base_model_id}")
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            original_size = len(base_tokenizer)
            
            with open(sp_vocab_path, "r", encoding="utf-8") as f:
                sp_vocab = f.readlines()

            indic_tokens = []
            for line in sp_vocab:
                token, _ = line.split('\t')
                token = token.replace(' ', ' ')
                if len(token) > 1 and not token.startswith('<0x'):
                    indic_tokens.append(token)

            num_added = base_tokenizer.add_tokens(indic_tokens)
            logger.info(f"Injected {num_added} unique Indic tokens natively.")
            
            save_path = os.path.join(self.output_dir, "merged_tokenizer")
            base_tokenizer.save_pretrained(save_path)
            
            # W&B / MLflow compatible metrics
            metrics = {
                "base_model": self.base_model_id,
                "original_vocab_size": original_size,
                "new_vocab_size": len(base_tokenizer),
                "tokens_added": num_added
            }
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"Tokenizer merged! Saved to: {save_path}")
            return metrics
            
        except Exception as e:
            logger.error(f"Merger logic failed: {e}")
            raise
