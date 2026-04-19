import os
from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CPTPacker:
    """
    Handles 'Sequence Packing' for Continuous Pre-Training (CPT).
    Instead of feeding the GPU short sentences, which wastes expensive compute on 'Padding',
    this strings all documents together separated by an <EOS> token, and chops them into 
    perfect blocks of exactly 4096 tokens to achieve 100% GPU saturation.
    """
    def __init__(self, tokenizer_path: str, max_seq_length: int = 4096):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = max_seq_length

    def prepare_dataset(self, text_file_path: str, output_dir: str):
        logger.info(f"Loading raw corpus from {text_file_path}")
        
        # Load the raw text file as a HF Dataset
        dataset = load_dataset('text', data_files={'train': text_file_path})['train']
        
        logger.info("Step 1: Tokenizing entire dataset (this may take a while)...")
        # Tokenize everything but don't pad or truncate yet
        tokenized_datasets = dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=os.cpu_count() or 4, # Use all available CPU cores for speed
            remove_columns=["text"],
            desc="Running Custom Indic Tokenizer"
        )
        
        logger.info(f"Step 2: Packing tokenized data into massive blocks of {self.max_seq_length} tokens...")
        # Pack the sequences
        packed_dataset = tokenized_datasets.map(
            self._group_texts,
            batched=True,
            num_proc=os.cpu_count() or 4,
            desc="Packing sequences for 100% GPU saturation"
        )
        
        logger.info(f"Saving packed arrow dataset to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        packed_dataset.save_to_disk(output_dir)
        
        logger.info(f" Prepared {len(packed_dataset)} strictly packed 4096-token blocks for CPT.")
        return output_dir

    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Fast batched tokenizer mapping."""
        # Append EOS token so the model learns where documents natively end
        texts = [t + self.tokenizer.eos_token for t in examples["text"] if len(t.strip()) > 0]
        return self.tokenizer(texts, add_special_tokens=False)

    def _group_texts(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Concatenates all texts and splits them into perfectly uniform chunks."""
        # 1. Smash all tokenized documents into one gigantic 1-dimensional array
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 2. Drop the small remainder at the end so chunks are absolutely perfectly sized
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        
        # 3. Chop the massive array into exactly 4096 token blocks
        result = {
            k: [t[i : i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        
        # In causal language modeling, the labels are just the input_ids shifted by one (HF handles the shift)
        result["labels"] = result["input_ids"].copy()
        return result
