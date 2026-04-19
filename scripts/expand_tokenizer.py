import argparse
import sys
import os

# Append project root to path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.extraction import CorpusExtractor
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.merger import TokenizerMerger
from src.utils.logger import get_logger

logger = get_logger("pipeline_main")

def run_pipeline(args):
    """Orchestrates the extraction, training, and merging pipeline."""
    corpus_file = os.path.join(args.output_dir, "data", "raw_corpus.txt")
    spm_prefix = os.path.join(args.output_dir, "models", "bpe", "indic_spm")

    logger.info("========== Starting Tokenizer Expansion Pipeline ==========")
    
    # 1. Extraction Phase
    extractor = CorpusExtractor(args.dataset_name, args.dataset_config)
    extractor.stream_to_disk(corpus_file, limit=args.doc_limit)
    
    # 2. Training Phase
    trainer = BPETrainer(corpus_file, spm_prefix)
    vocab_path = trainer.train(vocab_size=args.vocab_add_size)
    
    # 3. Model Merge Phase
    merger = TokenizerMerger(args.base_model, args.output_dir)
    merger.merge_and_save(vocab_path)
    
    logger.info("========== Pipeline Execution Success ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand Foundation Tokenizer")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base HuggingFace Tokenizer")
    parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia", help="HF Dataset ID")
    parser.add_argument("--dataset_config", type=str, default="20231101.hi", help="Dataset split/language")
    parser.add_argument("--vocab_add_size", type=int, default=15000, help="Vocabulary growth target")
    parser.add_argument("--doc_limit", type=int, default=50000, help="Number of documents to extract")
    parser.add_argument("--output_dir", type=str, default="./pipeline_output", help="Output storage path")
    
    args = parser.parse_args()
    
    try:
         run_pipeline(args)
    except Exception as e:
         logger.critical("Pipeline halted due to fatal error.")
         sys.exit(1)
