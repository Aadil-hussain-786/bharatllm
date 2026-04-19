import os
import sentencepiece as spm
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BPETrainer:
    """
    Core engine handling Byte-Pair Encoding logic using Google's SentencePiece.
    """
    
    def __init__(self, corpus_path: str, output_prefix: str):
        self.corpus_path = corpus_path
        self.output_prefix = output_prefix
        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)

    def train(self, vocab_size: int = 15000) -> str:
        logger.info(f"Initiating SentencePiece training. Target Vocab Size: {vocab_size}")
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus missing: {self.corpus_path}")

        try:
            spm.SentencePieceTrainer.train(
                input=self.corpus_path,
                model_prefix=self.output_prefix,
                vocab_size=vocab_size,
                model_type='bpe',
                character_coverage=0.9995,
                normalization_rule_name='nmt_nfkc',
                pad_id=0, unk_id=1, bos_id=2, eos_id=3
            )
            logger.info(f"SentencePiece training completed! Artifact prefix: {self.output_prefix}")
            return f"{self.output_prefix}.vocab"
        except Exception as e:
            logger.error(f"SentencePiece training failed: {e}")
            raise
