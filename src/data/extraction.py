import os
from datasets import load_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CorpusExtractor:
    """
    Handles streaming and extracting large datasets from HuggingFace to local storage
    without causing out-of-memory (OOM) errors in production instances.
    """
    
    def __init__(self, dataset_path: str, config_name: str, split: str = "train"):
        self.dataset_path = dataset_path
        self.config_name = config_name
        self.split = split

    def stream_to_disk(self, output_file: str, limit: int = 50000):
        logger.info(f"Connecting to HF Dataset: {self.dataset_path} ({self.config_name})")
        try:
            dataset = load_dataset(self.dataset_path, self.config_name, split=self.split, streaming=True)
            logger.info(f"Extracting docs to {output_file} (Limit: {limit})")
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                count = 0
                for item in dataset:
                    text = item.get('text', '').replace('\n', ' ').strip()
                    if text:
                        f.write(text + "\n")
                        count += 1
                    if count >= limit:
                        break
            logger.info(f"Extracted {count} documents successfully.")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
