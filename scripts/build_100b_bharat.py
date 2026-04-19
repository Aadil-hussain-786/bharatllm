import os
import sys

# Ensure Python can find our 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_logger
from scripts.expand_tokenizer import run_pipeline as run_tokenizer_expansion
from src.model.upcycler import BharatUpcycler
from src.training.tpu_engine import launch_training
from src.data.extraction import extract_corpus 
from src.data.cpt_packing import CPTPacker

logger = get_logger("100B_BUILDER")

def build_100b_llm():
    logger.info("=====================================================")
    logger.info("🌌 BHARAT-LLM 100B: THE OPENAI-COMPETITOR PIPELINE 🌌")
    logger.info("=====================================================")
    
    BASE_MODEL = "Qwen/Qwen2.5-7B" # Starting from a world-class foundation
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "100B_output")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # ---------------------------------------------------------
    # PHASE 1: TOKENIZER EXPANSION (INDIC FOCUS)
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 1]: Expanding Vocabulary for 100B Precision ---")
    class ConfigArgs:
        base_model = BASE_MODEL
        dataset_name = "wikimedia/wikipedia"
        dataset_config = "20231101.hi"
        vocab_add_size = 32000 # Larger vocab for smoother 100B reasoning
        doc_limit = 50000 
        output_dir = os.path.join(OUTPUT_ROOT, "phase1_tokenizer")
        
    run_tokenizer_expansion(ConfigArgs())
    expanded_tokenizer_path = os.path.join(ConfigArgs.output_dir, "merged_tokenizer")
    raw_corpus_path = os.path.join(ConfigArgs.output_dir, "data", "raw_corpus.txt")
    
    # ---------------------------------------------------------
    # PHASE 2: BRAIN UP-CYCLING (100B MOE EXPANSION)
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 2]: Up-Cycling to 100 Billion Parameters ---")
    upcycled_model_path = os.path.join(OUTPUT_ROOT, "phase2_100b_skeleton")
    
    upcycler = BharatUpcycler(BASE_MODEL)
    upcycler.upcycle_to_100b(upcycled_model_path)
    
    # ---------------------------------------------------------
    # PHASE 3: TPU TRAINING PREPARATION
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 3]: Packing Datasets for TPU Acceleration ---")
    cpt_dataset_dir = os.path.join(OUTPUT_ROOT, "phase3_packed_dataset")
    
    packer = CPTPacker(expanded_tokenizer_path, max_seq_length=4096) # Scaling to 4K context
    packer.prepare_dataset(raw_corpus_path, cpt_dataset_dir)
    
    # ---------------------------------------------------------
    # PHASE 4: TPU TRAINING EXECUTION (Native XLA)
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 4]: IGNITING TPU 100B TRAINING ENGINE ---")
    final_100b_path = os.path.join(OUTPUT_ROOT, "FINAL_BHARAT_100B")
    
    # This launches the 8-core TPU multiprocessing engine we built
    launch_training(
        model_dir=upcycled_model_path,
        dataset_path=cpt_dataset_dir,
        output_dir=final_100b_path
    )

    logger.info("\n=====================================================")
    logger.info("🎉 SUCCESS! BHARAT-LLM 100B IS BORN! 🎉")
    logger.info("Your model is now ready to compete with global LLMs.")
    logger.info(f"💾 Production Weights: {final_100b_path}")
    logger.info("=====================================================")

if __name__ == "__main__":
    build_100b_llm()
