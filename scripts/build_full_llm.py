import os
import sys

# Ensure Python can find our 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_logger
from scripts.expand_tokenizer import run_pipeline as run_tokenizer_expansion
from src.model.resizer import ModelResizer
from src.data.cpt_packing import CPTPacker
from src.training.cpt_trainer import QLoRACPTTrainer
from src.training.sft_trainer import IndicSFTTrainer

logger = get_logger("master_builder")

def build_llm():
    logger.info("=====================================================")
    logger.info("🚀 IGNITING BHARAT-LLM FULL PIPELINE (PHASES 1 to 4) 🚀")
    logger.info("=====================================================")
    
    # Central Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B" 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "pipeline_output")
    
    # ---------------------------------------------------------
    # PHASE 1: VOCABULARY EXPANSION
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 1]: Expanding Tokenizer for Indian Languages ---")
    class ConfigArgs:
        base_model = BASE_MODEL
        dataset_name = "wikimedia/wikipedia"
        dataset_config = "20231101.hi"
        vocab_add_size = 10000
        doc_limit = 20000 # Keeping small for quick local tests
        output_dir = os.path.join(OUTPUT_ROOT, "phase1_tokenizer")
        
    run_tokenizer_expansion(ConfigArgs())
    expanded_tokenizer_path = os.path.join(ConfigArgs.output_dir, "merged_tokenizer")
    raw_corpus_path = os.path.join(ConfigArgs.output_dir, "data", "raw_corpus.txt")
    
    # ---------------------------------------------------------
    # PHASE 1.5: PYTORCH MATRIX RESIZING 
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 1.5]: Mathematically Resizing The Neural Network ---")
    resized_model_path = os.path.join(OUTPUT_ROOT, "phase1_resized_model")
    
    resizer = ModelResizer(BASE_MODEL, expanded_tokenizer_path, resized_model_path)
    resizer.resize_and_save()
    
    # ---------------------------------------------------------
    # PHASE 2: CONTINUOUS PRE-TRAINING (CPT)
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 2]: Teaching the Model Grammar (CPT) ---")
    cpt_dataset_dir = os.path.join(OUTPUT_ROOT, "phase2_packed_dataset")
    
    packer = CPTPacker(expanded_tokenizer_path, max_seq_length=2048)
    packer.prepare_dataset(raw_corpus_path, cpt_dataset_dir)
    
    cpt_output_model = os.path.join(OUTPUT_ROOT, "phase2_cpt_trained_model")
    cpt_trainer = QLoRACPTTrainer(resized_model_path, cpt_dataset_dir, cpt_output_model)
    cpt_trainer.train()
    
    # ---------------------------------------------------------
    # PHASE 3: DISTILLATION
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 3]: Knowledge Distillation (Groq Generation) ---")
    # For a fully automated pipeline, you can import and trigger `DistillationEngine` here.
    # Assuming you already ran `scripts/run_distillation.py` manually, we point to its output:
    distilled_json_path = os.path.join(OUTPUT_ROOT, "sft_distilled_data.jsonl")
    
    # ---------------------------------------------------------
    # PHASE 4: SUPERVISED FINE-TUNING (SFT)
    # ---------------------------------------------------------
    logger.info("\n--- [PHASE 4]: Supervised Fine Tuning (Behavior/Intelligence) ---")
    final_output_model = os.path.join(OUTPUT_ROOT, "FINAL_BHARAT_LLM")
    
    sft_trainer = IndicSFTTrainer(cpt_output_model, distilled_json_path, final_output_model)
    sft_trainer.train()

    logger.info("\n=====================================================")
    logger.info("🎉 SUCCESS! YOUR INDIC LLM IS FULLY TRAINED AND READY! 🎉")
    logger.info(f"💾 Load from: {final_output_model}")
    logger.info("=====================================================")

if __name__ == "__main__":
    build_llm()
