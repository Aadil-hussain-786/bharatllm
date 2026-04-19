import os
import sys

# Attempt to import dotenv, fail gracefully if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure we can import the src/ modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.distillation_engine import DistillationEngine
from src.utils.logger import get_logger

logger = get_logger("distillation_runner")

def main():
    logger.info("========== Starting Phase 3: Knowledge Distillation ==========")
    
    # Securely retrieve the key we saved in the .env file
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.critical("Failed to find GROQ_API_KEY. Did you save the .env file?")
        sys.exit(1)
        
    try:
        # Initialize the hardware-accelerated Groq Engine
        # Let's use Google's Gemma 2 which is extremely stable and highly capable of reasoning
        engine = DistillationEngine(api_key=api_key, teacher_model="gemma2-9b-it")
        
        # A startup-grade test bed: We ask it complex Western philosophy, coding, and math
        # and forcefully demand the output steps be rendered in perfect Hindi.
        seed_topics_test = [
            "Explain PyTorch Backpropagation using an analogy of a busy street market in Mumbai.",
            "Write a Python function to solve the Traveling Salesperson Problem and explain the computational complexity.",
            "Compare and contrast the economic impacts of quantum computing on modern banking logistics.",
            "If it takes 4 painters 6 hours to paint 2 houses, exactly how many hours will it take 8 painters to paint 5 houses? Provide step-by-step logic."
        ]
        
        output_data_path = os.path.join(os.path.dirname(__file__), "..", "pipeline_output", "sft_distilled_data.jsonl")
        
        # Trigger the LPU infrastructure!
        engine.generate_sft_dataset(
            seed_topics=seed_topics_test,
            output_jsonl_path=output_data_path,
            target_language="Hindi"
        )
        
        logger.info("========== Distillation Run Successful ==========")
        
    except Exception as e:
        logger.critical(f"Fatal error during distillation run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
