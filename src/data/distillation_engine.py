import os
import json
import time
from groq import Groq
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DistillationEngine:
    """
    Phase 3: Knowledge Distillation.
    Connects to ultra-fast inference on Groq's LPU servers to generate synthetic,
    high-quality reasoning data to fine-tune our smaller Qwen "Student" model.
    Using Llama-3-70B or Google's Gemma via Groq generates answers at 800+ tokens per second.
    """
    def __init__(self, api_key: str, teacher_model: str = "llama3-70b-8192"):
        if not api_key:
            raise ValueError("Groq API Key is missing. Required for lightning-fast distillation.")
        
        # Configure the Groq client natively
        self.client = Groq(api_key=api_key)
        self.teacher_model = teacher_model
        
    def generate_sft_dataset(self, seed_topics: list, output_jsonl_path: str, target_language: str = "Hindi"):
        """
        Uses a list of topics to trigger Groq into generating highly complex Q&A pairs in the target Indic language.
        """
        logger.info(f"Initiating Distillation sequence via Groq on '{self.teacher_model}' for {len(seed_topics)} topics...")
        
        # Enforcing strict JSON output so it cleanly parses into our dataset format
        system_instruction = f"""
        You are a PhD-level agent generating Supervised Fine-Tuning (SFT) data to train a new AI model for India.
        For the given topic, write an extremely complex, challenging question in perfectly natural {target_language}.
        Then, write a flawless, step-by-step reasoning answer to that question in {target_language}.
        
        Output EXCLUSIVELY as raw JSON matching this schema exactly:
        {{"user_prompt": "<the complex question>", "assistant_response": "<the step by step answer>"}}
        """
        
        sft_data = []
        
        for i, topic in enumerate(seed_topics):
            logger.info(f"Extracting intelligence [{i+1}/{len(seed_topics)}]: {topic}")
            try:
                # Trigger the distillation via Groq API
                completion = self.client.chat.completions.create(
                    model=self.teacher_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Topic: {topic}"}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"} # Forces perfect JSON mode
                )
                
                # Parse the JSON out of the Groq response
                clean_text = completion.choices[0].message.content
                extracted_json = json.loads(clean_text)
                
                # Restructure into standard ChatML / OpenAI message format which Qwen natively expects
                formatted_message = {
                    "messages": [
                        {"role": "user", "content": extracted_json["user_prompt"]},
                        {"role": "assistant", "content": extracted_json["assistant_response"]}
                    ]
                }
                
                sft_data.append(formatted_message)
                
                # Prevent rate-limits (Groq is fast but still enforces RPM boundaries)
                time.sleep(1.5)
                
            except Exception as e:
                logger.error(f"Failed to distill knowledge for topic '{topic}'. Skipped. Error: {e}")
                
        # Persist as JSONL (JSON Lines) which is the industry standard for LLM training
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Distillation complete! Extracted {len(sft_data)} high-variance data points via Groq.")
        logger.info(f"Dataset successfully saved to: {output_jsonl_path}")
        
        return output_jsonl_path
