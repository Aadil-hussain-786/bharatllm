"""
Bharat-3B Smart-Core: Data Configuration
=========================================
Settings for synthetic data generation, tokenizer, and data pipeline.
"""

import ml_collections


def get_data_config() -> ml_collections.ConfigDict:
    """Returns the default data pipeline configuration."""
    config = ml_collections.ConfigDict()

    # ========================================
    # Synthetic Data Engine
    # ========================================
    config.synthetic = ml_collections.ConfigDict()
    config.synthetic.total_tokens_target = 1_000_000_000_000  # 1T tokens
    config.synthetic.output_dir = "gs://bharat-3b-data/synthetic/"
    config.synthetic.batch_size = 100  # API calls per batch
    config.synthetic.max_concurrent_requests = 50
    config.synthetic.retry_attempts = 3
    config.synthetic.retry_delay_seconds = 5

    # Teacher API Configuration
    config.synthetic.teachers = ml_collections.ConfigDict()
    config.synthetic.teachers.gemini = ml_collections.ConfigDict()
    config.synthetic.teachers.gemini.model = "gemini-1.5-pro"
    config.synthetic.teachers.gemini.api_key_env = "GOOGLE_API_KEY"
    config.synthetic.teachers.gemini.max_output_tokens = 8192
    config.synthetic.teachers.gemini.temperature = 0.7

    config.synthetic.teachers.llama = ml_collections.ConfigDict()
    config.synthetic.teachers.llama.model = "llama-3.1-405b-reasoning"
    config.synthetic.teachers.llama.api_key_env = "GROQ_API_KEY"
    config.synthetic.teachers.llama.max_output_tokens = 4096
    config.synthetic.teachers.llama.temperature = 0.7

    config.synthetic.teachers.gpt4 = ml_collections.ConfigDict()
    config.synthetic.teachers.gpt4.model = "gpt-4-turbo"
    config.synthetic.teachers.gpt4.api_key_env = "OPENAI_API_KEY"
    config.synthetic.teachers.gpt4.max_output_tokens = 4096
    config.synthetic.teachers.gpt4.temperature = 0.7

    # Data Categories & Weights
    config.synthetic.categories = ml_collections.ConfigDict()
    config.synthetic.categories.math_reasoning = 0.25
    config.synthetic.categories.python_code = 0.20
    config.synthetic.categories.logical_fallacies = 0.15
    config.synthetic.categories.hindi_comprehension = 0.15
    config.synthetic.categories.science_explanations = 0.10
    config.synthetic.categories.conversation = 0.10
    config.synthetic.categories.creative_writing = 0.05

    # Gold Token Filtering
    config.synthetic.gold_filter = ml_collections.ConfigDict()
    config.synthetic.gold_filter.min_consensus = 2  # At least 2 teachers must agree
    config.synthetic.gold_filter.similarity_threshold = 0.85  # Cosine similarity
    config.synthetic.gold_filter.quality_score_min = 0.8  # Minimum quality score
    config.synthetic.gold_filter.save_rejected = True  # Save rejected for analysis

    # ========================================
    # Tokenizer Configuration
    # ========================================
    config.tokenizer = ml_collections.ConfigDict()
    config.tokenizer.vocab_size = 50_257
    config.tokenizer.algorithm = "bpe"  # Byte-Pair Encoding
    config.tokenizer.model_type = "unigram"  # SentencePiece model type
    config.tokenizer.character_coverage = 0.9999  # Cover 99.99% of characters
    config.tokenizer.num_sub_iterations = 2

    # Special Tokens
    config.tokenizer.special_tokens = [
        "<pad>",    # 0
        "<bos>",    # 1
        "<eos>",    # 2
        "<unk>",    # 3
        "<mask>",   # 4
        "<sep>",    # 5
        "<cls>",    # 6
        "<mem>",    # 7 - Memory token for RMT
        "<think>",  # 8 - Chain of thought marker
        "</think>", # 9
    ]

    # Language Coverage
    config.tokenizer.languages = [
        "hindi",
        "english",
        "hinglish",
        "bengali",
        "tamil",
        "telugu",
        "marathi",
        "gujarati",
        "kannada",
        "malayalam",
        "punjabi",
        "odia",
    ]

    # Training Data for Tokenizer
    config.tokenizer.training_data = "gs://bharat-3b-data/tokenizer_corpus/"
    config.tokenizer.training_sentences = 10_000_000  # 10M sentences
    config.tokenizer.output_dir = "gs://bharat-3b-data/tokenizer/"

    # ========================================
    # Data Loading Pipeline
    # ========================================
    config.loader = ml_collections.ConfigDict()
    config.loader.num_workers = 8
    config.loader.prefetch_factor = 4
    config.loader.shuffle_buffer_size = 100_000
    config.loader.seed = 42
    config.loader.data_dir = "gs://bharat-3b-data/processed/"
    config.loader.file_format = "tfrecord"  # Options: "tfrecord", "jsonl", "parquet"

    return config
