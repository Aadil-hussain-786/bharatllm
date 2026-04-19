"""
Bharat-3B Smart-Core: Synthetic Data Generation Script
========================================================
Phase 1, Step 1.2: Generate 1T tokens of reasoning-heavy data.

Usage:
    python scripts/generate_data.py --category math_reasoning --num-samples 10000
    python scripts/generate_data.py --category python_code --num-samples 10000
    python scripts/generate_data.py --all --output-dir data/synthetic/
"""

import argparse
import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


# Prompt templates per category
CATEGORY_PROMPTS = {
    "math_reasoning": [
        "Solve step by step: If a train travels at 60 km/h for 2 hours, then at 90 km/h for 3 hours, what is the average speed for the entire journey?",
        "Prove that the sum of first n natural numbers is n(n+1)/2 using mathematical induction.",
        "A shopkeeper gives 20% discount on MRP and still makes 25% profit. If the cost price is ₹800, what is the MRP?",
        "Find the probability that in a group of 23 people, at least two share the same birthday.",
        "Solve the system of equations: 3x + 2y = 12, 5x - 3y = 1",
    ],
    "python_code": [
        "Write a Python function to find the longest common subsequence of two strings. Include type hints and docstring.",
        "Implement a binary search tree with insert, delete, and search operations in Python.",
        "Write a Python decorator that caches function results with a TTL (time-to-live) expiry.",
        "Debug this code: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)",
        "Write a Python generator that yields prime numbers indefinitely using the Sieve of Eratosthenes.",
    ],
    "logical_fallacies": [
        "Identify the logical fallacy: 'Everyone is buying this phone, so it must be the best.'",
        "Analyze this argument for validity: 'All birds can fly. Penguins are birds. Therefore, penguins can fly.'",
        "Is this a strawman fallacy? 'You want to reduce military spending? So you want our country to be defenseless?'",
        "Construct a sound deductive argument about the relationship between education and income.",
        "What is wrong with the argument: 'My grandfather smoked all his life and lived to 95, so smoking isn't harmful.'",
    ],
    "hindi_comprehension": [
        "निम्नलिखित विषय पर एक विस्तृत निबंध लिखें: 'भारत में डिजिटल क्रांति और इसका समाज पर प्रभाव'",
        "इस कविता का भावार्थ समझाएं: 'हिमालय के आँगन में उसे प्रथम किरण का दे उपहार, उषा ने हँस कर कहा कि जागो, इस सोने के संसार'",
        "'कृत्रिम बुद्धिमत्ता' (AI) पर एक Hindi article लिखें जो general public को समझ आए।",
        "भारत की शिक्षा प्रणाली में सुधार के लिए 10 सुझाव दीजिए।",
        "'स्वच्छ भारत अभियान' की सफलताओं और चुनौतियों पर चर्चा करें।",
    ],
    "science_explanations": [
        "Explain quantum entanglement in simple terms. Include an analogy that a 10-year-old would understand.",
        "How does CRISPR gene editing work? Explain the mechanism step by step.",
        "Why is the sky blue? Explain using Rayleigh scattering with mathematical detail.",
        "What causes black holes to form? Explain the Chandrasekhar limit.",
        "How do mRNA vaccines work? Explain the immune response they trigger.",
    ],
}


async def generate_category(
    category: str,
    num_samples: int,
    output_dir: str,
    max_concurrent: int = 10,
):
    """Generate synthetic data for a specific category."""
    from src.data.synthetic_engine import SyntheticDataEngine

    engine = SyntheticDataEngine(
        consensus_threshold=2,
        similarity_threshold=0.85,
        quality_score_min=0.8,
    )

    prompts = CATEGORY_PROMPTS.get(category, [])
    if not prompts:
        console.print(f"[red]Unknown category: {category}[/red]")
        return

    # Expand prompts to reach num_samples
    expanded_prompts = []
    while len(expanded_prompts) < num_samples:
        expanded_prompts.extend(prompts)
    expanded_prompts = expanded_prompts[:num_samples]

    console.print(f"[cyan]Generating {num_samples} samples for category: {category}[/cyan]")

    # Detect language
    language = "hindi" if category == "hindi_comprehension" else "english"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(f"Generating {category}...", total=num_samples)

        batch_size = min(max_concurrent, num_samples)
        for i in range(0, num_samples, batch_size):
            batch_prompts = expanded_prompts[i:i+batch_size]
            samples = await engine.generate_batch(
                category=category,
                prompts=batch_prompts,
                language=language,
                max_concurrent=max_concurrent,
            )

            # Save batch
            category_dir = os.path.join(output_dir, category)
            engine.save_samples(samples, category_dir)

            progress.update(task, advance=len(batch_prompts))

    stats = engine.get_stats()
    console.print(f"[green]✅ Done! Stats: {stats}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Bharat-3B Synthetic Data Generation")
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORY_PROMPTS.keys()),
        help="Data category to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all categories",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per category",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic/",
        help="Output directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent API requests",
    )

    args = parser.parse_args()

    if args.all:
        categories = list(CATEGORY_PROMPTS.keys())
    elif args.category:
        categories = [args.category]
    else:
        parser.error("Specify --category or --all")

    for category in categories:
        asyncio.run(generate_category(
            category=category,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
        ))

    console.print("[bold green]🎉 All data generation complete![/bold green]")


if __name__ == "__main__":
    main()
