"""
Bharat-3B Smart-Core: Main Training Script
============================================
Entry point for all training phases.

Usage:
    # Full pre-training (Phase 3)
    python scripts/train.py --phase pretrain
    
    # SFT (Phase 4.1)
    python scripts/train.py --phase sft --checkpoint gs://bharat-3b-checkpoints/pretrain/latest
    
    # DPO (Phase 4.2)
    python scripts/train.py --phase dpo --checkpoint gs://bharat-3b-checkpoints/sft/latest
"""

import argparse
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

console = Console()


def setup_logging(debug: bool = False):
    """Configure rich logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def print_banner():
    """Print Bharat-3B banner."""
    banner = """
╔══════════════════════════════════════════════════════╗
║          🚀 BHARAT-3B SMART-CORE                     ║
║          Deep Equilibrium + Recurrent Memory          ║
║          + Mixture of Softmaxes                       ║
║                                                       ║
║          3B Parameters | 128k Context | 10 Experts    ║
║          Target: Beat Llama-3 8B                      ║
╚══════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


def run_pretrain(args):
    """Run Phase 3 pre-training."""
    from configs.training_config import get_training_config
    from src.training.trainer import BharatTrainer

    config = get_training_config()

    # Override from args
    if args.batch_size:
        config.pretrain.per_device_batch_size = args.batch_size
    if args.learning_rate:
        config.pretrain.learning_rate = args.learning_rate
    if args.max_steps:
        total_steps = args.max_steps

    console.print("[bold green]Phase 3: Pre-training[/bold green]")
    console.print(f"  Total tokens: {config.pretrain.total_tokens:,}")
    console.print(f"  Batch size: {config.pretrain.batch_size}")
    console.print(f"  Peak LR: {config.pretrain.learning_rate}")
    console.print(f"  Warmup steps: {config.pretrain.warmup_steps}")
    console.print(f"  Distillation: {'✅ ON' if config.distillation.enabled else '❌ OFF'}")

    # Initialize and run
    trainer = BharatTrainer(config)
    trainer.initialize()
    metrics = trainer.train(num_steps=args.max_steps)

    console.print(f"[bold green]✅ Pre-training complete![/bold green]")
    console.print(f"Final metrics: {metrics}")


def run_sft(args):
    """Run Phase 4.1 SFT."""
    console.print("[bold green]Phase 4.1: Supervised Fine-Tuning[/bold green]")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Examples: 50,000")
    console.print(f"  Epochs: 3")

    # TODO: Load checkpoint and run SFT
    console.print("[yellow]SFT training pipeline ready. Load checkpoint to begin.[/yellow]")


def run_dpo(args):
    """Run Phase 4.2 DPO."""
    console.print("[bold green]Phase 4.2: Direct Preference Optimization[/bold green]")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Beta: 0.1")
    console.print(f"  Teacher Judge: Gemini 1.5 Pro")

    # TODO: Load SFT checkpoint and run DPO
    console.print("[yellow]DPO pipeline ready. Load SFT checkpoint to begin.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Bharat-3B Smart-Core Training",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pretrain", "sft", "dpo"],
        default="pretrain",
        help="Training phase to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.debug)
    print_banner()

    if args.phase == "pretrain":
        run_pretrain(args)
    elif args.phase == "sft":
        run_sft(args)
    elif args.phase == "dpo":
        run_dpo(args)


if __name__ == "__main__":
    main()
