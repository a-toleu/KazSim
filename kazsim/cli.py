#!/usr/bin/env python
"""
KazSim CLI - Command line interface for text simplification
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from .model import SimplificationModel
from .inference import SimplificationInference
from .trainer import SimplificationTrainer
from .data_processing import DataProcessor
from .config import ModelConfig, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_parser():
    """Setup argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="KazSim - Text Simplification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplify text
  kazsim simplify --model path/to/model --text "Your complex text"
  
  # Train a model
  kazsim train --data train.csv --output models/my_model
  
  # Batch process
  kazsim batch --model path/to/model --input test.csv --output results.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simplify command
    simplify_parser = subparsers.add_parser('simplify', help='Simplify text')
    simplify_parser.add_argument('--model', required=True, help='Path to model')
    simplify_parser.add_argument('--text', required=True, help='Text to simplify')
    simplify_parser.add_argument('--instruction', default='Simplify the following text')
    simplify_parser.add_argument('--max-tokens', type=int, default=256)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', required=True, help='Training data CSV')
    train_parser.add_argument('--output', required=True, help='Output directory')
    train_parser.add_argument('--epochs', type=int, default=2)
    train_parser.add_argument('--batch-size', type=int, default=2)
    train_parser.add_argument('--lr', type=float, default=2e-4)
    train_parser.add_argument('--wandb', action='store_true')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process batch of texts')
    batch_parser.add_argument('--model', required=True, help='Path to model')
    batch_parser.add_argument('--input', required=True, help='Input CSV file')
    batch_parser.add_argument('--output', required=True, help='Output CSV file')
    batch_parser.add_argument('--column', default='Complex Sentence')
    
    return parser

def simplify_command(args):
    """Handle simplify command."""
    logger.info(f"Loading model from {args.model}")
    
    config = ModelConfig()
    model_handler = SimplificationModel(config)
    model, tokenizer = model_handler.load_trained_model(args.model)
    
    inference = SimplificationInference(model, tokenizer)
    
    logger.info("Simplifying text...")
    result = inference.simplify_text(
        text=args.text,
        instruction=args.instruction,
        max_new_tokens=args.max_tokens
    )
    
    print("\n" + "="*50)
    print("Original:", args.text)
    print("\nSimplified:", result)
    print("="*50 + "\n")

def train_command(args):
    """Handle train command."""
    import pandas as pd
    
    logger.info("Initializing model...")
    model_config = ModelConfig()
    training_config = TrainingConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        use_wandb=args.wandb
    )
    
    # Load model
    model_handler = SimplificationModel(model_config)
    model, tokenizer = model_handler.load_base_model()
    model = model_handler.apply_peft()
    
    # Prepare data
    logger.info(f"Loading data from {args.data}")
    data_processor = DataProcessor(tokenizer)
    train_dataset = data_processor.prepare_dataset_from_csv(args.data)
    
    # Train
    logger.info("Starting training...")
    trainer = SimplificationTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        train_dataset=train_dataset
    )
    trainer.train()
    
    # Save
    model_handler.save_model(f"{args.output}/adapter")
    logger.info(f"Model saved to {args.output}/adapter")

def batch_command(args):
    """Handle batch command."""
    import pandas as pd
    
    logger.info(f"Loading model from {args.model}")
    config = ModelConfig()
    model_handler = SimplificationModel(config)
    model, tokenizer = model_handler.load_trained_model(args.model)
    
    inference = SimplificationInference(model, tokenizer)
    
    # Load input file
    logger.info(f"Loading input from {args.input}")
    df = pd.read_csv(args.input)
    
    # Process
    results = []
    total = len(df)
    for idx, row in df.iterrows():
        logger.info(f"Processing {idx+1}/{total}")
        text = row[args.column]
        simplified = inference.simplify_text(text, stream=False)
        results.append({
            'Original': text,
            'Simplified': simplified
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'simplify':
            simplify_command(args)
        elif args.command == 'train':
            train_command(args)
        elif args.command == 'batch':
            batch_command(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()