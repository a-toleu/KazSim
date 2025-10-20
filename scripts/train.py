#!/usr/bin/env python
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazsim import SimplificationModel, SimplificationTrainer, DataProcessor
from kazsim.config import ModelConfig, TrainingConfig, load_config

def main():
    parser = argparse.ArgumentParser(description="Train text simplification model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="models/kazsim_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    if args.config:
        config_dict = load_config(args.config)
        # Update configs from file
    
    # Initialize model
    print("Loading model...")
    model_handler = SimplificationModel(model_config)
    model, tokenizer = model_handler.load_base_model()
    model = model_handler.apply_peft()
    
    # Prepare data
    print("Preparing dataset...")
    data_processor = DataProcessor(tokenizer)
    train_dataset = data_processor.prepare_dataset_from_csv(args.data_path)
    
    # Optional: Save processed data
    data_processor.save_to_jsonl(
        train_dataset, 
        f"{args.output_dir}/train_data.jsonl"
    )
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = SimplificationTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        train_dataset=train_dataset
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    model_handler.save_model(f"{args.output_dir}/adapter")
    print("Training complete!")

if __name__ == "__main__":
    main()