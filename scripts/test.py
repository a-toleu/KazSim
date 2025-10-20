#!/usr/bin/env python
import argparse
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazsim import SimplificationModel, SimplificationInference
from kazsim.config import ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Test text simplification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input", type=str, help="Text to simplify")
    parser.add_argument("--test_file", type=str, help="CSV file with test data")
    parser.add_argument("--complex_col", type=str, default="Complex Sentence")
    parser.add_argument("--output_file", type=str, help="Output CSV file for results")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model_config = ModelConfig()
    model_handler = SimplificationModel(model_config)
    model, tokenizer = model_handler.load_trained_model(args.model_path)
    
    # Initialize inference
    inference = SimplificationInference(model, tokenizer)
    
    if args.input:
        # Simplify single text
        print("Original text:", args.input)
        print("\nSimplifying...")
        result = inference.simplify_text(args.input, stream=args.stream)
        if not args.stream:
            print("\nSimplified text:", result)
    
    elif args.test_file:
        # Process test file
        print(f"Processing test file: {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        
        results = []
        for idx, row in test_df.iterrows():
            complex_text = row[args.complex_col]
            print(f"\nProcessing {idx+1}/{len(test_df)}...")
            simplified = inference.simplify_text(complex_text, stream=False)
            results.append({
                "Original": complex_text,
                "Simplified": simplified
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        if args.output_file:
            results_df.to_csv(args.output_file, index=False)
            print(f"\nResults saved to {args.output_file}")
        else:
            print("\nResults:")
            print(results_df.head())
    
    else:
        print("Please provide either --input or --test_file")

if __name__ == "__main__":
    main()