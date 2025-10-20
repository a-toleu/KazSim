#!/usr/bin/env python
import argparse
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kazsim import DataProcessor
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--complex_col", type=str, default="Complex Sentence")
    parser.add_argument("--simple_col", type=str, default="Simple Sentence")
    
    args = parser.parse_args()
    
    # Load a tokenizer (just for EOS token)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Process data
    processor = DataProcessor(tokenizer)
    dataset = processor.prepare_dataset_from_csv(
        args.input_csv,
        args.complex_col,
        args.simple_col
    )
    
    # Save to JSONL
    processor.save_to_jsonl(dataset, args.output_jsonl)

if __name__ == "__main__":
    main()