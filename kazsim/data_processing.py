import pandas as pd
import json
import random
from datasets import Dataset, load_dataset
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
        
        self.instructions = [
            "Simplify the following text.",
            "Rewrite this in simpler words.",
            "Paraphrase this text in an easier way.",
            "Make this text easier to understand.",
            "Reword this sentence using simple language.",
        ]
    
    def format_prompts(self, examples: Dict) -> Dict:
        """Format examples for training."""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = self.prompt_template.format(
                instruction, input_text, output
            ) + self.tokenizer.eos_token
            texts.append(text)
        
        return {"text": texts}
    
    def prepare_dataset_from_csv(
        self, 
        csv_path: str,
        complex_col: str = "Complex Sentence",
        simple_col: str = "Simple Sentence"
    ) -> Dataset:
        """Load and prepare dataset from CSV file."""
        df = pd.read_csv(csv_path)
        
        dataset_dict = {
            "instruction": [random.choice(self.instructions) for _ in range(len(df))],
            "input": df[complex_col].tolist(),
            "output": df[simple_col].tolist(),
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        formatted_dataset = dataset.map(self.format_prompts, batched=True)
        
        return formatted_dataset
    
    def save_to_jsonl(self, dataset: Dataset, output_path: str):
        """Save dataset to JSONL format."""
        dataset.to_json(output_path)
        print(f"✅ Dataset saved to {output_path}")
    
    def load_from_jsonl(self, jsonl_path: str) -> Dataset:
        """Load dataset from JSONL file."""
        return load_dataset("json", data_files=jsonl_path, split="train")