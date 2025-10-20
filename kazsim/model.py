from unsloth import FastLanguageModel
import torch
from typing import Optional, Tuple
from .config import ModelConfig

class SimplificationModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self) -> Tuple:
        """Load the base model and tokenizer."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        return self.model, self.tokenizer
    
    def load_trained_model(self, model_path: str) -> Tuple:
        """Load a trained model from path."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        return self.model, self.tokenizer
    
    def apply_peft(self):
        """Apply PEFT (LoRA) to the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model
    
    def save_model(self, output_path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_pretrained(output_path)
        print(f"✅ Model saved to {output_path}")