from dataclasses import dataclass
from typing import Optional, List
import yaml

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    
    # LoRA parameters
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    output_dir: str = "models/kazsim_model"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 2
    
    # WandB config
    wandb_project: str = "text_simplification"
    wandb_name: str = "kazsim_training"
    use_wandb: bool = True

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict