import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
from .config import TrainingConfig
from typing import Optional

class SimplificationTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        train_dataset: Dataset
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.trainer = None
    
    def setup_wandb(self):
        """Initialize WandB tracking."""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                config={
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.num_train_epochs,
                    "batch_size": self.config.per_device_train_batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                }
            )
    
    def prepare_training_args(self) -> TrainingArguments:
        """Prepare training arguments."""
        return TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.config.output_dir,
            report_to="wandb" if self.config.use_wandb else "none",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
        )
    
    def setup_trainer(self):
        """Initialize the SFTTrainer."""
        training_args = self.prepare_training_args()
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        return self.trainer
    
    def train(self, resume_from_checkpoint: bool = False):
        """Run training."""
        if self.trainer is None:
            self.setup_trainer()
        
        self.setup_wandb()
        trainer_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        if self.config.use_wandb:
            wandb.finish()
        
        return trainer_stats