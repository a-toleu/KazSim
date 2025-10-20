import pytest
import torch
from kazsim.model import SimplificationModel
from kazsim.config import ModelConfig

def test_model_initialization():
    """Test model initialization."""
    config = ModelConfig()
    model_handler = SimplificationModel(config)
    assert model_handler.model is None
    assert model_handler.tokenizer is None

def test_model_loading():
    """Test model loading (requires model files)."""
    config = ModelConfig()
    model_handler = SimplificationModel(config)
    
    # This would require actual model files
    # model, tokenizer = model_handler.load_base_model()
    # assert model is not None
    # assert tokenizer is not None

@pytest.mark.parametrize("lora_r", [8, 16, 32, 64])
def test_lora_configs(lora_r):
    """Test different LoRA configurations."""
    config = ModelConfig(lora_r=lora_r)
    assert config.lora_r == lora_r