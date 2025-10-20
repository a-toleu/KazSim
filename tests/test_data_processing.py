import pytest
import pandas as pd
from kazsim.data_processing import DataProcessor
from transformers import AutoTokenizer
import tempfile

def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use small model for testing
    processor = DataProcessor(tokenizer)
    assert processor.tokenizer is not None
    assert len(processor.instructions) > 0

def test_csv_processing():
    """Test CSV file processing."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Complex Sentence,Simple Sentence\n")
        f.write("This is complex.,This is simple.\n")
        temp_path = f.name
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    processor = DataProcessor(tokenizer)
    
    # Test processing
    # dataset = processor.prepare_dataset_from_csv(temp_path)
    # assert len(dataset) > 0